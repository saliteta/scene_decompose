import torch
from gsplat_ext import Dataset, Parser
from gsplat.distributed import cli
import os
from tqdm import tqdm
import torch.nn.functional as F
from gsplat_ext import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from argparser import DataArgs, DistillArgs, ModelArgs, parse_args
import torchvision.transforms as T
from PIL import Image
import open_clip

class OpenCLIPRunner:
    """Engine for training and testing with OpenCLIP features."""

    def __init__(self, data_args: DataArgs, distill_args: DistillArgs, model_args: ModelArgs) -> None:
        self.data_args = data_args
        self.distill_args = distill_args
        self.device = f"cuda"

        self.set_dataset(data_args)
        self.set_splat(distill_args)  # set the splat primitive and renderer
        self.set_model(model_args)

    def set_dataset(self, data_args: DataArgs):
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=data_args.dir,
            factor=data_args.factor,
            test_every=data_args.test_every,
        )
        # We are going to generate features on the fly
        trainset = Dataset(
            self.parser,
            split="train",
        )
        valset = Dataset(self.parser, split="val")

        self.trainLoader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False
        )
        self.valLoader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False
        )

        self.scene_scale = self.parser.scene_scale * 1.1

    def set_splat(self, distill_args: DistillArgs):
        if distill_args.method == "2DGS":
            self.splats = GaussianPrimitive2D()
            self.splats.from_file(distill_args.ckpt)
            self.renderer = GaussianRenderer2D(self.splats)
        elif distill_args.method == "3DGS":
            self.splats = GaussianPrimitive()
            self.splats.from_file(distill_args.ckpt)
            self.renderer = GaussianRenderer(self.splats)
        elif distill_args.method == "DBS":
            self.splats = BetaSplatPrimitive()
            self.splats.from_file(distill_args.ckpt)
            self.renderer = BetaSplatRenderer(self.splats)
        else:
            raise ValueError(f"Invalid splat method: {distill_args.method}")
        
        # Optional filtering
        #filter_mask = self.splats.filtering_alpha(percentage=0.5)
        #self.splats.mask(filter_mask)
        #self.splats.to(self.device)

    @torch.no_grad()
    def set_model(self, model_args: ModelArgs):
        """
        Initialize OpenCLIP model and preprocessing.
        """
        print("Loading OpenCLIP model...")
        
        # Load OpenCLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', 
            pretrained='laion2b_s34b_b88k',
            device=self.device
        )
        self.model.eval()
        
        # Get tokenizer for potential text features
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        
        print(f"✓ Loaded OpenCLIP ViT-B-16 model")
        print(f"✓ Model device: {next(self.model.parameters()).device}")
    
    @torch.no_grad()
    def get_features(self, image: torch.Tensor, target_size: tuple = (224, 224)):   
        """
        Extract global image embedding from OpenCLIP and broadcast to spatial resolution.
        
        Args:
            image: Input image tensor [B, H, W, C] in [0, 255] range
            target_size: Target spatial size (H, W) for feature map
            
        Returns:
            features: Spatial feature map [B, H, W, C] where each spatial location 
                     has the same global image embedding
        """
        # Convert from [B, H, W, C] to [B, C, H, W] and normalize to [0, 1]
        image = image.permute(0, 3, 1, 2) / 255.0
        
        # Apply OpenCLIP preprocessing
        # Note: preprocess expects PIL image, but we can manually apply the transforms
        # OpenCLIP typically uses: Resize(224), CenterCrop(224), Normalize(CLIP_MEAN, CLIP_STD)
        
        # Resize to 224x224 (OpenCLIP standard input size)
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Apply CLIP normalization
        # CLIP uses ImageNet normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        image_normalized = (image_resized - clip_mean) / clip_std
        
        # Extract global image embedding
        with torch.autocast("cuda"):
            image_features = self.model.encode_image(image_normalized)  # [B, feature_dim]
            
        # Normalize features (standard practice for CLIP)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Get feature dimension
        feature_dim = image_features.shape[-1]  # Typically 512 for ViT-B-32
        
        # Broadcast global embedding to spatial resolution
        # [B, feature_dim] -> [B, feature_dim, H, W] -> [B, H, W, feature_dim]
        H, W = target_size
        spatial_features = image_features.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
        spatial_features = spatial_features.expand(-1, -1, H, W)       # [B, feature_dim, H, W]
        spatial_features = spatial_features.permute(0, 2, 3, 1).to(torch.float32)       # [B, H, W, feature_dim]
        
        
        return spatial_features

    def rescale_ks(self, Ks: torch.Tensor, width, height, width_new, height_new):
        """Rescale camera intrinsics for new image dimensions."""
        Ks_new = Ks.clone()
        Ks_new[0, 0] *= width_new / width
        Ks_new[1, 1] *= height_new / height
        Ks_new[0, 2] = width_new / 2
        Ks_new[1, 2] = height_new / 2
        return Ks_new

    @torch.no_grad()
    def distill(self, distill_args: DistillArgs, data_args: DataArgs):
        """Entry for distillation using OpenCLIP features."""
        print("Running OpenCLIP feature distillation...")
        means = self.splats.geometry["means"]
        
        # OpenCLIP ViT-B-32 produces 512-dimensional features
        feature_dim = 512
        target_spatial_size = (224, 224)  # Standard size for backprojection
        
        self.splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=self.device
        )
        self.splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=self.device
        )

        print(f"Initialized splat features: {self.splat_features.shape}")
        print(f"Target spatial size: {target_spatial_size}")

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Distilling OpenCLIP features",
            total=len(self.trainLoader),
        ):
            if i in [0, 300, 600]:
                print(data["image_name"])
                continue
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]
            
            # Extract OpenCLIP features (global embedding broadcasted to spatial)
            features = self.get_features(pixels, target_size=target_spatial_size)  # [1, H, W, 512]
            
            # Rescale camera intrinsics for the target spatial size
            Ks_rescaled = self.rescale_ks(
                Ks.squeeze(0), width, height, 
                target_spatial_size[1], target_spatial_size[0]  # (W, H)
            ).unsqueeze(0)

            # Backproject features to Gaussian splats
            (
                splat_features_per_image,
                splat_weights_per_image,
                ids,
            ) = self.renderer.inverse_render(
                K=Ks_rescaled,
                extrinsic=camtoworlds,
                width=features.shape[2],   # W
                height=features.shape[1],  # H
                features=features,
            )

            # Accumulate features
            self.splat_features[ids] += splat_features_per_image[:, :feature_dim]
            self.splat_weights[ids] += splat_weights_per_image
            # Clean up GPU memory
            del splat_features_per_image, splat_weights_per_image, ids, features
            torch.cuda.empty_cache()

        # Normalize accumulated features by weights
        print("Normalizing accumulated features...")
        del self.model
        torch.cuda.empty_cache()
        self.splat_features /= self.splat_weights[..., None]
        self.splat_features = torch.nan_to_num(self.splat_features.cpu(), nan=0.0)

        # Clean up
        del self.splat_weights
        torch.cuda.empty_cache()

        # Save features
        self.basename, _ = os.path.splitext(distill_args.ckpt)
        output_path = self.basename + "_openclip_features.pt"
        
        self.splat_features = self.splat_features.cpu()
        torch.save(self.splat_features, output_path)
        
        print(f"✓ Saved OpenCLIP features to: {output_path}")
        print(f"✓ Feature shape: {self.splat_features.shape}")
        print(f"✓ Feature statistics:")
        print(f"    Mean: {self.splat_features.mean():.6f}")
        print(f"    Std: {self.splat_features.std():.6f}")
        print(f"    Min: {self.splat_features.min():.6f}")
        print(f"    Max: {self.splat_features.max():.6f}")

    @torch.no_grad()
    def extract_text_features(self, text_queries: list):
        """
        Extract text features for given queries (useful for semantic matching).
        
        Args:
            text_queries: List of text strings
            
        Returns:
            text_features: Normalized text embeddings [N, feature_dim]
        """
        text_tokens = self.tokenizer(text_queries)
        
        with torch.autocast("cuda"):
            text_features = self.model.encode_text(text_tokens.to(self.device))
            
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features


def main(local_rank: int, world_rank, world_size: int, args):
    data_args, distill_args, model_args = args

    runner = OpenCLIPRunner(data_args, distill_args, model_args)
    runner.distill(distill_args, data_args)


if __name__ == "__main__":
    data_args, distill_args, model_args = parse_args()
    cli(main, (data_args, distill_args, model_args))
