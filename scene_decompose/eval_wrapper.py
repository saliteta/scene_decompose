import torch
from gsplat_ext import Dataset, Parser
from gsplat.distributed import cli
import os
from tqdm import tqdm
import torch.nn.functional as F
import csv
from pathlib import Path
from gsplat_ext import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from hierachical_utils.hierachical_primitive import HierachicalPrimitive
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from argparser import DataArgs, DistillArgs, ModelArgs, parse_args
import torchvision.transforms as T
from general_wrapper.feature_extractor import FeatureExtractorOpenCLIP, FeatureExtractorConfig, FeatureExtractorJAFAR
from query_system.querySystem import FeatureQuerySystem, FeatureDatabase
from hierachical_utils.image_dashboard import  FeatureProcessor

FEATURE_MAX_DIM = 512

class Runner:
    """Engine for training and testing."""

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
            self.splats.from_file(distill_args.ckpt, feature_path=distill_args.feature_ckpt)
            self.renderer = GaussianRenderer2D(self.splats)
        elif distill_args.method == "3DGS":
            self.splats = GaussianPrimitive()
            self.splats.from_file(distill_args.ckpt, feature_path=distill_args.feature_ckpt)
            self.renderer = GaussianRenderer(self.splats)
        elif distill_args.method == "DBS":
            self.splats = BetaSplatPrimitive()
            self.splats.from_file(distill_args.ckpt, feature_path=distill_args.feature_ckpt)
            self.renderer = BetaSplatRenderer(self.splats)
        elif distill_args.method == "H_GS":
            self.splats = HierachicalPrimitive(with_feature=True)  # Initialize with feature support
            self.splats.load_from_file(distill_args.ckpt)
            self.renderer: GaussianRenderer = self.splats.get_renderer(0)
            # Check if feature key exists after loading
            if "feature" not in self.splats.source:
                raise ValueError("Loaded HierachicalPrimitive does not contain 'feature' key. Ensure the file was saved with features.")
            database = FeatureDatabase(num_layers=len(self.splats.source["feature"]), features=self.splats.source["feature"])
            self.query_system = FeatureQuerySystem(database)
            self.feature_extractor = FeatureProcessor(backbone=model_args.backbone, model_path=model_args.model_path, device=self.device)
        else:
            raise ValueError(f"Invalid splat method: {distill_args.method}")

    @torch.no_grad()
    def set_model(self, model_args: ModelArgs):
        """
            We add on the fly feature generation pipeline for current experiments
            We assume that we are actually using the dinov3 model
        """
        config = FeatureExtractorConfig(model_type=model_args.model_type, 
                                        model_architecture=model_args.backbone, 
                                        model_path=model_args.model_path,
                                        device=self.device)
        if model_args.model_type == "jafar":
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractorJAFAR(config)
        elif model_args.model_type == "openclip":
                self.feature_extractor = FeatureExtractorOpenCLIP(config)
        else:
            raise ValueError(f"Invalid model type: {model_args.model_type}")
    
    @torch.no_grad()
    def get_features(self, image: torch.Tensor) -> torch.Tensor: # [B, H, W, 3] -> [B, H', W', C]
        """
            Feature extraction pipeline (supports JAFAR and OpenCLIP)
            Input: [B, H, W, 3] tensor with pixel values in [0, 255]
            Output: [B, H', W', C] tensor
        """
        if self.distill_args.method != "H_GS":
            # extract_features expects (H, W, 3) format, so handle batch dimension
            if image.dim() == 4:
                # If batched, process each image separately or squeeze if B=1
                if image.shape[0] == 1:
                    image = image.squeeze(0)  # [H, W, 3]
                else:
                    # For multiple images, process first one or raise error
                    image = image[0]  # Take first image
            features = self.feature_extractor.extract_features(image)  # (H, W, C)
            features = features.mean(dim=(0, 1))  # (C,) - mean over spatial dimensions H and W
            features = F.normalize(features, p=2, dim=0)  # Normalize to unit vector (L2 norm)
        else:
            features, _, _, _ = self.feature_extractor.quantized_feature(image)
        return features


    def rescale_ks(self, Ks: torch.Tensor, width, height, width_new, height_new):
        Ks[0, 0] *= width_new / width
        Ks[1, 1] *= height_new / height
        Ks[0, 2] = width_new / 2
        Ks[1, 2] = height_new / 2
        return Ks


    def get_feature_dim(self):
        data = self.trainLoader.dataset[0]
        features = self.get_features(data["image"].cuda().unsqueeze(0))
        return features.shape[-1]

    @torch.no_grad()
    def distill(self, distill_args: DistillArgs, data_args: DataArgs):
        """Entry for distillation."""
        print("Running distillation...")
        
        # Initialize list to store similarity results
        results = []

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Computing similarity",
            total=len(self.trainLoader),
        ):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]
            if self.distill_args.method != "H_GS":
                query_features = self.get_features(pixels)[0]
            else:
                query_features = self.get_features(pixels)

            Ks = self.rescale_ks(Ks.squeeze(0), width, height, 448, 448).unsqueeze(0)
            # query features all set

            dummy_attention = torch.ones(448, 448, 1).cuda()

            if self.distill_args.method != "H_GS":
                query_attention_scores = query_features @ self.splats.feature.T
            else:
                query_attention_scores = self.query_system.query(query_features)
                query_attention_scores = self.splats.propagate_feature(query_attention_scores.unsqueeze(-1))[-1].squeeze(-1)

            (
                splat_features_per_image,
                splat_weights_per_image,
                ids,
            ) = self.renderer.inverse_render(
                K=Ks,
                extrinsic=camtoworlds,
                width=448,
                height=448,
                features=dummy_attention.unsqueeze(0),
            )
            eps = torch.finfo(splat_features_per_image.dtype).eps
            splat_features_per_image /= splat_weights_per_image[..., None] + eps
            splat_features_per_image = splat_features_per_image.unsqueeze(-1)
            
            # Get number of Gaussians
            N = self.renderer.primitives.geometry["means"].shape[0]
            
            # Initialize attention scores with -1 for all Gaussians
            gt_attention_scores = torch.full((N,), -1.0, device="cuda", dtype=splat_features_per_image.dtype)
            
            # Replace values for splats that appear in the image (have valid ids)
            # ids contains the indices of splats visible in this camera view
            if ids.numel() > 0:  # Check if there are any visible splats
                valid_mask = ids >= 0  # Filter out invalid IDs (typically -1)
                valid_ids = ids[valid_mask]
                valid_features = splat_features_per_image[valid_mask].flatten()
                
                # Assign attention scores to the corresponding splat indices
                gt_attention_scores[valid_ids] = valid_features
            
            similarity = self.distribution_similarity(query_attention_scores, gt_attention_scores)
            precision_at_10 = self.precision_at(query_attention_scores, gt_attention_scores, at=10)
            recall_at_10 = self.recall_at(query_attention_scores, gt_attention_scores, at=10)
            
            # Store results
            image_name = data.get("image_name", f"image_{i}")
            image_id = data.get("image_id", i)
            results.append({
                "image_index": i,
                "image_id": image_id,
                "image_name": image_name,
                "similarity": similarity.item() if isinstance(similarity, torch.Tensor) else float(similarity),
                "precision_at_10": precision_at_10.item() if isinstance(precision_at_10, torch.Tensor) else float(precision_at_10),
                "recall_at_10": recall_at_10.item() if isinstance(recall_at_10, torch.Tensor) else float(recall_at_10)
            })
        
        # Save results to CSV
        output_csv_path = Path(distill_args.ckpt).parent / "metrics.csv"
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_index', 'image_id', 'image_name', 'similarity', 'precision_at_10', 'recall_at_10']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nSimilarity results saved to: {output_csv_path}")
        print(f"Total images processed: {len(results)}")
        print(f"Average similarity: {sum(r['similarity'] for r in results) / len(results):.4f}")
        print(f"Average precision at 10: {sum(r['precision_at_10'] for r in results) / len(results):.4f}")
        print(f"Average recall at 10: {sum(r['recall_at_10'] for r in results) / len(results):.4f}")
        
        
        
        
    def distribution_similarity(self, query_attention_scores: torch.Tensor, gt_attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distribution similarity between query and gt attention scores.
        Rescales both vectors to [-1, 1] range using min-max scaling, then computes similarity.
        
        Args:
            query_attention_scores: torch.Tensor of shape (N,)
            gt_attention_scores: torch.Tensor of shape (N,)
        Returns:
            similarity: torch.Tensor - similarity divided by length of gt_normalized
        """
        # Rescale query_attention_scores to [-1, 1] range
        query_min = query_attention_scores.min()
        query_max = query_attention_scores.max()
        if query_max != query_min:
            query_normalized = 2.0 * (query_attention_scores - query_min) / (query_max - query_min) - 1.0
        else:
            query_normalized = torch.zeros_like(query_attention_scores)
        
        # Rescale gt_attention_scores to [-1, 1] range
        gt_min = gt_attention_scores.min()
        gt_max = gt_attention_scores.max()
        if gt_max != gt_min:
            gt_normalized = 2.0 * (gt_attention_scores - gt_min) / (gt_max - gt_min) - 1.0
        else:
            gt_normalized = torch.zeros_like(gt_attention_scores)
        
        # Calculate similarity (dot product)
        similarity = (query_normalized * gt_normalized).sum()
        
        # Divide by the length (L2 norm) of gt_normalized
        gt_length = len(gt_normalized)
        if gt_length > 0:
            similarity = similarity / gt_length
        else:
            similarity = torch.tensor(0.0, device=similarity.device, dtype=similarity.dtype)
        similarity = (similarity + 1.0) / 2.0
        return similarity


    def precision_at(self, query_attention_scores: torch.Tensor, gt_attention_scores: torch.Tensor, at = 10) -> torch.Tensor:
        """
        Calculate the true positive rate between query and gt attention scores.
        """
        _, top_indices = torch.topk(query_attention_scores, k=int(at/100*query_attention_scores.shape[0]), dim=0)
        query_attention_scores = torch.zeros_like(query_attention_scores)
        query_attention_scores[top_indices] = 1.0
        
        # Convert gt_attention_scores: values < 0 -> 0, values >= 0 -> 1
        gt_attention_scores_binary = torch.where(gt_attention_scores < 0, 
                                                  torch.tensor(0.0, device=gt_attention_scores.device, dtype=gt_attention_scores.dtype),
                                                  torch.tensor(1.0, device=gt_attention_scores.device, dtype=gt_attention_scores.dtype))
        
        return (query_attention_scores * gt_attention_scores_binary).sum() / query_attention_scores.sum()

    def recall_at(self, query_attention_scores: torch.Tensor, gt_attention_scores: torch.Tensor, at = 10) -> torch.Tensor:
        """
        Calculate the recall at a given threshold.
        """
        _, top_indices = torch.topk(query_attention_scores, k=int(at/100*query_attention_scores.shape[0]), dim=0)
        query_attention_scores = torch.zeros_like(query_attention_scores)
        query_attention_scores[top_indices] = 1.0
        gt_attention_scores_binary = torch.where(gt_attention_scores < 0, 
                                                  torch.tensor(0.0, device=gt_attention_scores.device, dtype=gt_attention_scores.dtype),
                                                  torch.tensor(1.0, device=gt_attention_scores.device, dtype=gt_attention_scores.dtype))
        return (query_attention_scores * gt_attention_scores_binary).sum() / gt_attention_scores_binary.sum()

def main(local_rank: int, world_rank, world_size: int, args):
    data_args, distill_args, model_args = args

    runner = Runner(data_args, distill_args, model_args)
    runner.distill(distill_args, data_args)


if __name__ == "__main__":
    data_args, distill_args, model_args = parse_args()
    cli(main, (data_args, distill_args, model_args))
