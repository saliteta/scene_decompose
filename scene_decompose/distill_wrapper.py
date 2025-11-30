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
from jafar import load_model
from general_wrapper.feature_extractor import (
    FeatureExtractorOpenCLIP,
    FeatureExtractorConfig,
    FeatureExtractorJAFAR,
)

FEATURE_MAX_DIM = 512


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, data_args: DataArgs, distill_args: DistillArgs, model_args: ModelArgs
    ) -> None:

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

        # filter_mask = self.splats.filtering_alpha(percentage=0.5)
        # self.splats.mask(filter_mask)
        # self.splats.to(self.device)

    @torch.no_grad()
    def set_model(self, model_args: ModelArgs):
        """
        We add on the fly feature generation pipeline for current experiments
        We assume that we are actually using the dinov3 model
        """
        config = FeatureExtractorConfig(
            model_type=model_args.model_type,
            model_architecture=model_args.backbone,
            model_path=model_args.model_path,
            device=self.device,
        )
        if model_args.model_type == "jafar":
            self.feature_extractor = FeatureExtractorJAFAR(config)
        elif model_args.model_type == "openclip":
            self.feature_extractor = FeatureExtractorOpenCLIP(config)
        else:
            raise ValueError(f"Invalid model type: {model_args.model_type}")

    @torch.no_grad()
    def get_features(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # [B, H, W, 3] -> [B, H', W', C]
        """
        Feature extraction pipeline (supports JAFAR and OpenCLIP)
        Input: [B, H, W, 3] tensor with pixel values in [0, 255]
        Output: [B, H', W', C] tensor
        """
        # extract_features expects (H, W, 3) format, so handle batch dimension
        if image.dim() == 4:
            # If batched, process each image separately or squeeze if B=1
            if image.shape[0] == 1:
                image = image.squeeze(0)  # [H, W, 3]
            else:
                # For multiple images, process first one or raise error
                image = image[0]  # Take first image
        features = self.feature_extractor.extract_features(image)
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
        means = self.splats.geometry["means"]
        # we can deal with 2DGS and 3DGS differently

        feature_dim = self.get_feature_dim()
        self.splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=self.device
        )
        self.splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=self.device
        )

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Distilling features",
            total=len(self.trainLoader),
        ):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]
            features = self.get_features(pixels)[0]

            Ks = self.rescale_ks(
                Ks.squeeze(0), width, height, features.shape[0], features.shape[1]
            ).unsqueeze(0)
            #features = F.normalize(features, p=2, dim=-1).unsqueeze(0)

            # Process features in chunks if feature_dim > FEATURE_MAX_DIM
            if feature_dim > FEATURE_MAX_DIM:
                # Process features in chunks of FEATURE_MAX_DIM
                for start_idx in range(0, feature_dim, FEATURE_MAX_DIM):
                    end_idx = min(start_idx + FEATURE_MAX_DIM, feature_dim)
                    features_chunk = features[:, :, :, start_idx:end_idx]

                    (
                        splat_features_per_image,
                        splat_weights_per_image,
                        ids,
                    ) = self.renderer.inverse_render(
                        K=Ks,
                        extrinsic=camtoworlds,
                        width=features_chunk.shape[1],
                        height=features_chunk.shape[2],
                        features=features_chunk,
                    )

                    # Store the chunk in the appropriate feature dimension range
                    self.splat_features[
                        ids, start_idx:end_idx
                    ] += splat_features_per_image
                    if start_idx == 0:
                        self.splat_weights[ids] += splat_weights_per_image
                    del splat_features_per_image, splat_weights_per_image
                    torch.cuda.empty_cache()
            else:
                # Original processing for feature_dim <= FEATURE_MAX_DIM
                (
                    splat_features_per_image,
                    splat_weights_per_image,
                    ids,
                ) = self.renderer.inverse_render(
                    K=Ks,
                    extrinsic=camtoworlds,
                    width=features.shape[1],
                    height=features.shape[2],
                    features=features[:,:,:,:feature_dim],
                )

                self.splat_features[ids] += splat_features_per_image[:, :feature_dim]
                self.splat_weights[ids] += splat_weights_per_image
                del splat_features_per_image, splat_weights_per_image

            del ids, features
            torch.cuda.empty_cache()

        self.splat_features /= self.splat_weights[..., None]
        self.splat_features = torch.nan_to_num(self.splat_features, nan=0.0)

        del self.splat_weights
        torch.cuda.empty_cache()

        self.basename, _ = os.path.splitext(distill_args.ckpt)

        self.splat_features = self.splat_features.cpu()
        torch.save(self.splat_features, self.distill_args.feature_ckpt)


def main(local_rank: int, world_rank, world_size: int, args):
    data_args, distill_args, model_args = args

    runner = Runner(data_args, distill_args, model_args)
    runner.distill(distill_args, data_args)


if __name__ == "__main__":
    data_args, distill_args, model_args = parse_args()
    cli(main, (data_args, distill_args, model_args))
