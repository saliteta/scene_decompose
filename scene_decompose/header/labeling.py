import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import argparse

# Import the head class from classifier
from classifier import PatchClassifierHead


class FeatureLabeler:
    """
    A class to load the trained classification head and generate labels
    from n×c channel feature inputs.
    """

    def __init__(
        self,
        head_checkpoint_path: str = "best_patch_head.pth",
        in_channels: int = 1024,
        num_classes: int = 151,
        device: str = "cuda",
    ):
        """
        Initialize the labeler with trained head weights.

        Args:
            head_checkpoint_path: Path to the saved head checkpoint
            in_channels: Number of input channels (should match training)
            num_classes: Number of output classes (should match training)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        # Initialize the head architecture (same as in classifier.py)
        self.head = PatchClassifierHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=0.0,
            use_bn=False,
        ).to(self.device)

        # Load the trained weights
        self.load_head_weights(head_checkpoint_path)

        # Set to evaluation mode
        self.head.eval()

    def load_head_weights(self, checkpoint_path: str):
        """Load the trained head weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "head" in checkpoint:
            self.head.load_state_dict(checkpoint["head"])
            print(f"✓ Loaded head weights from {checkpoint_path}")
        else:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'head' key.")

    def load_features(self, feature_path: str) -> torch.Tensor:
        """
        Load n×c channel features from file.

        Args:
            feature_path: Path to the feature file

        Returns:
            torch.Tensor: Features tensor of shape [n, c] or [c, h, w] or [n, c, h, w]
        """
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        # Try different loading methods based on file extension
        file_ext = Path(feature_path).suffix.lower()

        if file_ext == ".npy":
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()
        elif file_ext == ".pt" or file_ext == ".pth":
            features = torch.load(feature_path, map_location="cpu")
        elif file_ext == ".npz":
            data = np.load(feature_path)
            # Assume the features are stored under 'features' key, adjust as needed
            if "features" in data:
                features = torch.from_numpy(data["features"]).float()
            else:
                # Take the first array if no 'features' key
                key = list(data.keys())[0]
                features = torch.from_numpy(data[key]).float()
                print(f"Using key '{key}' from npz file")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        print(f"✓ Loaded features with shape: {features.shape}")
        return features

    def prepare_features_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for the classification head.
        The head expects input of shape [B, C, H, W].

        Args:
            features: Input features tensor

        Returns:
            torch.Tensor: Reshaped features ready for the head
        """
        original_shape = features.shape

        if len(features.shape) == 2:
            # Shape: [n, c] -> treat as [n, c, 1, 1]
            n, c = features.shape
            features = features.view(n, c, 1, 1)
            print(f"Reshaped [n={n}, c={c}] -> [n={n}, c={c}, h=1, w=1]")

        elif len(features.shape) == 3:
            # Shape: [c, h, w] -> [1, c, h, w]
            features = features.unsqueeze(0)
            print(f"Reshaped {original_shape} -> {features.shape}")

        elif len(features.shape) == 4:
            # Shape: [n, c, h, w] -> already correct
            print(f"Features already in correct shape: {features.shape}")

        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")

        return features.to(self.device)

    @torch.no_grad()
    def generate_labels(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate labels from features using the trained head.

        Args:
            features: Input features tensor

        Returns:
            torch.Tensor: Predicted labels
        """
        # Prepare features for the head
        features = self.prepare_features_for_head(features)

        # Forward pass through the head
        logits = self.head(features)  # [B, num_classes, H, W]

        # Get predicted labels
        labels = logits.argmax(dim=1)  # [B, H, W]

        return labels.cpu()

    def label_from_file(
        self, feature_path: str, output_path: str = None
    ) -> torch.Tensor:
        """
        Complete pipeline: load features from file and generate labels.

        Args:
            feature_path: Path to input feature file
            output_path: Optional path to save the labels

        Returns:
            torch.Tensor: Predicted labels
        """
        print(f"Loading features from: {feature_path}")
        features = self.load_features(
            "/data0/vpr/CUHK_LOWER/ckpts/ckpt_6998_rank0features.pt"
        )
        features_1 = self.load_features(
            "/data0/vpr/CUHK_LOWER/ckpts/ckpt_6998_rank0features1.pt"
        )
        features = torch.cat([features, features_1], dim=1)

        print("Generating labels...")
        labels = self.generate_labels(features)

        print(f"✓ Generated labels with shape: {labels.shape}")
        print(f"Label range: {labels.min().item()} - {labels.max().item()}")

        # Save labels if output path is provided
        if output_path:
            if output_path.endswith(".npy"):
                np.save(output_path, labels.numpy())
            elif output_path.endswith(".pt") or output_path.endswith(".pth"):
                torch.save(labels.squeeze(1).squeeze(1), output_path)
            else:
                # Default to numpy
                np.save(output_path, labels.numpy())
            print(f"✓ Saved labels to: {output_path}")

        return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate labels from features using trained head"
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to input feature file (n×c channels)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_patch_head.pth",
        help="Path to head checkpoint file",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save output labels"
    )
    parser.add_argument(
        "--in_channels", type=int, default=1024, help="Number of input channels"
    )
    parser.add_argument(
        "--num_classes", type=int, default=151, help="Number of output classes"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )

    args = parser.parse_args()

    # Initialize labeler
    labeler = FeatureLabeler(
        head_checkpoint_path=args.checkpoint,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        device=args.device,
    )

    # Generate labels
    labels = labeler.label_from_file(args.features, args.output)

    print(f"\n✓ Labeling complete!")
    print(f"Input features: {args.features}")
    print(f"Output labels shape: {labels.shape}")
    if args.output:
        print(f"Labels saved to: {args.output}")


if __name__ == "__main__":
    main()
