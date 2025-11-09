import numpy as np
import torch
import tyro
from pathlib import Path
from gsplat_ext import GaussianPrimitive
from hierachical_gs.tree.pca_tree import PCABinaryTree
from hierachical_gs.splatNet import SplatCondenseNet, GSMMNet
from hierachical_utils.hierachical_primitive import HierachicalPrimitive
from dataclasses import dataclass
from typing import Optional, Annotated


@dataclass
class HierarchicalGSConfig:
    """Configuration for constructing hierarchical Gaussian Splats"""

    # Input paths
    ckpt_path: Annotated[Path, tyro.conf.arg(aliases=["-c"])]
    """Path to the checkpoint file"""

    feature_path: Annotated[Path, tyro.conf.arg(aliases=["-f"])]
    """Path to the feature file"""

    # Output path
    output_path: Annotated[Path, tyro.conf.arg(aliases=["-o"])]
    """Path to save the hierarchical Gaussian Splat"""

    # Network parameters
    attention_start_layer: Annotated[int, tyro.conf.arg(aliases=["-asl"])] = 3
    """Layer to start attention mechanism"""

    hops_for_attention: Annotated[int, tyro.conf.arg(aliases=["-hfa"])] = 3
    """Number of hops for attention"""

    # Processing options
    with_feature: Annotated[bool, tyro.conf.arg(aliases=["-wf"])] = True
    """Whether to include features in the hierarchical primitive"""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = True
    """Whether to print progress information"""


def form_features(primitive: GaussianPrimitive):
    means = primitive.geometry["means"]
    quats = primitive.geometry["quats"]
    scales = primitive.geometry["scales"]
    opacities = primitive.geometry["opacities"].unsqueeze(-1)
    colors = primitive.color["colors"].reshape(means.shape[0], -1)
    return torch.cat([means, quats, scales, opacities, colors], dim=-1)


def main(config: HierarchicalGSConfig):
    """Main function to construct hierarchical Gaussian Splats"""

    if config.verbose:
        print("🚀 Starting Hierarchical Gaussian Splat Construction")
        print(f"📁 Checkpoint: {config.ckpt_path}")
        print(f"📁 Features: {config.feature_path}")
        print(f"📁 Output: {config.output_path}")
        print(f"⚙️  Attention start layer: {config.attention_start_layer}")
        print(f"⚙️  Hops for attention: {config.hops_for_attention}")
        print("-" * 50)

    # Load Gaussian Primitive
    if config.verbose:
        print("📥 Loading Gaussian Primitive...")

    gs = GaussianPrimitive()
    gs.from_file(str(config.ckpt_path), feature_path=str(config.feature_path))

    if config.verbose:
        print(f"✓ Loaded {gs.geometry['means'].shape[0]} Gaussian splats")

    # Extract geometry and features
    xyz = gs.geometry["means"]
    features = gs.feature

    # Build PCA tree
    if config.verbose:
        print("🌳 Building PCA Binary Tree...")

    tree = PCABinaryTree(xyz, features=features)

    if config.verbose:
        print("✓ Tree processing completed!")

    # Create condensation network
    if config.verbose:
        print("🧠 Creating SplatCondenseNet...")

    tree = tree.to("cpu")

    condense_net = SplatCondenseNet(
        tree,
        attention_start_layer=config.attention_start_layer,
        hops_for_attention=config.hops_for_attention,
    )
    h_gs = HierachicalPrimitive(with_feature=config.with_feature)
    processed_tree = condense_net()

    # Process tree layers for features
    if config.verbose:
        print("📊 Processing tree layers for features...")

    tree_info = processed_tree.get_tree_info()
    for layer_info in tree_info["layers"]:
        layer_id = layer_info["layer_id"]
        layer_features = processed_tree.get_layer_features(layer_id)
        h_gs.add_primitive(splat_content=None, feature=layer_features)

    # Process splat features
    if config.verbose:
        print("🎨 Processing splat features...")

    splat_features = form_features(gs)
    tree.reset_to_base_features(splat_features, feature_dim=splat_features.shape[1])
    gm_net = GSMMNet(tree)
    processed_tree = gm_net()

    tree_info = processed_tree.get_tree_info()
    for layer_info in tree_info["layers"]:
        layer_id = layer_info["layer_id"]
        layer_features = processed_tree.get_layer_features(layer_id)
        h_gs.add_primitive(splat_content=layer_features, feature=None)

    # Save hierarchical Gaussian Splat
    if config.verbose:
        print(f"💾 Saving hierarchical Gaussian Splat to {config.output_path}...")

    # Create output directory if it doesn't exist
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    h_gs.save_to_file(str(config.output_path))

    if config.verbose:
        print("✅ Hierarchical Gaussian Splat construction completed!")
        print(f"📁 Saved to: {config.output_path}")


if __name__ == "__main__":
    tyro.cli(main)
