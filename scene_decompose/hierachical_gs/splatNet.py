"""
SplatCondenseNet implementation for the new HierarchicalTree structure.
This implementation is clean, efficient, and leverages the new tree's capabilities.


Notice that this implementation can be generalized to other features
"""

import torch
import torch.nn as nn
from .tree.tree import HierarchicalTree
from .splatLayer.splat_self_attention import SplatMeanPooling, SplatAttentionPooling
from tqdm import trange
from abc import ABC, abstractmethod
from typing import Optional
from .splatLayer.gaussian_moment_matching import GaussianSplatMomentMatching


class SplatNet(nn.Module, ABC):
    """
    Base class for SplatNet.
    """

    def __init__(self, tree: Optional[HierarchicalTree] = None):
        super(SplatNet, self).__init__()

        self.tree = tree

    def forward(self, tree: Optional[HierarchicalTree] = None):
        """
        Process features hierarchically using mean pooling for early layers and attention for later layers.

        Args:
            tree: Optional tree to process (uses self.tree if None)

        Returns:
            The processed tree with updated features
        """
        assert tree is not None or self.tree is not None, "Tree is required"
        if tree is None:
            tree = self.tree
        else:
            self.tree = tree

        tree_info = tree.get_tree_info()
        num_layers = tree_info["num_layers"]

        print(f"\nProcessing tree with {num_layers} layers...")

        # Process each layer bottom-up (skip layer 0 - leaves)
        for current_layer_id in trange(1, num_layers, desc="Processing layers"):
            processed_features = self.layer_forward(tree, current_layer_id)
            tree.update_layer_features(current_layer_id, processed_features)

        print(f"\n✓ SplatNet processing completed!")
        return tree

    @abstractmethod
    def layer_forward(self, tree: HierarchicalTree, layer_id: int):
        """
        Process features for a single layer.
        """
        pass

    @abstractmethod
    def verbose(self) -> str:
        """
        Get a verbose string representation of the network.
        """
        summary = {
            "name": self.__class__.__name__,
            "tree_structure": self.tree.get_tree_info(),
        }
        return summary

    def dict_to_printable(self, summary: dict, indent: int = 0) -> str:
        """
        Convert a (possibly nested) dictionary or iterable into an indented printable string.
        Handles dicts, lists, tuples, and sets recursively.
        """
        lines = []
        indent_str = "  " * indent

        if isinstance(summary, dict):
            for key, value in summary.items():
                if isinstance(value, (dict, list, tuple, set)):
                    lines.append(f"{indent_str}{key}:")
                    nested = self.dict_to_printable(value, indent + 1)
                    if nested:
                        lines.extend(nested.splitlines())
                else:
                    lines.append(f"{indent_str}{key}: {value}")
        elif isinstance(summary, (list, tuple, set)):
            for item in summary:
                if isinstance(item, (dict, list, tuple, set)):
                    lines.append(f"{indent_str}-")
                    nested = self.dict_to_printable(item, indent + 1)
                    if nested:
                        lines.extend(nested.splitlines())
                else:
                    lines.append(f"{indent_str}- {item}")
        else:
            lines.append(f"{indent_str}{summary}")

        return "\n".join(lines)


class SplatCondenseNet(SplatNet):
    """
    Efficient hierarchical feature condensation using the new tree structure.

    Strategy:
    - Layers 1-2: Use mean pooling (2 children per parent, attention not useful)
    - Layer 3+: Use attention pooling (8+ children per parent, attention becomes useful)
    """

    def __init__(
        self,
        tree: HierarchicalTree,
        attention_start_layer: int = 3,
        hops_for_attention: int = 3,
    ):
        super().__init__()

        self.tree = tree
        self.attention_start_layer = attention_start_layer
        self.hops_for_attention = hops_for_attention

        # Initialize pooling modules
        self.mean_pooling = SplatMeanPooling()
        self.attention_pooling = SplatAttentionPooling()

        print(f"SplatCondenseNet initialized:")
        print(f"  - Attention starts at layer {attention_start_layer}")
        print(
            f"  - Using {hops_for_attention} hops for attention (2^{hops_for_attention} = {2**hops_for_attention} children per parent)"
        )

    def layer_forward(self, tree: HierarchicalTree, layer_id: int) -> None:
        """
        Process features for a single layer.
        """
        if layer_id < self.attention_start_layer:
            processed_features = self._process_with_mean_pooling(tree, layer_id)
        else:
            processed_features = self._process_with_attention_pooling(tree, layer_id)
        return processed_features

    def _process_with_mean_pooling(
        self, tree: HierarchicalTree, parent_layer_id: int
    ) -> torch.Tensor:
        """
        Process features using mean pooling (direct parent-child relationship).
        Processes in batches of 4096 for memory efficiency.
        """
        # Get parent-child features: [num_parents, 2, feature_dim]
        parent_child_features = tree.get_parent_child_features(
            parent_layer_id
        )  # [num_parents, 2, feature_dim]
        host_device = parent_child_features.device
        num_parents = parent_child_features.shape[0]
        batch_size = 4096

        # Process in batches
        processed_batches = []
        for batch_start in range(0, num_parents, batch_size):
            batch_end = min(batch_start + batch_size, num_parents)
            batch_features = parent_child_features[batch_start:batch_end].to("cuda")
            batch_processed = self.mean_pooling(
                batch_features
            )  # [batch_size, feature_dim]
            processed_batches.append(batch_processed.to(host_device))

        # Concatenate all batches
        processed_features = torch.cat(
            processed_batches, dim=0
        )  # [num_parents, feature_dim]
        return processed_features

    def _process_with_attention_pooling(
        self, tree: HierarchicalTree, parent_layer_id: int
    ) -> torch.Tensor:
        """
        Process features using attention pooling (multiple hops to get more children per parent).
        Processes in batches of 512 for memory efficiency.
        """
        # Calculate source layer (where we collect features from)
        source_layer_id = max(0, parent_layer_id - self.hops_for_attention)

        # Get ancestor-descendant features: [num_ancestors, 2^hops, feature_dim]
        ancestor_descendant_features = tree.get_ancestor_descendant_features(
            source_layer_id, self.hops_for_attention
        )
        host_device = ancestor_descendant_features.device
        num_ancestors = ancestor_descendant_features.shape[0]
        batch_size = 512

        # Process in batches
        processed_batches = []
        for batch_start in range(0, num_ancestors, batch_size):
            batch_end = min(batch_start + batch_size, num_ancestors)
            batch_features = ancestor_descendant_features[batch_start:batch_end].to(
                "cuda"
            )
            batch_processed = self.attention_pooling(
                batch_features
            )  # [batch_size, feature_dim]
            processed_batches.append(batch_processed.to(host_device))

        # Concatenate all batches
        new_ancestor_features = torch.cat(
            processed_batches, dim=0
        )  # [num_ancestors, feature_dim]

        target_layer_id = source_layer_id + self.hops_for_attention
        assert (
            target_layer_id == parent_layer_id
        ), "Target layer ID does not match parent layer ID"
        return new_ancestor_features

    def verbose(self) -> dict:
        """Get a summary of the processing strategy."""
        summary = self.super().verbose()
        summary["attention_start_layer"] = self.attention_start_layer
        summary["hops_for_attention"] = self.hops_for_attention
        summary["children_per_parent_attention"] = 2**self.hops_for_attention
        summary["processing_strategy"] = {}

        for layer_id in range(1, summary["tree_structure"]["num_layers"]):
            if layer_id < self.attention_start_layer:
                summary["processing_strategy"][layer_id] = {
                    "method": "mean_pooling",
                    "children_per_parent": 2,
                    "description": "Direct parent-child relationship",
                }
            else:
                summary["processing_strategy"][layer_id] = {
                    "method": "attention_pooling",
                    "children_per_parent": 2**self.hops_for_attention,
                    "description": f"Multi-level ancestor-descendant relationship ({self.hops_for_attention} hops)",
                }

        return summary


class GSMMNet(SplatNet):
    """
    GSMomentMatchingNet is a network that is used to calculate merged Gaussian Splat from the Gaussian Splat at the lower layer
    The proposed algorithm merged exponential family of distributions into a single distribution, the Gaussian Splat.
    The proposed method is called Moment Matching.
    Detail in GaussianSplatMomentMatching class.
    """

    def __init__(self, tree: HierarchicalTree):
        super().__init__(tree)

        self.gsmm = GaussianSplatMomentMatching()

    def layer_forward(self, tree: HierarchicalTree, layer_id: int):
        """
        Process features for a single layer.
        """
        parent_child_features = tree.get_parent_child_features(layer_id)
        parent_child_features = parent_child_features.permute(1, 0, 2)
        processed_features = self.gsmm(parent_child_features)

        return processed_features

    def verbose(self) -> dict:
        """Get a summary of the processing strategy."""
        summary = self.super().verbose()
        summary["gsmm"] = {
            "method": "gaussian_moment_matching",
            "description": "Merge Gaussian Splat using Moment Matching",
        }
        summary["Input sequence"] = (
            "[ mean(3) | quat(4) | scale(3) | opacity(1) | features(F) ]"
        )
        return summary
