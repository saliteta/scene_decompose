"""
Hierarchical Gaussian Splatting Module

This module provides tools for creating and working with hierarchical
representations of Gaussian Splatting scenes.

Components:
- tree: Tree data structures for hierarchical processing
- splatLayer: Attention and pooling layers for splat processing
- splatNet: Neural networks for splat processing
"""

from .tree import PCABinaryTree, HierarchicalTree, TreeNode, LayerInfo
from .splatNet import SplatCondenseNet, GSMMNet

__all__ = [
    "PCABinaryTree",
    "HierarchicalTree", 
    "TreeNode",
    "LayerInfo",
    "SplatCondenseNet",
    "GSMMNet",
]
