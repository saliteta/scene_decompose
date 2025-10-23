"""
Scene Decompose: Hierarchical Gaussian Splatting with Query System

A comprehensive package for hierarchical 3D scene representation and query-based
scene decomposition using Gaussian Splatting techniques.

Main Components:
- header: Image processing and classification utilities
- hierachical_gs: Hierarchical Gaussian Splatting implementation
- query_system: Query-based scene analysis and retrieval system

Example Usage:
    from scene_decompose.hierachical_gs.tree import PCABinaryTree
    from scene_decompose.query_system.database import FeatureDatabase
    from scene_decompose.header.classifier import ImageClassifier
"""

__version__ = "0.1.0"
__author__ = "Butian Xiong"
__email__ = "xiongbutian768@gmail.com"

# Import main components for easy access
from .hierachical_gs.tree import PCABinaryTree, HierarchicalTree, TreeNode, LayerInfo
from .query_system.database import Database, FeatureDatabase
from .header.classifier import ImageClassifier

__all__ = [
    # Tree components
    "PCABinaryTree",
    "HierarchicalTree", 
    "TreeNode",
    "LayerInfo",
    # Database components
    "Database",
    "FeatureDatabase",
    # Header components
    "ImageClassifier",
    # Package info
    "__version__",
    "__author__",
    "__email__",
]
