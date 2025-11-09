"""
Header Module

This module provides image processing, classification, and data loading utilities
for the scene decomposition pipeline.

Components:
- ade20k_loader: ADE20K dataset loading utilities
- classifier: Image classification tools
- labeling: Image labeling and annotation utilities
"""

from .ade20k_loader import ADE20KLoader
from .classifier import ImageClassifier
from .labeling import ImageLabeler

__all__ = [
    "ADE20KLoader",
    "ImageClassifier",
    "ImageLabeler",
]
