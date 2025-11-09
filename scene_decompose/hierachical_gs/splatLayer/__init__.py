"""
Splat Layer Module

This module contains various neural network layers and attention mechanisms
for processing Gaussian Splatting data.

Components:
- splat_self_attention: Self-attention mechanisms for splat features
- splat_image_attention: Cross-attention between splat and image features
- gaussian_moment_matching: Gaussian moment matching utilities
"""

from .splat_self_attention import (
    SplatAttention,
    SplatMeanPooling,
    SplatAttentionPooling,
)
from .splat_image_attention import SplatImageAttention
from .gaussian_moment_matching import GaussianSplatMomentMatching

__all__ = [
    "SplatAttention",
    "SplatMeanPooling",
    "SplatAttentionPooling",
    "SplatImageAttention",
    "GaussianSplatMomentMatching",
]
