import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    Splat Layer is a network according to the tree structure provided by Some Tree:
    - PCA Binary Tree
    - ...

    The network will be used to condense the features of the Splats
    The so called condense means that:
    1. We summarize the features of the Splats at lower layer to upper layer
    2. The feature dimension will stay the same, but it the feature token might increased

    For example: 
    - Layer 0: [100, 1, 1024]
    - Layer 1: [50, 1, 1024]
    - Layer 2: [25, 1, 1024]
    - Layer 3: [12, 2, 1024]
    - Layer 4: [6, 3, 1024]
    - Layer 5: [3, 6, 1024]

    Which layer in layer 0 is corresponding to the layer in layer 1 is fixed according to the tree structure
    However, the way we convert the features in lower layer to upper layer is not fixed.
    We can use different ways to convert the features in lower layer to upper layer.
    For example:
    - Mean Pooling
    - Attention Pooling
    - Attention Summation # NOT POOLING
    
    We are suppose to provide the way to convert the features in lower layer to upper layer as a class:
    It could be convolution, attention, pooling, etc.
"""


class SplatImageAttention(nn.Module):
    """
    Parameter-free Cross-Attention over the C
    Splat Block Features: 
    -  [2**layer_level, 2**token_level, C] 
    we have 2 ** layer_level blocks
    we have 2 ** token_level token in each block

    - input image features: [K, C]
    K should be compatible with 2 ** token_level
    """
    def __init__(self):
        super().__init__()

    def forward(self, splat_block_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # Input would be [2**layer_level, 2**token_level, C] and [K, C]
        assert splat_block_features.ndim == 3, f"Splat block features must be a 3D tensor, got {splat_block_features.ndim}"
        assert image_features.ndim == 2, f"Image features must be a 2D tensor, got {image_features.ndim}"
        assert splat_block_features.shape[1] == image_features.shape[0], f"Token per block should be same as token per image,\
             got token per block: {splat_block_features.shape[1]} and token per image: {image_features.shape[0]}"
        assert splat_block_features.shape[2] == image_features.shape[1], \
            f"feature dimension should align with each other, got splat block feature dimension: \
                {splat_block_features.shape[2]} and image feature dimension: {image_features.shape[1]}"

        # Move to [B, N, C] so N is the sequence length, C is feature dim


        Q = splat_block_features  # [B, K, C]
        K = image_features.unsqueeze(0)  # [1, K, C]

        # Attention scores: [B, K, K]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # (Q @ K^T)
        
        # Max over K dimension: [B, K]
        attn_scores = attn_scores.max(dim=-1).values  # [B, K]
        
        # Sum over K dimension: [B]
        attn_scores = attn_scores.sum(dim=-1)  # [B]
        
        return attn_scores
    


if __name__ == "__main__":
    splat_block_features = torch.randn(32, 32, 1024)
    image_features = torch.randn(32, 1024)
    attention = SplatImageAttention()
    attn_scores = attention(splat_block_features, image_features)
    print(attn_scores.shape)