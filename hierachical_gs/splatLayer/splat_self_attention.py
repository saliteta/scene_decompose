# 

import torch
import torch.nn as nn
from abc import ABC

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


class SplatMeanPooling(nn.Module):
    def __init__(self):
        super(SplatMeanPooling, self).__init__()
    
    def forward(self, x: torch.Tensor):
        y = x.mean(dim=0)  # Average across the first dimension (children)
        y = torch.nn.functional.normalize(y, dim=-1)
        return y


class SplatAttention(nn.Module):
    """
    Parameter-free self-attention for tensor X with shape (N, C)
    Thinks it as a strong filter to the features
    """
    def __init__(self):
        super(SplatAttention, self).__init__()
    
    def forward(self, x):
        """
        Parameter-free self-attention for tensor X with shape (N, C)
        
        Args:
            x: Input tensor of shape (N, C) where N is sequence length, C is feature dimension
            
        Returns:
            output: Attention-weighted features of shape (N, C)
        """
        N, C = x.shape
        
        # Use input X directly as Query, Key, Value (no learnable projections)
        Q = x  # (N, C)
        K = x  # (N, C)
        V = x  # (N, C)
        
        # Compute attention scores: Q @ K.T
        # Q: (N, C), K: (N, C) -> (N, N)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (N, N)
        
        # Apply attention to values: attention_weights @ V
        # (N, N) @ (N, C) -> (N, C)
        output = torch.matmul(attention_weights, V)
        
        return output


class SplatAttentionPooling(nn.Module):
    def __init__(self):
        super(SplatAttentionPooling, self).__init__()
        self.pooling = SplatMeanPooling()
        self.attention = SplatAttention()
    
    def forward(self, x):
        x = self.attention(x)
        x = self.pooling(x)
        return x



