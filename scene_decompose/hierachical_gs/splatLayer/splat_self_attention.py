#

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


class SplatMeanPooling(nn.Module):
    """
    Input:  x ∈ R^{B × N × C}
    Op:     mean over N (dim=1), then L2-normalize over last dim
    Output: y ∈ R^{B × C}
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape checks
        assert (
            x.ndim == 3
        ), f"SplatMeanPooling expects a 3D tensor [B, C, N], got shape {tuple(x.shape)}"
        B, C, N = x.shape
        assert C > 0 and N > 0, f"Invalid [C, N] dimensions: C={C}, N={N}"

        # mean over C dimension (C)
        y = x.mean(dim=1)  # [B, N]
        y = torch.nn.functional.normalize(y, dim=-1)
        return y


class SplatAttention(nn.Module):
    """
    Parameter-free self-attention over the N dimension.
    Input:  x ∈ R^{B × C × N}
    Output: y ∈ R^{B × C × N}
    Q=K=V=x (no projections). Softmax over N.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input would be [B,C,N], Output would be [B,C,N] check PCA difference
        if x.dim() != 3:
            raise ValueError(f"SplatAttention expects [B, C, N], got {tuple(x.shape)}")
        B, C, N = x.shape
        if C <= 0 or N <= 0:
            raise ValueError(f"Invalid dimensions: C={C}, N={N}")

        # Move to [B, N, C] so N is the sequence length, C is feature dim

        Q = x  # [B, N, C]
        K = x  # [B, N, C]
        V = x  # [B, N, C]

        # Attention scores: [B, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # (Q @ K^T)

        # Softmax over keys (last dim = N)
        attn = F.softmax(attn_scores, dim=-1)  # [B, N, N]

        # Weighted sum: [B, N, C]
        Y = torch.matmul(attn, V)

        # Back to [B, C, N] # (64, 8, 1024)
        return Y


class SplatAttentionPooling(nn.Module):
    """
    Parameter-free attention over N followed by mean-pooling over C.
    Memory-friendly: processes the batch B in chunks to avoid CUDA OOM.

    Input:  x ∈ R^{B × C × N}
    Output: y ∈ R^{B × N}
    """

    def __init__(
        self, chunk_size: int = 64, eps: float = 1e-12, use_scale: bool = True
    ):
        super().__init__()
        self.pooling = SplatMeanPooling(eps=eps)  # expects [B, C, N] -> [B, N]
        self.attention = SplatAttention()  # [B, C, N] -> [B, C, N]
        self.chunk_size = int(chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape checks
        if x.dim() != 3:
            raise ValueError(
                f"SplatAttentionPooling expects [B, C, N], got {tuple(x.shape)}"
            )
        B, C, N = x.shape
        if C <= 0 or N <= 0:
            raise ValueError(f"Invalid [C, N]: {C}, {N}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")

        # process in chunks along batch dimension
        outs = []
        for start in range(0, B, self.chunk_size):
            end = min(start + self.chunk_size, B)
            x_chunk = x[start:end]  # [b, C, N]
            x_chunk = self.attention(x_chunk)  # [b, C, N]
            y_chunk = self.pooling(x_chunk)  # [b, N]
            outs.append(y_chunk)

        y = torch.cat(outs, dim=0)  # [B, N]
        return y


if __name__ == "__main__":
    x = torch.randn(64, 8, 1024)
    pooling = SplatMeanPooling()
    attention = SplatAttention()
    attention_pooling = SplatAttentionPooling()
    y = attention_pooling(x)
    print(y.shape)
    exit()
