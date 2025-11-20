#
#

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Splat Affinity Transformer (SAT)
    - Determines the affinity matrix between multiple splats
    - If affinity score > 0, merge splats into one semantic block; otherwise, separate them
    - Uses learnable self-attention layers to compute affinities
    
    Input:  x ∈ R^{B × N × C} or R^{B × C × N}
    Output: affinity_matrix ∈ R^{B × N × N} or features ∈ R^{B × N × C}
"""


class SplatAffinityTransformer(nn.Module):
    """
    Simple transformer layer with learnable parameters for computing splat affinities.
    
    Input:  x ∈ R^{B × N × C}  (N splats, each with C features)
    Output: 
        - If return_affinity=False: y ∈ R^{B × N × C}  (transformed features)
        - If return_affinity=True: tuple (affinity ∈ R^{B × N × N}, y ∈ R^{B × N × C})
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        return_affinity: bool = False,
        input_format: str = "BNC",  # "BNC" or "BCN"
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.return_affinity = return_affinity
        self.input_format = input_format
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = dim // num_heads
        
        # Learnable projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape [B, N, C] or [B, C, N] depending on input_format
        Returns:
            If return_affinity=False: transformed features [B, N, C] or [B, C, N]
            If return_affinity=True: tuple (affinity [B, N, N], features [B, N, C] or [B, C, N])
        """
        # Convert to [B, N, C] format if needed
        if self.input_format == "BCN":
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor [B, C, N], got {tuple(x.shape)}")
            B, C, N = x.shape
            x = x.transpose(1, 2)  # [B, N, C]
        else:  # BNC
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor [B, N, C], got {tuple(x.shape)}")
            B, N, C = x.shape
        
        if C != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {C}")
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [B, N, dim]
        K = self.k_proj(x)  # [B, N, dim]
        V = self.v_proj(x)  # [B, N, dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        attn = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        
        # Compute affinity matrix (average over heads): [B, H, N, N] -> [B, N, N]
        affinity = attn.mean(dim=1) if self.return_affinity else None
        
        # Apply attention to values
        y = torch.matmul(attn, V)  # [B, H, N, head_dim]
        
        # Reshape back: [B, H, N, head_dim] -> [B, N, dim]
        y = y.transpose(1, 2).contiguous().view(B, N, self.dim)
        
        # Output projection
        y = self.out_proj(y)  # [B, N, dim]
        
        # Convert back to original format if needed
        if self.input_format == "BCN":
            y = y.transpose(1, 2)  # [B, dim, N]
        
        # Return both affinity and features if requested
        if self.return_affinity:
            return affinity, y
        
        return y


class SplatAffinityTransformerStack(nn.Module):
    """
    Stack of multiple SplatAffinityTransformer layers.
    Can be used to build deeper networks for computing affinities.
    
    When return_affinity=True, returns both:
    - affinity: [B, N, N] affinity matrix
    - features: [B, N, C] or [B, C, N] latent features
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        return_affinity: bool = False,
        input_format: str = "BNC",
    ):
        super().__init__()
        self.return_affinity = return_affinity
        self.input_format = input_format
        self.layers = nn.ModuleList([
            SplatAffinityTransformer(
                dim=dim,
                num_heads=num_heads,
                return_affinity=(return_affinity and i == num_layers - 1),  # Only last layer returns affinity
                input_format=input_format if i == 0 else "BNC",  # First layer handles format conversion
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, N, C] or [B, C, N]
        Returns:
            If return_affinity=False: features [B, N, C] or [B, C, N]
            If return_affinity=True: tuple (affinity [B, N, N], features [B, N, C] or [B, C, N])
        """
        for i, layer in enumerate(self.layers):
            if self.return_affinity and i == len(self.layers) - 1:
                # Last layer returns both affinity and features
                affinity, x = layer(x)
                return affinity, x
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    # Test with BNC format
    x_bnc = torch.randn(2, 64, 1024)  # [B, N, C]
    
    # Test stack - returns both affinity and features
    sat_stack = SplatAffinityTransformerStack(dim=1024, num_layers=3, num_heads=8, return_affinity=True)
    affinity, features = sat_stack(x_bnc)
    print(f"Affinity matrix shape: {affinity.shape}")  # [2, 64, 64]
    print(f"Latent features shape: {features.shape}")  # [2, 64, 1024]
    
    # Test without affinity
    sat_stack_no_aff = SplatAffinityTransformerStack(dim=1024, num_layers=3, num_heads=8, return_affinity=False)
    features_only = sat_stack_no_aff(x_bnc)
    print(f"Features only shape: {features_only.shape}")  # [2, 64, 1024]

    '''
    Affinity matrix shape: torch.Size([2, 64, 64])
    Latent features shape: torch.Size([2, 64, 1024])
    Features only shape: torch.Size([2, 64, 1024])
    '''