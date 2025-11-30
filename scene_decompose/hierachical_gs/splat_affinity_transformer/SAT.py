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
        # Convert to [B, N, C] format if needed and store residual
        if self.input_format == "BCN":
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor [B, C, N], got {tuple(x.shape)}")
            B, C, N = x.shape
            x_bnc = x.transpose(1, 2)  # [B, N, C]
            residual = x_bnc  # Store residual in BNC format
        else:  # BNC
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor [B, N, C], got {tuple(x.shape)}")
            B, N, C = x.shape
            x_bnc = x
            residual = x_bnc  # Store residual in BNC format
        
        if C != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {C}")
        
        # Project to Q, K, V
        Q = self.q_proj(x_bnc)  # [B, N, dim]
        K = self.k_proj(x_bnc)  # [B, N, dim]
        V = self.v_proj(x_bnc)  # [B, N, dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        
        # 1. Compute Raw Attention Scores (Scaled Dot Product)
        # These raw scores (logits) are best for the affinity matrix as they are not bounded yet.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        
        # 2. Compute Probabilities for Feature Aggregation (Standard Transformer)
        # We use Softmax here to ensure stable feature magnitude preservation.
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, H, N, N]
        
        # 3. Extract Affinity (if requested)
        # We use the RAW scores (averaged over heads) for the affinity output.
        # This preserves the random initialization behavior (centered around 0).
        if self.return_affinity:
            affinity = attn_scores.mean(dim=1) # [B, N, N]
        else:
            affinity = None
        
        # 4. Apply attention to values (Standard Transformer)
        y = torch.matmul(attn_probs, V)  # [B, H, N, head_dim]
        
        # Reshape back: [B, H, N, head_dim] -> [B, N, dim]
        y = y.transpose(1, 2).contiguous().view(B, N, self.dim)
        
        # Output projection
        y = self.out_proj(y)  # [B, N, dim]
        
        # Add residual connection
        y = y + residual  # [B, N, dim]
        
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
    - affinity: [B, N, N] affinity matrix (bounded to [-1, 1] via tanh)
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
        
        # Layer normalization between layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, N, C] or [B, C, N]
        Returns:
            If return_affinity=False: features [B, N, C] or [B, C, N]
            If return_affinity=True: tuple (affinity [B, N, N], features [B, N, C] or [B, C, N])
                where affinity is bounded to [-1, 1] via tanh
        """
        for i, layer in enumerate(self.layers):
            if self.return_affinity and i == len(self.layers) - 1:
                # Last layer returns both affinity (raw scores) and features
                affinity, x = layer(x)
                
                # --- MODIFICATION ---
                # Since 'affinity' now contains raw scores (centered around 0),
                # we simply apply Tanh. No need for scaling or shifting.
                # This ensures outputs are distributed naturally between -1 and 1.
                affinity = torch.tanh(affinity)  # [-inf, inf] -> [-1, 1]
                
                # Apply layer norm to features (post-norm style)
                x = self.layer_norms[i](x)
                
                # Convert back to original format if needed
                if self.input_format == "BCN":
                    x = x.transpose(1, 2)  # [B, C, N]
                
                return affinity, x
            else:
                # Regular layer
                x = layer(x)
                
                # Apply layer norm
                x = self.layer_norms[i](x)
        
        # Convert back to original format if needed
        if self.input_format == "BCN":
            x = x.transpose(1, 2)  # [B, C, N]
        
        return x


if __name__ == "__main__":
    # Set seed for reproducibility check
    torch.manual_seed(42)

    # Test with BNC format
    x_bnc = torch.randn(2, 64, 1024)  # [B, N, C]
    
    # Test stack - returns both affinity and features
    sat_stack = SplatAffinityTransformerStack(dim=1024, num_layers=3, num_heads=8, return_affinity=True)
    affinity, features = sat_stack(x_bnc)
    
    print("Affinity Matrix Sample (Top-left 5x5):")
    print(affinity[0, :5, :5])
    
    # Check if values are identical (They shouldn't be now)
    std_dev = affinity.std()
    print(f"\nStandard Deviation of Affinity: {std_dev.item():.4f}")
    if std_dev > 0.001:
        print("Success: Affinity values are diverse.")
    else:
        print("Warning: Affinity values are still identical.")