"""
Splat Mixture Layer is a layer that is used to mix the Gaussian representation of the GS
The input of features are in the following format:
[N, C]
C is:
Mean, Quat, Scale, Opacity, SH0, SHN
3, 4, 3, 1, 3, xx
xx is the number of SH coefficients

We will mix the features of the GS according to the tree structure
The mixture function is following the Gaussian Mixture Model

We first convert the mean, quat and scale to covariance matrix
Then according to the opacity which is the weight of the Gaussian, we will mix a new Gaussian
We will also convert the SH0 and SHN to the color of the Gaussian

According to:
Csiszár, Imre. "I-divergence geometry of probability distributions and minimization problems." The annals of probability (1975): 146-158.
Amari, Shun-ichi. "Information geometry and its applications." Journal of Mathematical Psychology 49 (2005): 101-102.
Wainwright, Martin J., and Michael I. Jordan. "Graphical models, exponential families, and variational inference." Foundations and Trends® in Machine Learning 1.1–2 (2008): 1-305.

We are using a Moment Matching method to mix the Gaussian
"""

import torch
from torch import nn
from gsplat import quat_scale_to_covar_preci
from gsplat_ext import covar_to_quat_scale 

def _outer3(v: torch.Tensor) -> torch.Tensor:
    """v: [...,3] -> [...,3,3]"""
    return v[..., :, None] * v[..., None, :]

class GaussianSplatMomentMatching(nn.Module):
    """
    Input x: [K, N, C] with layout per splat:
        [ mean(3) | quat(4) | scale(3) | opacity(1) | features(F) ]
    Output torch.Tensor:
        [ mean(3) | quat(4) | scale(3) | opacity(1) | features(F) ]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):

        assert x.dim() == 3, "x must be [K, N, C], usually it is [2, N, C]"
        K, N, C = x.shape
        dtype, device = x.dtype, x.device
        eps = torch.finfo(dtype).eps

        means = x[:, :, 0:3]           # [K,N,3]
        quats = x[:, :, 3:7]           # [K,N,4]
        scales = x[:, :, 7:10]         # [K,N,3]
        opacities = x[:, :, 10:11]     # [K,N,1]
        feats = x[:, :, 11:]           # [K,N,F] (F can be 0)

        # covars from (quat, scale)
        # quat_scale_to_covar_preci is expected to return (covar, precision)
        quats = quats.reshape(-1, 4)
        scales = scales.reshape(-1, 3)
        covars, _ = quat_scale_to_covar_preci(quats, scales)   # [K,N,3,3]
        covars = covars.reshape(K, N, 3, 3)

        # Normalize weights across K for each item n
        wsum = torch.clamp(opacities.sum(dim=0, keepdim=True), min=eps)  # [1,N,1]
        w = opacities / wsum   # [K,N,1], sum_k w[k,n] = 1

        # Moment-matched mean
        mu = (w * means).sum(dim=0)  # [N,3]

        # Moment-matched covariance: Σ = Σ_k w_k (Σ_k + (μ_k-μ)(μ_k-μ)^T)
        diffs = means - mu.unsqueeze(0)     # [K,N,3]
        Sigma = (w.unsqueeze(-1) * (covars + _outer3(diffs))).sum(dim=0)  # [N,3,3]
        # Convert Sigma [N,3,3] to packed upper-triangular [N,6]: [s00, s01, s02, s11, s12, s22]
        Sigma = torch.stack([Sigma[:,0,0], Sigma[:,0,1], Sigma[:,0,2],
                             Sigma[:,1,1], Sigma[:,1,2], Sigma[:,2,2]], dim=-1)  # [N,6]

        # convert Sigma [N,6] to quat and scale
        quats_rec_gsplat, scales_rec_gsplat = covar_to_quat_scale(Sigma)

        opacity_m = torch.clamp(opacities.sum(dim=0), min=eps, max=1.0)  # [N,1]

        # Merge features by weight-averaging with normalized w
        # This might not be correct depends on the sherical harmonic function
        if feats.numel() > 0:
            feats_m = (w * feats).sum(dim=0)  # [N,F]
        else:
            feats_m = feats.new_zeros((N, 0))
        return torch.cat([mu, quats_rec_gsplat, scales_rec_gsplat, opacity_m, feats_m], dim=-1)


