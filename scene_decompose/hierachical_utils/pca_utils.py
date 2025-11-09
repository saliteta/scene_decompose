import torch
from typing import Tuple
from torch.linalg import svd


def pca_torch_extract_directions(
    features_NC: torch.Tensor, n_components: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract PCA directions and projections using PyTorch tensors.
    Args:
        features_NC: [N, C] feature tensor
        n_components: number of principal components to extract
    Returns:
        projected_features: [N, n_components] - projected features
        pc_directions: [n_components, C] - principal component directions
        mean: [C] - mean of original features
    """
    # Center the data
    mean = features_NC.mean(dim=0, keepdim=True)  # [1, C]
    X_centered = features_NC - mean  # [N, C]

    # SVD decomposition using PyTorch
    U, S, Vt = svd(X_centered, full_matrices=False)  # X_centered = U @ S @ Vt

    # Extract principal components
    pc_directions = Vt[:n_components, :]  # [n_components, C]
    projected_features = U[:, :n_components] @ torch.diag(
        S[:n_components]
    )  # [N, n_components]

    return projected_features, pc_directions, mean.squeeze(0)


def apply_pca_directions_torch(
    features_NC: torch.Tensor, pc_directions: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """
    Apply pre-computed PCA directions to new features using PyTorch tensors.
    Args:
        features_NC: [N, C] new feature tensor
        pc_directions: [n_components, C] principal component directions
        mean: [C] mean from original PCA computation
    Returns:
        projected_features: [N, n_components] - projected features
    """
    # Center the new data using the original mean
    device = features_NC.device
    features_NC = features_NC.to(device)
    pc_directions = pc_directions.to(device)
    mean = mean.to(device)
    X_centered = features_NC - mean  # [N, C]

    # Project using the principal component directions
    projected_features = X_centered @ pc_directions.T  # [N, n_components]

    return projected_features
