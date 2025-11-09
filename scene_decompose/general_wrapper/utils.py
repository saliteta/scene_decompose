from cuml.cluster import KMeans as cuKMeans
from ..hierachical_utils.pca_utils import pca_torch_extract_directions
import torch

def pca_quantized_feature(features_flat: torch.Tensor, k: int, random_state: int = 42, n_pca: int = 16) -> torch.Tensor:
    features_pca_flat, pc_directions, pca_mean = pca_torch_extract_directions(
        features_flat, n_components=n_pca
    )  # [H*W, n_pca]
        
    # Apply cuML K-means clustering on PCA-reduced features (GPU-accelerated)
    kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels_flat = kmeans.fit_predict(features_pca_flat.cpu().numpy())  # [H*W]
    cluster_centers_pca = torch.from_numpy(kmeans.cluster_centers_).float().to(features_pca_flat.device)
    cluster_centers = cluster_centers_pca @ pc_directions + pca_mean  # [K, C]
    return cluster_centers
