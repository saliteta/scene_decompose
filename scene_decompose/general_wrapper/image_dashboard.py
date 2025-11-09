# image_dashboard.py
"""
Modular image dashboard for single image processing with:
- Image query submission (feature extraction)
- Ground truth selection (camera pose selection)
- Local file access support
- PCA visualization
"""
from typing import Callable, Optional, Tuple, Dict, List, Any
import threading
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
import torch
import torchvision.transforms as T

try:
    # Try absolute imports first (when run as module)
    from scene_decompose.hierachical_utils.pca_utils import (
        apply_pca_directions_torch,
        pca_torch_extract_directions,
    )
    from scene_decompose.general_wrapper.query_data import QueryData
except ImportError:
    # Fall back to relative imports (when imported as a module)
    from ..hierachical_utils.pca_utils import (
        apply_pca_directions_torch,
        pca_torch_extract_directions,
    )
    from .query_data import QueryData

from jafar import load_model
import os
from cuml.cluster import KMeans as cuKMeans
from gsplat_ext import Parser, Dataset

try:
    # Try absolute imports first (when run as module)
    from scene_decompose.general_wrapper.feature_extractor import (
        FeatureExtractorConfig,
        FeatureExtractorJAFAR,
        FeatureExtractorOpenCLIP,
    )
except ImportError:
    # Fall back to relative imports (when imported as a module)
    from .feature_extractor import (
        FeatureExtractorConfig,
        FeatureExtractorJAFAR,
        FeatureExtractorOpenCLIP,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def _pil_shape(img: Image.Image) -> Tuple[int, int, int]:
    """Get shape of PIL image as (H, W, C)."""
    arr = np.asarray(img.convert("RGB"))
    return arr.shape


def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    """
    Load image from local file path.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image or None if loading fails
    """
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            print(f"[Error] Image path does not exist: {image_path}")
            return None
    except Exception as e:
        print(f"[Error] Failed to load image from {image_path}: {e}")
        return None


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to torch tensor in [H, W, 3] format with values in [0, 255].

    Args:
        img: PIL Image

    Returns:
        torch.Tensor of shape (H, W, 3) with values in [0, 255]
    """
    img_array = np.array(img.convert("RGB"))
    tensor_img = torch.from_numpy(img_array).float()
    return tensor_img


def extract_image_feature_token(
    img: Image.Image, feature_processor: "FeatureProcessor"
) -> torch.Tensor:
    """
    Extract feature token from image using feature processor.

    Args:
        img: PIL Image
        feature_processor: FeatureProcessor instance

    Returns:
        torch.Tensor: Feature token of shape (F,)
    """
    tensor_img = pil_to_tensor(img)

    # Convert to format expected by feature processor [B, H, W, 3]
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0)

    # Extract features
    features = feature_processor.extract_features(
        tensor_img
    )  # [B, H', W', C] or [H', W', C]

    return features


# =============================================================================
# Feature Processor Class
# =============================================================================


class FeatureProcessor:
    """
    Feature extraction processor supporting both JAFAR and OpenCLIP models.
    Uses FeatureExtractor abstraction for model management.
    """

    def __init__(
        self,
        model_type: str = "jafar",
        backbone: str = "vit_large_patch16_dinov3.lvd1689m",
        model_path: str = None,
        device: str = "cuda",
        n_pca: int = 16,
        k: int = 32,
    ):
        """
        Initialize FeatureProcessor with specified model type.

        Args:
            model_type: "jafar" or "openclip"
            backbone: Model architecture name (e.g., "vit_large_patch16_dinov3.lvd1689m" for JAFAR, "ViT-B-16" for OpenCLIP)
            model_path: Path to model checkpoint (required for JAFAR, optional for OpenCLIP)
            device: Device to run inference on
            n_pca: Number of PCA components for quantization
            k: Number of K-means clusters for quantization
        """
        self.device = device
        self.model_type = model_type.lower()
        self.backbone = backbone
        self.model_path = model_path
        self.n_pca = n_pca
        self.k = k

        # Initialize FeatureExtractor based on model type
        self._setup_extractor()

        print("=" * 60)
        print(
            f"\033[92m[FeatureProcessor] {self.model_type.upper()} model initialized successfully\033[0m"
        )
        if self.model_type == "jafar":
            print(
                f"\033[92m[FeatureProcessor] Backbone: {self.backbone}, Model path: {self.model_path}\033[0m"
            )
        elif self.model_type == "openclip":
            print(f"\033[92m[FeatureProcessor] Architecture: {self.backbone}\033[0m")
        print(
            f"\033[92m[K-means] K-means initialized with {self.k} clusters and {self.n_pca} PCA components\033[0m"
        )
        print("=" * 60)

    def _setup_extractor(self):
        """Initialize FeatureExtractor based on model type."""
        config = FeatureExtractorConfig(
            model_type=self.model_type,
            model_architecture=self.backbone,
            model_path=self.model_path,
            device=self.device,
        )

        if self.model_type == "jafar":
            self.extractor = FeatureExtractorJAFAR(config)
        elif self.model_type == "openclip":
            self.extractor = FeatureExtractorOpenCLIP(config)
        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. Must be 'jafar' or 'openclip'"
            )

        # For backward compatibility, expose these attributes
        # Note: FeatureExtractor handles model management internally
        self.model = None  # Deprecated, use extractor instead
        self.backbone_model = None  # Deprecated, use extractor instead

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the configured FeatureExtractor (JAFAR or OpenCLIP).

        Args:
            image: [H, W, 3] tensor with values in [0, 255]
        Returns:
            features: [H', W', C] tensor
        """
        # Ensure proper input format: FeatureExtractor expects (H, W, 3) [0-255]
        features = self.extractor.extract_features(image[0])[0]
        print(f"features shape: {features.shape}")
        features = features.mean(dim=(0, 1))  # [C]

        print(
            f"\033[92m[FeatureProcessor] Extracted features shape: {features.shape}\033[0m"
        )
        features /= features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def quantized_feature(
        self, image: torch.Tensor, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract high-resolution features, apply PyTorch PCA reduction, then quantize using cuML K-means clustering.

        Args:
            image: [B, H, W, 3] or [H, W, 3] tensor with values in [0, 255]
            random_state: Random state for cuML K-means reproducibility

        Returns:
            Tuple of:
            - cluster_centers: [K, C] tensor - cluster centers in original feature space
            - quantized_features: [H', W', C] tensor - quantized features in original feature space
            - original_features: [H', W', C] tensor - original high-res features
            - cluster_labels: [H', W'] tensor - cluster assignment labels
        """
        # Extract high-resolution features first using FeatureExtractor
        hr_features = self.extract_features(image)  # [B, H', W', C] or [H', W', C]

        # Ensure we have the right dimensions
        if hr_features.dim() == 4:
            hr_features = hr_features.squeeze(0)  # Remove batch dimension: [H', W', C]

        H, W, C = hr_features.shape

        # Flatten features for PCA: [H*W, C]
        features_flat = hr_features.reshape(-1, C)  # [H*W, C]

        # Apply PyTorch PCA to reduce dimensions from C to n_pca
        features_pca_flat, pc_directions, pca_mean = pca_torch_extract_directions(
            features_flat, n_components=self.n_pca
        )  # [H*W, n_pca]

        # Apply cuML K-means clustering on PCA-reduced features (GPU-accelerated)
        kmeans = cuKMeans(
            n_clusters=self.k, random_state=random_state, n_init=10, max_iter=300
        )
        cluster_labels_flat = kmeans.fit_predict(
            features_pca_flat.cpu().numpy()
        )  # [H*W]

        # Get cluster centers in PCA space: [K, n_pca]
        cluster_centers_pca = (
            torch.from_numpy(kmeans.cluster_centers_).float().to(hr_features.device)
        )

        # Transform cluster centers back to original C-dimensional space
        cluster_centers = cluster_centers_pca @ pc_directions + pca_mean  # [K, C]

        # Reshape cluster labels back to spatial dimensions: [H', W']
        cluster_labels = torch.from_numpy(cluster_labels_flat).long().reshape(H, W)

        # Create quantized features in original C-dimensional space
        quantized_features = cluster_centers[cluster_labels]  # [H', W', C]

        return cluster_centers, quantized_features, hr_features, cluster_labels


# =============================================================================
# Gradio UI State Holder
# =============================================================================


class GradioUI:
    """
    Container for UI state and heavy objects (e.g., FeatureProcessor, GT/colmap db).
    Keeps callbacks and shared resources in one place.
    """

    def __init__(
        self,
        on_submit: Optional[Callable[[QueryData], None]],
        pca_directions: Optional[torch.Tensor],
        pca_mean: Optional[torch.Tensor],
        feature_processor: Optional[FeatureProcessor] = None,
        gt_db: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.on_submit = on_submit
        self.pca_directions = pca_directions
        self.pca_mean = pca_mean
        self.gt_db_path = gt_db  # Original path to COLMAP database
        self.gt_db = None  # Will be set to Dataset instance after setup
        self.gt_names: List[str] = []  # List of image names for dropdown
        self.device = device
        self.feature_processor: Optional[FeatureProcessor] = feature_processor
        # Setup GT DB if path is provided
        if self.gt_db_path is not None:
            self.setup_gt_db()

    def setup_gt_db(self):
        """Initialize COLMAP database and extract image names."""
        if self.gt_db_path is None:
            print(f"\033[33m[GradioUI] No GT DB provided, skipping GT DB setup\033[0m")
            return

        parser = Parser(self.gt_db_path, test_every=10000000)  # use all data for query
        dataset = Dataset(parser, split="train")
        self.gt_names = parser.image_names
        self.gt_db = dataset
        print(
            f"\033[92m[GradioUI] GT DB setup successfully with {len(self.gt_db)} images\033[0m"
        )
        print(
            f"\033[92m[GradioUI] Available GT images: {len(self.gt_names)} names\033[0m"
        )

    def ensure_feature_processor(
        self,
        model_path: str,
        model_type: str = "jafar",
        backbone: str = "vit_large_patch16_dinov3.lvd1689m",
    ) -> None:
        """
        Ensure FeatureProcessor is initialized with the specified configuration.

        Args:
            model_path: Path to model checkpoint (required for JAFAR, optional for OpenCLIP)
            model_type: "jafar" or "openclip"
            backbone: Model architecture name
        """
        if model_type == "jafar" and not model_path:
            # JAFAR requires model_path, but OpenCLIP doesn't
            return

        # Check if we need to reinitialize the processor
        needs_reinit = (
            self.feature_processor is None
            or self.feature_processor.model_type != model_type.lower()
            or self.feature_processor.model_path != model_path
            or self.feature_processor.backbone != backbone
        )

        if needs_reinit:
            print(
                f"[GradioUI] Initializing FeatureProcessor: {model_type.upper()} with {backbone} @ {model_path}"
            )
            self.feature_processor = FeatureProcessor(
                model_type=model_type,
                backbone=backbone,
                model_path=model_path,
                device=self.device,
            )

    # ---- Handlers wired to Gradio ----
    def process_image_query(
        self,
        img: Optional[Image.Image],
        local_path: str,
        model_path: str,
        model_type: str = "jafar",
        backbone: str = "vit_large_patch16_dinov3.lvd1689m",
    ) -> Tuple[
        Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]
    ]:
        """
        Process image query with feature extraction.

        Args:
            img: Uploaded PIL Image
            local_path: Local image file path
            model_path: Path to model checkpoint
            model_type: "jafar" or "openclip"
            backbone: Model architecture name

        Returns:
            Tuple of (image, info_text, pca_img_quantized, pca_img_original)
        """
        # Load image from upload or local path
        if img is None and local_path:
            img = load_image_from_path(local_path)
        if img is None:
            return (
                None,
                "No image provided. Please upload an image or provide a local path.",
                None,
                None,
            )

        shape = _pil_shape(img)
        pca_img_quantized = None
        pca_img_original = None
        pca_info = "PCA visualization not available"

        # Ensure feature processor is initialized
        self.ensure_feature_processor(
            model_path, model_type=model_type, backbone=backbone
        )
        if self.feature_processor is None:
            return (
                img,
                f"{model_type.upper()} feature extraction not available. Please provide model path.",
                None,
                None,
            )

        feature_token = extract_image_feature_token(img, self.feature_processor)

        if self.pca_directions is not None and self.pca_mean is not None:
            try:
                cluster_centers, pca_img_quantized, pca_img_original, pca_info = (
                    process_quantized_jafar_pca(
                        img,
                        self.feature_processor,
                        self.pca_directions,
                        self.pca_mean,
                        shape,
                    )
                )
            except Exception as e:
                print(f"[GradioUI] PCA processing failed: {e}")
                pca_info = f"PCA visualization failed: {e}"

        if self.on_submit is not None:
            q = QueryData(
                query_type="image_feature",
                image_feature_token=feature_token,
                gt_camera=None,
            )
            self.on_submit(q)

        info_text = f"Image shape (H, W, C): {shape}\nFeature token shape: {feature_token.shape}\n{pca_info}"
        return img, info_text, pca_img_quantized, pca_img_original

    def process_gt_selection(
        self, gt_index: int
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Process ground truth selection by index.

        Args:
            gt_index: Index into gt_names list and gt_db dataset

        Returns:
            Tuple of (parsed image, info text)
        """
        if self.gt_db is None:
            return None, "GT database not initialized. Please provide a GT DB path."

        if gt_index < 0 or gt_index >= len(self.gt_names):
            return (
                None,
                f"Invalid GT index: {gt_index}. Must be between 0 and {len(self.gt_names)-1}.",
            )

        # Retrieve data from dataset using index
        data = self.gt_db[gt_index]
        self.on_submit(QueryData(query_type="gt_camera_pose", gt_camera=data))

        # Convert torch tensor image [H, W, 3] to PIL Image for Gradio
        image_tensor = data["image"]
        if isinstance(image_tensor, torch.Tensor):
            # Clamp, convert to uint8 and numpy
            image_np = image_tensor.cpu().numpy()
            # Sometimes tensor could be float 0-255 or 0-1; assume 0-255 here, cast & clip
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            if image_np.shape[-1] == 1:
                # Grayscale, expand to RGB
                image_np = np.repeat(image_np, 3, axis=-1)
            pil_img = Image.fromarray(image_np)
        else:
            pil_img = image_tensor  # Presume already PIL

        Info_text = f"Image shape (H, W, C): {image_tensor.shape}"
        Info_text += "Image Name: " + data["image_name"] + " "
        Info_text += "Image id: " + str(data["image_id"]) + " "

        return pil_img, Info_text


# =============================================================================
# PCA Visualization Functions
# =============================================================================


def process_jafar_pca(
    img: Image.Image,
    feature_processor: FeatureProcessor,
    pca_directions: torch.Tensor,
    pca_mean: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> Tuple[Image.Image, str]:
    """
    Process image using JAFAR feature extraction and PCA visualization.

    Args:
        img: Input PIL Image
        feature_processor: JAFAR feature processor
        pca_directions: PCA directions tensor
        pca_mean: PCA mean tensor
        target_shape: Target shape (H, W, C) for resizing

    Returns:
        Tuple of (PCA visualization image, info string)
    """
    print("[JAFAR] Extracting features using JAFAR...")

    # Convert PIL to tensor
    tensor_img = pil_to_tensor(img)

    # Convert to format expected by feature processor
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0)

    features = feature_processor.extract_features(tensor_img).squeeze(0)  # [H', W', C]

    # Reshape features to [H'*W', C] for PCA processing
    H_feat, W_feat, C_feat = features.shape
    features_flat = features.reshape(-1, C_feat)  # [H'*W', C_feat]

    # Apply PCA directions to get color visualization
    projected_features = apply_pca_directions_torch(
        features_flat, pca_directions, pca_mean
    )

    # Normalize to [0, 1] range
    mins = projected_features.min(dim=0).values
    maxs = projected_features.max(dim=0).values
    ranges = maxs - mins
    eps = 1e-8
    projected_features = (projected_features - mins) / (ranges + eps)

    # Reshape back to [H', W', 3] and resize to original image size
    pca_array = projected_features.reshape(H_feat, W_feat, 3).cpu().numpy()
    pca_array = (pca_array * 255).astype(np.uint8)
    pca_img = Image.fromarray(pca_array)

    # Resize to match original image dimensions
    pca_img = pca_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)

    pca_info = f"JAFAR PCA visualization generated using {pca_directions.shape[0]} components (feature shape: {features.shape})"

    return pca_img, pca_info


def process_quantized_jafar_pca(
    img: Image.Image,
    feature_processor: FeatureProcessor,
    pca_directions: torch.Tensor,
    pca_mean: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Image.Image, Image.Image, str]:
    """
    Process image using feature extraction (JAFAR or OpenCLIP), quantized features, and PCA visualization.
    Returns both quantized and original feature PCA visualizations.

    Args:
        img: Input PIL Image
        feature_processor: FeatureProcessor (supports JAFAR or OpenCLIP)
        pca_directions: PCA directions tensor
        pca_mean: PCA mean tensor
        target_shape: Target shape (H, W, C) for resizing

    Returns:
        Tuple of (cluster_centers, quantized PCA image, original PCA image, info string)
    """
    # Convert PIL to tensor in [H, W, 3] format with values in [0, 255]
    tensor_img = pil_to_tensor(img)

    # quantized_feature expects [H, W, 3] with values in [0, 255]
    # No normalization needed - pil_to_tensor already provides [0, 255] values

    # Extract quantized features using K-means
    cluster_centers, quantized_features, original_features, cluster_labels = (
        feature_processor.quantized_feature(tensor_img)
    )

    # Process quantized features for PCA visualization
    H_feat, W_feat, C_feat = quantized_features.shape
    quantized_features_flat = quantized_features.reshape(-1, C_feat)  # [H'*W', C_feat]

    # Apply PCA directions to quantized features
    projected_quantized = apply_pca_directions_torch(
        quantized_features_flat, pca_directions, pca_mean
    )

    # Normalize quantized features to [0, 1] range
    mins_q = projected_quantized.min(dim=0).values
    maxs_q = projected_quantized.max(dim=0).values
    ranges_q = maxs_q - mins_q
    eps = 1e-8
    projected_quantized = (projected_quantized - mins_q) / (ranges_q + eps)

    # Reshape quantized features back to [H', W', 3] and resize to original image size
    pca_array_quantized = projected_quantized.reshape(H_feat, W_feat, 3).cpu().numpy()
    pca_array_quantized = (pca_array_quantized * 255).astype(np.uint8)
    pca_img_quantized = Image.fromarray(pca_array_quantized)
    pca_img_quantized = pca_img_quantized.resize(
        (target_shape[1], target_shape[0]), Image.BILINEAR
    )

    # Process original features for PCA visualization
    original_features_flat = original_features.reshape(-1, C_feat)  # [H'*W', C_feat]

    # Apply PCA directions to original features
    projected_original = apply_pca_directions_torch(
        original_features_flat, pca_directions, pca_mean
    )

    # Normalize original features to [0, 1] range
    mins_o = projected_original.min(dim=0).values
    maxs_o = projected_original.max(dim=0).values
    ranges_o = maxs_o - mins_o
    projected_original = (projected_original - mins_o) / (ranges_o + eps)

    # Reshape original features back to [H', W', 3] and resize to original image size
    pca_array_original = projected_original.reshape(H_feat, W_feat, 3).cpu().numpy()
    pca_array_original = (pca_array_original * 255).astype(np.uint8)
    pca_img_original = Image.fromarray(pca_array_original)
    pca_img_original = pca_img_original.resize(
        (target_shape[1], target_shape[0]), Image.BILINEAR
    )

    model_type_str = feature_processor.model_type.upper()
    pca_info = f"Quantized {model_type_str} PCA visualization: Cluster Center Shape {cluster_centers.shape}, {feature_processor.k} clusters, {feature_processor.n_pca} PCA components (quantized: {quantized_features.shape}, original: {original_features.shape})"

    return cluster_centers, pca_img_quantized, pca_img_original, pca_info


def process_fallback_pca(
    img: Image.Image, pca_directions: torch.Tensor, pca_mean: torch.Tensor
) -> Tuple[Image.Image, str]:
    """
    Process image using direct image processing and PCA visualization (fallback method).

    Args:
        img: Input PIL Image
        pca_directions: PCA directions tensor
        pca_mean: PCA mean tensor

    Returns:
        Tuple of (PCA visualization image, info string)
    """
    print("[PCA] Using direct image processing...")
    img_array = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).float()

    # Reshape to [H*W, 3] for PCA processing
    H, W, C = img_tensor.shape
    img_flat = img_tensor.reshape(-1, C)  # [H*W, 3]

    # Apply PCA directions to get color visualization
    projected_features = apply_pca_directions_torch(img_flat, pca_directions, pca_mean)

    # Normalize to [0, 1] range
    mins = projected_features.min(dim=0).values
    maxs = projected_features.max(dim=0).values
    ranges = maxs - mins
    eps = 1e-8
    projected_features = (projected_features - mins) / (ranges + eps)

    # Reshape back to [H, W, 3] and convert to PIL
    pca_array = projected_features.reshape(H, W, 3).cpu().numpy()
    pca_array = (pca_array * 255).astype(np.uint8)
    pca_img = Image.fromarray(pca_array)

    pca_info = f"Direct PCA visualization generated using {pca_directions.shape[0]} components (directions shape: {pca_directions.shape})"

    return pca_img, pca_info


# =============================================================================
# Ground Truth Parsing Functions (Placeholder for implementation)
# =============================================================================


def parse_gt_option(gt_option: str) -> Dict[str, torch.Tensor]:
    """
    Parse ground truth option string into camera pose dictionary.

    Args:
        gt_option: String option selected from dropdown (e.g., "camera_001", "pose_42")

    Returns:
        Dict containing camera pose information:
        - "camera_pose": torch.Tensor (4, 4) - camera extrinsic matrix
        - "K": torch.Tensor (3, 3) - camera intrinsic matrix
        - "height": torch.Tensor (1,) - image height
        - "width": torch.Tensor (1,) - image width

    TODO: Implement actual parsing logic based on your data format
    """
    # Placeholder implementation - replace with actual parsing logic
    print(f"[GT Parser] Parsing ground truth option: {gt_option}")

    # Example: Load from file or database based on gt_option
    # This is a placeholder - implement based on your data structure
    camera_pose = torch.eye(4, dtype=torch.float32)  # Placeholder
    K = torch.eye(3, dtype=torch.float32)  # Placeholder
    height = torch.tensor([512], dtype=torch.float32)
    width = torch.tensor([512], dtype=torch.float32)

    return {"camera_pose": camera_pose, "K": K, "height": height, "width": width}


def get_gt_options(ui: Optional["GradioUI"] = None) -> List[str]:
    """
    Get list of available ground truth options for dropdown.

    Args:
        ui: Optional GradioUI instance to get gt_names from

    Returns:
        List of option strings (e.g., ["camera_001", "camera_002", ...])
    """
    if ui is not None and ui.gt_names:
        return ui.gt_names

    # Fallback: return placeholder options
    return ["camera_001", "camera_002", "camera_003", "pose_001", "pose_002"]


# =============================================================================
# Main Dashboard Function
# =============================================================================


def launch_image_dashboard(
    on_submit: Optional[Callable[[QueryData], None]] = None,
    pca_directions: Optional[torch.Tensor] = None,
    pca_mean: Optional[torch.Tensor] = None,
    port: int = 7860,
    server_name: str = "0.0.0.0",
    open_browser: bool = False,
    gt_options: Optional[List[str]] = None,
    gt_db: Optional[Path] = None,
):
    """
    Launch a Gradio dashboard with two submit modes:
    1. Image Query: Extract features from uploaded image
    2. Ground Truth Selection: Select GT camera pose from dropdown

    Args:
        on_submit: Callback function that receives QueryData dict
        pca_directions: Optional PCA directions for visualization
        pca_mean: Optional PCA mean for visualization
        port: Port number for Gradio server
        server_name: Server address
        open_browser: Whether to open browser automatically
        gt_options: List of ground truth options for dropdown (if None, uses get_gt_options())

    Returns:
        Tuple of (thread, url)
    """
    # Initialize default feature processor (can be changed via UI)
    # Default to JAFAR, but users can switch via UI
    feature_processor = FeatureProcessor(
        model_type="jafar",
        backbone="vit_large_patch16_dinov3.lvd1689m",
        model_path="/data1/vfms/vit_large_patch16_dinov3.lvd1689m.pth",
        device="cuda",
    )

    # Centralized UI state
    ui = GradioUI(
        on_submit=on_submit,
        pca_directions=pca_directions,
        pca_mean=pca_mean,
        gt_db=gt_db,
        feature_processor=feature_processor,
    )

    # Get GT options from UI's gt_names if available, otherwise use provided or default
    if gt_options is None:
        gt_options = get_gt_options(ui)

    # Create dropdown choices with index mapping: display names but use indices
    # Gradio dropdown can use tuples: (display_name, index_value)
    # But we'll use a simpler approach: store index as string and convert
    gt_dropdown_choices = (
        [f"{i}: {name}" for i, name in enumerate(gt_options)] if gt_options else []
    )

    # =========================================================================
    # Image Query Processing Function
    # =========================================================================
    def _process_image_query(
        img: Optional[Image.Image],
        local_path: str,
        model_path: str,
        model_type: str,
        backbone: str,
    ):
        return ui.process_image_query(
            img, local_path, model_path, model_type=model_type, backbone=backbone
        )

    # =========================================================================
    # Gradio Interface
    # =========================================================================
    with gr.Blocks(title="Image Query & Ground Truth Dashboard") as demo:
        gr.Markdown(
            "## Image Query & Ground Truth Selection Dashboard\n"
            "Upload an image or provide local path for feature extraction, or select a ground truth camera pose."
        )

        # Image Query Tab
        with gr.Tab("Image Query"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Image Input")
                    img_upload = gr.Image(type="pil", label="Upload Image")
                    local_path = gr.Textbox(
                        label="Local Image Path",
                        placeholder="/path/to/image.jpg",
                        value="",
                    )

                    with gr.Group():
                        gr.Markdown("### Model Configuration")
                        model_type = gr.Dropdown(
                            choices=["jafar", "openclip"],
                            value="jafar",
                            label="Model Type",
                            info="Select JAFAR or OpenCLIP for feature extraction",
                        )
                        backbone = gr.Textbox(
                            label="Model Architecture",
                            placeholder="vit_large_patch16_dinov3.lvd1689m (for JAFAR) or ViT-B-16 (for OpenCLIP)",
                            value="vit_large_patch16_dinov3.lvd1689m",
                            info="Backbone architecture name",
                        )
                        model_path = gr.Textbox(
                            label="Model Path (Optional for OpenCLIP)",
                            placeholder="Path to model checkpoint (required for JAFAR)",
                            value="",
                            info="Required for JAFAR models, optional for OpenCLIP (will download if not provided)",
                        )

                    btn_image_query = gr.Button("Submit Image Query", variant="primary")

                with gr.Column():
                    out_img_query = gr.Image(label="Input Image")
                    out_txt_query = gr.Textbox(label="Info", lines=8)

            with gr.Row():
                with gr.Column():
                    out_pca_img_quantized = gr.Image(
                        label="Quantized PCA Visualization"
                    )
                with gr.Column():
                    out_pca_img_original = gr.Image(label="Original PCA Visualization")

        # Ground Truth Selection Tab
        with gr.Tab("Ground Truth Selection"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Select Ground Truth Camera Pose")
                    gt_dropdown = gr.Dropdown(
                        choices=gt_dropdown_choices,
                        label="Ground Truth Option",
                        value=gt_dropdown_choices[0] if gt_dropdown_choices else None,
                        interactive=True,
                        info="Select an image from the COLMAP database",
                    )
                    btn_gt_submit = gr.Button("Submit GT Selection", variant="primary")

                with gr.Column():
                    out_img_gt = gr.Image(label="Parsed Image / Camera View")
                    out_txt_gt = gr.Textbox(label="GT Info", lines=8)

        # Bind events
        btn_image_query.click(
            _process_image_query,
            inputs=[img_upload, local_path, model_path, model_type, backbone],
            outputs=[
                out_img_query,
                out_txt_query,
                out_pca_img_quantized,
                out_pca_img_original,
            ],
        )

        def _process_gt_with_index(gt_choice: str):
            """Extract index from dropdown choice and call process_gt_selection."""
            if not gt_choice or ":" not in gt_choice:
                return None, "Invalid selection. Please select a valid option."
            # Extract index from "index: name" format
            try:
                gt_index = int(gt_choice.split(":")[0])
                return ui.process_gt_selection(gt_index)
            except (ValueError, IndexError) as e:
                return None, f"Error parsing selection: {e}"

        btn_gt_submit.click(
            _process_gt_with_index,
            inputs=[gt_dropdown],
            outputs=[out_img_gt, out_txt_gt],
        )

    # Launch server
    def _run():
        demo.launch(
            server_name=server_name,
            server_port=port,
            share=False,
            inbrowser=open_browser,
            show_error=True,
        )

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    url = f"http://localhost:{port}"
    print(f"[Gradio] Image dashboard running at {url}")
    return th, url
