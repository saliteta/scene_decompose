# image_dashboard.py
from typing import Callable, Optional, Tuple
import threading
from PIL import Image
import numpy as np
import gradio as gr
import torch
import torchvision.transforms as T
from .pca_utils import apply_pca_directions_torch, pca_torch_extract_directions
from jafar import load_model
import os
from cuml.cluster import KMeans as cuKMeans

def _pil_shape(img: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    return arr.shape  # (H, W, 3)

class FeatureProcessor:
    """JAFAR-based feature extraction processor."""
    
    def __init__(self, backbone: str = "vit_large_patch16_dinov3.lvd1689m", 
    model_path: str = None, device: str = "cuda", n_pca: int = 16, k: int = 32):
        self.device = device
        self.backbone = backbone
        self.model_path = model_path
        self.model = None
        self.backbone_model = None
        self.transform = None
        self.n_pca = n_pca
        self.k = k
        self._setup_transform()
        self._setup_model()
        print("="*60)
        
    
    def _setup_model(self):
        """Initialize JAFAR model and backbone."""
        project_root = os.getcwd()
        self.model, self.backbone_model = load_model(
            self.backbone, project_root, self.model_path
        )
        self.model.eval()
        self.backbone_model.eval()
        self.model = self.model.to(self.device)
        self.backbone_model = self.backbone_model.to(self.device)
        print(f"\033[92m[JAFAR] Model initialized successfully with backbone: {self.backbone} and model path: {self.model_path}\033[0m")
        print(f"\033[92m[K-means] K-means initialized successfully with {self.k} clusters and {self.n_pca} PCA components\033[0m")
    
    def _setup_transform(self):
        """Setup image preprocessing transform."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.Normalize(mean=mean, std=std),
        ])
    
    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor: # [B, H, W, 3] -> [B, H', W', C]
        """
        JAFAR-specific feature extraction pipeline.
        Args:
            image: [B, H, W, 3] or [H, W, 3] tensor
        Returns:
            features: [B, H', W', C] or [H', W', C] tensor
        """
        # Ensure proper input format
        image = image.cuda()
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Convert to CHW format and normalize
        image = image.permute(0, 3, 1, 2).squeeze(0)
        image = self.transform(image)
        image = image.unsqueeze(0)
        
        # Extract features using JAFAR
        lr_feats, _ = self.backbone_model(image)
        hr_feats = self.model(image, lr_feats, (448, 448))
        
        # Convert back to HWC format
        return hr_feats.permute(0, 2, 3, 1)
    
    @torch.no_grad()
    def quantized_feature(self, image: torch.Tensor, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract high-resolution features, apply PyTorch PCA reduction, then quantize using cuML K-means clustering.
        
        Args:
            image: [B, H, W, 3] or [H, W, 3] tensor
            random_state: Random state for cuML K-means reproducibility
            
        Returns:
            Tuple of:
            - cluster_centers: [K, C] tensor - cluster centers in original feature space
            - quantized_features: [H, W, C] tensor - quantized features in original feature space
            - original_features: [H, W, C] tensor - original high-res features
            - cluster_labels: [H, W] tensor - cluster assignment labels
        """
        # Extract high-resolution features first
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
        kmeans = cuKMeans(n_clusters=self.k, random_state=random_state, n_init=10, max_iter=300)
        cluster_labels_flat = kmeans.fit_predict(features_pca_flat.cpu().numpy())  # [H*W]
        
        # Get cluster centers in PCA space: [K, n_pca]
        cluster_centers_pca = torch.from_numpy(kmeans.cluster_centers_).float().to(hr_features.device)
        
        # Transform cluster centers back to original C-dimensional space
        # Using the PCA inverse transformation: cluster_centers_pca @ pc_directions.T + pca_mean
        cluster_centers = cluster_centers_pca @ pc_directions + pca_mean  # [K, C]
        
        # Reshape cluster labels back to spatial dimensions: [H, W]
        cluster_labels = torch.from_numpy(cluster_labels_flat).long().reshape(H, W)
        
        # Create quantized features in original C-dimensional space
        quantized_features = cluster_centers[cluster_labels]  # [H, W, C]


        
        return cluster_centers, quantized_features, hr_features, cluster_labels
    

def process_jafar_pca(
    img: Image.Image, 
    feature_processor: FeatureProcessor, 
    pca_directions: torch.Tensor, 
    pca_mean: torch.Tensor,
    target_shape: Tuple[int, int, int]
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
    print(f"Image: {img}")
    
    # Convert PIL Image to numpy array, then to torch tensor
    img_array = np.array(img)  # Convert PIL Image to numpy array
    print(f"Image array shape: {img_array.shape}")
    
    # Convert to torch tensor and ensure proper format [C, H, W]
    if len(img_array.shape) == 3:  # RGB image
        tensor_img = torch.from_numpy(img_array).permute(2, 0, 1).float()  # [C, H, W]
    else:  # Grayscale
        tensor_img = torch.from_numpy(img_array).unsqueeze(0).float()  # [1, H, W]
    
    # Normalize to [0, 1] if needed
    if tensor_img.max() > 1.0:
        tensor_img = tensor_img / 255.0
    
    # Add batch dimension: [1, C, H, W]
    tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
    
    features = feature_processor.extract_features(tensor_img).squeeze(0)  # [H', W', C]
    
    # Reshape features to [H'*W', C] for PCA processing
    H_feat, W_feat, C_feat = features.shape
    features_flat = features.reshape(-1, C_feat)  # [H'*W', C_feat]
    print(f"Features flat shape: {features_flat.shape}")
    print(f"PCA directions shape: {pca_directions.shape}")
    print(f"PCA mean shape: {pca_mean.shape}")
    
    # Apply PCA directions to get color visualization
    projected_features = apply_pca_directions_torch(
        features_flat,
        pca_directions, 
        pca_mean
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
    Process image using JAFAR feature extraction, quantized features, and PCA visualization.
    Returns both quantized and original feature PCA visualizations.
    
    Args:
        img: Input PIL Image
        feature_processor: JAFAR feature processor
        pca_directions: PCA directions tensor
        pca_mean: PCA mean tensor
        target_shape: Target shape (H, W, C) for resizing
        
    Returns:
        Tuple of (quantized PCA image, original PCA image, info string)
    """
    
    # Convert PIL Image to numpy array, then to torch tensor
    img_array = np.array(img)  # Convert PIL Image to numpy array
    
    # Convert to torch tensor and ensure proper format [C, H, W]
    if len(img_array.shape) == 3:  # RGB image
        tensor_img = torch.from_numpy(img_array).permute(2, 0, 1).float()  # [C, H, W]
    else:  # Grayscale
        tensor_img = torch.from_numpy(img_array).unsqueeze(0).float()  # [1, H, W]
    
    # Normalize to [0, 1] if needed
    if tensor_img.max() > 1.0:
        tensor_img = tensor_img / 255.0
    
    # Add batch dimension: [1, C, H, W]
    tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
    
    # Extract quantized features using K-means
    cluster_centers, quantized_features, original_features, cluster_labels = feature_processor.quantized_feature(
        tensor_img
    )
    
    # Process quantized features for PCA visualization
    H_feat, W_feat, C_feat = quantized_features.shape
    quantized_features_flat = quantized_features.reshape(-1, C_feat)  # [H'*W', C_feat]
    
    # Apply PCA directions to quantized features
    projected_quantized = apply_pca_directions_torch(
        quantized_features_flat,
        pca_directions, 
        pca_mean
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
    pca_img_quantized = pca_img_quantized.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    
    # Process original features for PCA visualization
    original_features_flat = original_features.reshape(-1, C_feat)  # [H'*W', C_feat]
    
    # Apply PCA directions to original features
    projected_original = apply_pca_directions_torch(
        original_features_flat,
        pca_directions, 
        pca_mean
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
    pca_img_original = pca_img_original.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    
    pca_info = f"Quantized JAFAR PCA visualization: Cluster Center Shpae {cluster_centers.shape}, {feature_processor.k} clusters, {feature_processor.n_pca} PCA components (quantized: {quantized_features.shape}, original: {original_features.shape})"
    
    return cluster_centers, pca_img_quantized, pca_img_original, pca_info


def process_fallback_pca(
    img: Image.Image, 
    pca_directions: torch.Tensor, 
    pca_mean: torch.Tensor
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


def launch_image_dashboard(
    on_submit: Optional[Callable[[Tuple[int,int,int]], None]] = None,
    pca_directions: Optional[torch.Tensor] = None,
    pca_mean: Optional[torch.Tensor] = None,
    port: int = 7860,
    server_name: str = "0.0.0.0",
    open_browser: bool = False,
):

    """
    Launch a Gradio site that:
      - lets user upload an image
      - shows it back
      - on submit, computes image shape and calls `on_submit(shape)`
      - generates PCA-based color visualization if PCA info is provided
      - allows user to input JAFAR model path through UI
    Returns the started thread and the URL (e.g., http://localhost:7860).
    """

    # Global feature processor that will be initialized in the UI thread
    feature_processor = None

    def _process(img: Image.Image, model_path: str = "", use_jafar: bool = True):
        if img is None:
            return None, "No image uploaded.", None
        
        shape = _pil_shape(img)
        
        # Initialize feature processor if needed
        nonlocal feature_processor
        if use_jafar and model_path and (feature_processor is None or feature_processor.model_path != model_path):
            print(f"[JAFAR] Initializing feature processor with model: {model_path}")
            feature_processor = FeatureProcessor(model_path=model_path)
            print(f"[JAFAR] Feature processor initialized successfully")
        
        # Generate PCA visualization if PCA information is available
        pca_img = None
        pca_info = "PCA visualization not available (no PCA directions provided)"
        
        if pca_directions is not None and pca_mean is not None:
            if use_jafar and feature_processor is not None:
                # Use quantized JAFAR feature extraction
                cluster_centers, pca_img_quantized, pca_img_original, pca_info = process_quantized_jafar_pca(
                    img, feature_processor, pca_directions, pca_mean, shape
                )
            else:
                # Use direct image processing (fallback)
                pca_img_original, pca_info = process_fallback_pca(
                    img, pca_directions, pca_mean
                )
                pca_img_quantized = None
                
        else:
            pca_info = "PCA visualization not available (PCA directions not provided or not computed yet)"
            pca_img_quantized = None
            pca_img_original = None
        
        if on_submit is not None:
            on_submit(cluster_centers)  # notify the parent (HierachicalViewer)
        
        return img, f"Image shape (H, W, C): {shape}\n{pca_info}", pca_img_quantized, pca_img_original

    with gr.Blocks(title="2D Image PCA Dashboard") as demo:
        gr.Markdown("## 2D Image PCA Visualization\nUpload an image to see both quantized and original PCA-based color visualizations using JAFAR features.")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload Image")
                
                # JAFAR configuration
                with gr.Group():
                    gr.Markdown("### JAFAR Configuration")
                    use_jafar = gr.Checkbox(value=True, label="Use JAFAR Feature Extraction")
                    model_path = gr.Textbox(
                        label="JAFAR Model Path", 
                        placeholder="Path to JAFAR model checkpoint",
                        value=""
                    )
                
                btn = gr.Button("Submit")
            with gr.Column():
                out_img = gr.Image(label="Original Image")
                out_txt = gr.Textbox(label="Info", lines=5)
        with gr.Row():
            with gr.Column():
                out_pca_img_quantized = gr.Image(label="Quantized PCA Visualization")
            with gr.Column():
                out_pca_img_original = gr.Image(label="Original PCA Visualization")
        btn.click(_process, inputs=[img_in, model_path, use_jafar], outputs=[out_img, out_txt, out_pca_img_quantized, out_pca_img_original])

    def _run():
        demo.launch(server_name=server_name, server_port=port, share=False, inbrowser=open_browser, show_error=True)

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    url = f"http://localhost:{port}"
    print(f"[Gradio] 2D dashboard running at {url}")
    return th, url


