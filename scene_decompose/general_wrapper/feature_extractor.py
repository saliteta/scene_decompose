"""
We directlu move on to the JAFAR pipeline
We also support OpenCLIP pipeline

Input of the system is a single image, and the output is a torch tensor of shape [H', W', C]
"""

from abc import ABC, abstractmethod
from typing import Literal, List, Callable
import torch
from dataclasses import dataclass, field
import torchvision.transforms as T
import os
from jafar import load_model
import open_clip
@dataclass
class FeatureExtractorConfig:
    """
    This is most similar to C structs/parameter groups!
    All fields are defined upfront, with types and default values.
    Benefits:
    - Type safety
    - Auto-generated __init__ and __repr__
    - Immutability option (frozen=True)
    - Clear parameter documentation
    """
    model_type: Literal["jafar", "openclip"] = "jafar"
    model_architecture: str = "vit_large_patch16_dinov3.lvd1689m"
    model_path: str|None = None
    device: str = "cuda"
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])



class FeatureExtractor(ABC):
    """
    Using a config object (like C parameter structs passed to functions).
    This separates configuration from behavior.
    """
    
    # Type hints for attributes that must be set by subclasses
    model: Callable[[torch.Tensor], torch.Tensor]  # Takes transformed tensor, returns (B, H, W, C)
    transform: Callable[[torch.Tensor], torch.Tensor]  # Takes (H, W, 3) [0-255], returns model input
    
    def __init__(self, config: FeatureExtractorConfig):
        self.config = config
        self.model_path = config.model_path
        self.setup_model()
        self.setup_transform()
    
    @abstractmethod
    def setup_model(self) -> None:
        """
        Setup the model callable.
        
        MUST set self.model to a callable that:
        - Takes input: torch.Tensor (transformed image, shape depends on model requirements)
        - Returns: torch.Tensor of shape (B, H, W, C) where:
          * B: batch dimension (usually 1 for single image)
          * H: height of feature map
          * W: width of feature map  
          * C: number of feature channels
        
        Example:
            self.model = torch.nn.Module(...)  # or any callable with above signature
        """
        pass
    
    @abstractmethod
    def setup_transform(self) -> None:
        """
        Setup the transform callable.
        
        MUST set self.transform to a callable that:
        - Takes input: torch.Tensor of shape (H, W, 3) with pixel values in [0, 255]
        - Returns: torch.Tensor that can be fed directly into self.model
          (typically normalized, converted to CHW format, etc.)
        
        Example:
            self.transform = T.Compose([
                T.ToTensor(),  # Converts [0, 255] HWC -> [0, 1] CHW
                T.Normalize(mean=[...], std=[...])
            ])
        """
        pass
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the image.
        Args:
            image: torch.Tensor of shape (H, W, 3) with pixel values in [0, 255]
        Returns:
            features: torch.Tensor, the extracted features
        """
        # Assert input shape is (H, W, 3)
        assert image.dim() == 3 and image.shape[2] == 3, \
            f"Expected shape (H, W, 3), got {tuple(image.shape)}"
        
        # Assert pixel values are in [0, 255] range (not normalized)
        min_val, max_val = image.min().item(), image.max().item()
        assert min_val >= 0 and max_val <= 255, \
            f"Expected pixel range [0, 255], got [{min_val:.2f}, {max_val:.2f}]"
        assert max_val > 1.0, \
            f"Image appears normalized to [0, 1]. Expected raw pixels in [0, 255]"
        image = self.transform(image)
        features: torch.Tensor = self.model(image)
        return features 



class FeatureExtractorJAFAR(FeatureExtractor):
    def setup_model(self) -> None:
        """
        Setup JAFAR model and backbone.
        Creates a callable wrapper that combines backbone + model calls.
        """
        project_root = os.getcwd()
        feature_expander, backbone_model = load_model(
            self.config.model_architecture, project_root, self.config.model_path
        )
        feature_expander.eval()
        backbone_model.eval()
        self.feature_expander = feature_expander.to(self.config.device)
        self.backbone_model = backbone_model.to(self.config.device)
        
        # Create a callable wrapper that matches the JAFAR pipeline pattern
        # This wrapper takes transformed image (CHW format) and returns (B, H, W, C) features
        def jafar_model_wrapper(transformed_image: torch.Tensor) -> torch.Tensor:
            """
            Wrapper that implements: lr_feats, _ = backbone(image); hr_feats = model(image, lr_feats, size)
            Args:
                transformed_image: torch.Tensor in CHW format (from transform)
            Returns:
                features: torch.Tensor of shape (B, H, W, C)
            """
            # Ensure batch dimension: CHW -> BCHW
            if transformed_image.dim() == 3:
                transformed_image = transformed_image.unsqueeze(0)
            transformed_image = transformed_image.to(self.config.device)
            # Extract low-resolution features using backbone
            lr_feats, _ = self.backbone_model(transformed_image)
            
            # Extract high-resolution features using model with backbone features
            # Model signature: model(image, lr_feats, size_tuple)
            hr_feats = self.feature_expander(transformed_image, lr_feats, (448, 448))
            
            # Convert from (B, C, H, W) to (B, H, W, C) format
            hr_feats = hr_feats.permute(0, 2, 3, 1)
            
            return hr_feats
        
        self.model = jafar_model_wrapper
    
    def setup_transform(self) -> None:
        """
        Setup transform that converts (H, W, 3) [0-255] to CHW format normalized [0-1].
        Note: The transform should output CHW format for JAFAR models.
        """
        def hwc_to_chw_normalize(image: torch.Tensor) -> torch.Tensor:
            """
            Convert (H, W, 3) [0-255] to (C, H, W) [0-1] normalized.
            """
            # Convert HWC -> CHW
            image = image.permute(2, 0, 1)
            # Normalize from [0, 255] to [0, 1]
            image = image / 255.0
            return image
        
        self.transform = T.Compose([
            T.Lambda(hwc_to_chw_normalize),  # Convert HWC [0-255] -> CHW [0-1]
            T.Resize((448, 448)),            # Resize to model input size
            T.Normalize(mean=self.config.mean, std=self.config.std),  # Normalize with ImageNet stats
        ])


class FeatureExtractorOpenCLIP(FeatureExtractor):
    def setup_model(self) -> None:
        """
        Setup OpenCLIP model.
        Creates a callable wrapper that extracts visual features.
        """
        model, _, openclip_transform = open_clip.create_model_and_transforms(
            self.config.model_architecture, pretrained="laion2b_s34b_b88k"
        )
        model.eval()
        model = model.to(self.config.device)
        # Store the original transform to filter it in setup_transform
        self._openclip_transform = openclip_transform
        
        # Create a callable wrapper that extracts features from transformed image
        def openclip_model_wrapper(transformed_image: torch.Tensor) -> torch.Tensor:
            """
            Extract visual features from transformed image using OpenCLIP.
            Args:
                transformed_image: torch.Tensor in CHW format (from transform)
            Returns:
                features: torch.Tensor of shape (B, H, W, C)
            """
            # Ensure batch dimension: CHW -> BCHW
            if transformed_image.dim() == 3:
                transformed_image = transformed_image.unsqueeze(0)
            transformed_image = transformed_image.to(self.config.device)
            
            # Get spatial dimensions from transformed image
            B, C_img, H_img, W_img = transformed_image.shape
            
            # Extract visual features - OpenCLIP returns global feature vector [B, C]
            with torch.no_grad():
                visual_features = model.encode_image(transformed_image).squeeze(0)
                visual_features /= visual_features.norm(dim=-1, keepdim=True)
                visual_features = visual_features.unsqueeze(0)
            # visual_features is [B, C] - a single global feature vector
            # Repeat it spatially to get [B, H, W, C]
            B_feat, C = visual_features.shape
            # Expand to [B, 1, 1, C] then repeat to [B, H, W, C]
            features = visual_features.view(B_feat, 1, 1, C).expand(B_feat, 448, 448, C)
            
            return features
        
        self.model = openclip_model_wrapper
    
    def setup_transform(self) -> None:
        """
        Filter OpenCLIP transform to remove PIL-specific operations.
        Removes _convert_to_rgb and ToTensor(), adds tensor preprocessing.
        """
        def hwc_to_chw_normalize(image: torch.Tensor) -> torch.Tensor:
            """
            Convert (H, W, 3) [0-255] to (C, H, W) [0-1] normalized.
            This replaces the PIL -> tensor conversion.
            """
            # Convert HWC -> CHW
            image = image.permute(2, 0, 1)
            # Normalize from [0, 255] to [0, 1]
            image = image / 255.0
            return image
        
        # Get the transforms from the OpenCLIP Compose
        # Filter out PIL-specific transforms (_convert_to_rgb and ToTensor)
        filtered_transforms = []
        
        # Check if it's a Compose object and has a transforms attribute
        if hasattr(self._openclip_transform, 'transforms'):
            for transform in self._openclip_transform.transforms:
                # Get transform identifiers for filtering
                transform_name = getattr(transform, '__name__', '')
                transform_type_name = type(transform).__name__
                transform_repr = str(transform)
                
                # Skip _convert_to_rgb (can be a function, lambda, or method)
                # Check by name, type name, or string representation
                if ('_convert_to_rgb' in transform_name or 
                    '_convert_to_rgb' in transform_repr or
                    'convert' in transform_name.lower() and 'rgb' in transform_name.lower()):
                    continue
                
                # Skip ToTensor() (converts PIL Image to tensor, we already have a tensor)
                if isinstance(transform, T.ToTensor) or transform_type_name == 'ToTensor':
                    continue
                
                # Keep all other transforms (Resize, Normalize, CenterCrop, etc.)
                filtered_transforms.append(transform)
        else:
            # If not a Compose, try to use it as-is (less common)
            filtered_transforms = [self._openclip_transform]
        
        # Create new transform pipeline:
        # 1. Convert HWC [0-255] -> CHW [0-1] (replaces PIL conversion + ToTensor)
        # 2. Apply filtered OpenCLIP transforms (Resize, Normalize, etc.)
        self.transform = T.Compose([
            T.Lambda(hwc_to_chw_normalize),  # Tensor preprocessing
            *filtered_transforms             # Remaining OpenCLIP transforms
        ])


if __name__ == "__main__":
    config = FeatureExtractorConfig(model_type="openclip", 
    model_architecture="ViT-B-16", 
    device="cuda")
    extractor = FeatureExtractorOpenCLIP(config)
    # Input should be (H, W, 3) format with values in [0, 255] range
    image = torch.randint(0, 256, (224, 224, 3), dtype=torch.float32)
    features = extractor.extract_features(image)
    print(features.shape)