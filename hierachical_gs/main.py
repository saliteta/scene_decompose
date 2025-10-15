import os
import struct
import numpy as np
import torch
from gsplat_ext import GaussianPrimitive
from tree.pca_tree import PCABinaryTree
from splatLayer.splat_self_attention import SplatAttention
from splatCondenseNet import SplatCondenseNet

def save_tensor_to_ply_binary(tensor, filepath):
    """
    Save a PyTorch tensor of shape [N, 3] to PLY binary format.
    
    Args:
        tensor: PyTorch tensor of shape [N, 3] containing xyz coordinates
        filepath: Output file path for the PLY file
    """
    # Ensure tensor is on CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    xyz = tensor.numpy().astype(np.float32)
    num_points = xyz.shape[0]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write PLY binary format
    with open(filepath, 'wb') as f:
        # Write PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
        f.write(header.encode('ascii'))
        
        # Write binary data (little-endian float32)
        for point in xyz:
            f.write(struct.pack('<fff', point[0], point[1], point[2]))





    # Example 1: Simple mean pooling
def mean_pooling(correspondence_features):
    """
    Args:
        correspondence_features: [N_parent, 2, C]
    Returns:
        processed_features: [N_parent, C]
    """
    return correspondence_features.mean(dim=1)  # Average over children

# Example 2: Attention pooling using our SplatAttention
attention = SplatAttention()

def attention_pooling(correspondence_features):
    """
    Args:
        correspondence_features: [N_parent, 2, C]
    Returns:
        processed_features: [N_parent, C]
    """
    N, num_children, C = correspondence_features.shape
    
    # Reshape for attention: [N*2, C] -> attention -> [N*2, C] -> reshape back
    features_flat = correspondence_features.view(-1, C)  # [N*2, C]
    attended_features = attention(features_flat)  # [N*2, C]
    attended_features = attended_features.view(N, num_children, C)  # [N, 2, C]
    
    # Average the attended features
    return attended_features.mean(dim=1)  # [N, C]


if __name__ == "__main__":
    # Synthetic example
    gs = GaussianPrimitive()
    gs.from_file("/data1/vpr/CUHK_LOWER/ckpts/ckpt_6998_rank0.pt", feature_path="/data1/vpr/CUHK_LOWER/ckpts/ckpt_6998_rank0_features.pt")


    xyz = gs.geometry["means"]
    features = gs.feature

    tree = PCABinaryTree(xyz, features=features)  # leaves at 0, root at 5


    location = tree.get_layer_locations(0)
    print(f"   Layer 0: {location.shape}")
    print(location.mean(dim=0), location.min(dim=0), location.max(dim=0))

    print("\n3. Processing with SplatCondenseNet...")
    condense_net = SplatCondenseNet(tree, attention_start_layer=3, hops_for_attention=3)
    processed_tree = condense_net()
    
    print("\n✓ SplatCondenseNet processing completed!")
    
    # 4. Save layer locations as PLY binary files and features as torch tensors
    print("\n4. Saving layer data...")
    
    # Create output directories
    os.makedirs("tree_structure", exist_ok=True)
    os.makedirs("layer_features", exist_ok=True)
    
    # Save each layer's data
    tree_info = processed_tree.get_tree_info()
    print(f"\nSaving {tree_info['num_layers']} layers:")
    
    for layer_info in tree_info['layers']:
        layer_id = layer_info['layer_id']
        
        # Get layer locations and features
        layer_locations = processed_tree.get_layer_locations(layer_id)
        layer_features = processed_tree.get_layer_features(layer_id)
        
        # Save locations as PLY binary
        ply_path = f"tree_structure/layer_{layer_id}.ply"
        save_tensor_to_ply_binary(layer_locations, ply_path)
        print(f"   Layer {layer_id}: {layer_locations.shape} -> {ply_path}")
        # Save features as torch tensor
        features_path = f"layer_features/layer_{layer_id}_features.pt"
        torch.save(layer_features, features_path)
        print(f"   Layer {layer_id}: {layer_features.shape} -> {features_path}")
    
 
    