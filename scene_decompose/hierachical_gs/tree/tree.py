"""
A clean, efficient tree data structure designed for hierarchical feature processing.
This design prioritizes:
1. Easy access to grouped features at different levels
2. Efficient parent-child relationships
3. Clear separation between tree structure and data
4. Support for hierarchical operations like attention pooling
"""

import torch
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import gc


@dataclass
class TreeNode:
    """
    A single node in the tree with clear parent-child relationships.
    Each node has a unique ID and knows its position in the hierarchy.
    """
    node_id: int                    # Unique node ID across the entire tree
    layer_id: int                   # Which layer this node belongs to (0=leaves, depth=root)
    local_id: int                   # Local ID within its layer (0, 1, 2, ...)
    
    # Tree structure
    parent_id: Optional[int] = None     # Parent node ID (None for root)
    children_ids: List[int] = None      # List of child node IDs (empty for leaves)
    
    # Data (optional)
    point_indices: Optional[torch.Tensor] = None  # Which points this node represents
    features: Optional[torch.Tensor] = None       # Node features
    location: Optional[torch.Tensor] = None       # 3D location
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


@dataclass 
class LayerInfo:
    """
    Information about a single layer in the tree.
    """
    layer_id: int
    nodes: List[TreeNode]           # All nodes in this layer
    num_nodes: int                  # Number of nodes in this layer
    
    # Precomputed indices for efficient access
    node_id_to_index: Dict[int, int]  # Maps node_id to index in nodes list
    parent_child_mapping: Dict[int, List[int]]  # Maps parent_id to list of child_ids


class HierarchicalTree:
    """
    A clean, efficient tree data structure designed for hierarchical operations.
    
    Key design principles:
    1. Flat storage with clear indexing
    2. Precomputed mappings for O(1) access
    3. Easy grouping operations for different hierarchy levels
    4. Clear separation between structure and data
    """
    
    def __init__(self, points: torch.Tensor, features: Optional[torch.Tensor] = None, 
                 max_depth: Optional[int] = None, min_points_per_leaf: int = 1):
        """
        Initialize the hierarchical tree.
        
        Args:
            points: [N, 3] tensor of 3D points
            features: [N, C] tensor of features (optional)
            max_depth: Maximum tree depth (auto-computed if None)
            min_points_per_leaf: Minimum points per leaf node
        """
        self.points = points
        self.features = features
        self.num_points = points.shape[0]
        self.feature_dim = features.shape[1] if features is not None else None
        
        # Compute tree depth
        if max_depth is None:
            eff = max(1, np.ceil(self.num_points / min_points_per_leaf).astype(int))
            self.max_depth = int(np.ceil(np.log2(eff)))
        else:
            self.max_depth = max_depth
            
        # Tree structure
        self.layers: List[LayerInfo] = []
        self.all_nodes: Dict[int, TreeNode] = {}  # Global node lookup
        
        # Build the tree
        self._build_tree()
    
    def _build_tree(self):
        """Build the complete binary tree structure."""
        print(f"Building hierarchical tree with depth {self.max_depth}, {self.num_points} points")
        
        # Initialize all layers
        for layer_id in range(self.max_depth + 1):
            num_nodes = 2 ** (self.max_depth - layer_id)
            nodes = []
            node_id_to_index = {}
            
            for local_id in range(num_nodes):
                node_id = self._compute_global_node_id(layer_id, local_id)
                
                # Determine parent and children
                parent_id = None
                children_ids = []
                
                if layer_id < self.max_depth:  # Not root
                    parent_local_id = local_id // 2
                    parent_id = self._compute_global_node_id(layer_id + 1, parent_local_id)
                
                if layer_id > 0:  # Not leaf
                    left_child_id = self._compute_global_node_id(layer_id - 1, local_id * 2)
                    right_child_id = self._compute_global_node_id(layer_id - 1, local_id * 2 + 1)
                    children_ids = [left_child_id, right_child_id]
                
                # Create node
                node = TreeNode(
                    node_id=node_id,
                    layer_id=layer_id,
                    local_id=local_id,
                    parent_id=parent_id,
                    children_ids=children_ids
                )
                
                nodes.append(node)
                node_id_to_index[node_id] = local_id
                self.all_nodes[node_id] = node
            
            # Create parent-child mapping for this layer
            parent_child_mapping = {}
            for node in nodes:
                if node.parent_id is not None:
                    if node.parent_id not in parent_child_mapping:
                        parent_child_mapping[node.parent_id] = []
                    parent_child_mapping[node.parent_id].append(node.node_id)  # correct


            
            # Create layer info
            layer_info = LayerInfo(
                layer_id=layer_id,
                nodes=nodes,
                num_nodes=num_nodes,
                node_id_to_index=node_id_to_index,
                parent_child_mapping=parent_child_mapping
            )
            
            self.layers.append(layer_info)
        
        # Assign points to leaf nodes
        self._assign_points_to_leaves()
        
        # Compute initial features and locations
        self._compute_node_data()
        
        print(f"Tree construction complete. Created {len(self.layers)} layers.")
    
    def _compute_global_node_id(self, layer_id: int, local_id: int) -> int:
        """
        Unique global id when layer 0 has 2**max_depth nodes and layer_id increases up to root.
        offset(layer_id) = sum_{k=0}^{layer_id-1} 2**(max_depth - k)
                         = 2**(max_depth - layer_id + 1) * (2**layer_id - 1)
        """
        if layer_id == 0:
            return local_id
        return (1 << (self.max_depth - layer_id + 1)) * ((1 << layer_id) - 1) + local_id

    
    def _assign_points_to_leaves(self):
        """Assign points to leaf nodes using a simple balanced approach."""
        leaf_layer = self.layers[0]
        points_per_leaf = max(1, self.num_points // len(leaf_layer.nodes))
        
        for i, node in enumerate(leaf_layer.nodes):
            start_idx = i * points_per_leaf
            end_idx = min((i + 1) * points_per_leaf, self.num_points)
            
            if i == len(leaf_layer.nodes) - 1:  # Last node gets remaining points
                end_idx = self.num_points
            
            if start_idx < self.num_points:
                node.point_indices = torch.arange(start_idx, end_idx, device=self.points.device)
    
    def _compute_node_data(self):
        """Compute features and locations for all nodes."""
        # Process from leaves up
        for i, layer in tqdm(enumerate(self.layers), desc="Computing node data, time decreasing exponentially", total=len(self.layers)):
            for node in layer.nodes:
                if i != 0:
                    break
                if node.point_indices is not None and node.point_indices.numel() > 0:
                    # Compute location (mean of assigned points)
                    node.location = self.points[node.point_indices].mean(dim=0)
                    
                    # Compute features if available
                    if self.features is not None:

                        node.features = self.features[node.point_indices].mean(dim=0)
                else:
                    # Empty node - copy from sibling instead of using zeros
                    self._fill_empty_node_from_sibling(node, layer)
    
    def _fill_empty_node_from_sibling(self, empty_node: TreeNode, layer: LayerInfo):
        """
        Fill an empty node by copying information from its sibling node.
        First tries to find the actual sibling (sharing same parent), 
        then falls back to the nearest sibling in the same layer.
        """
        # First try to find the actual sibling (sharing same parent)
        sibling_node = self._find_actual_sibling_with_data(empty_node)
        
        # If no actual sibling has data, find the nearest sibling in the same layer
        if sibling_node is None:
            sibling_node = self._find_sibling_with_data(empty_node, layer)
        
        if sibling_node is not None:
            # Copy location and features from sibling
            empty_node.location = sibling_node.location.clone()
            if sibling_node.features is not None:
                empty_node.features = sibling_node.features.clone()
            else:
                # Fallback to zeros if sibling also has no features
                empty_node.location = torch.zeros(3, device=self.points.device, dtype=self.points.dtype)
                if self.features is not None:
                    empty_node.features = torch.zeros(self.feature_dim, device=self.features.device, dtype=self.features.dtype)
        else:
            # No sibling with data found, use zeros as fallback
            empty_node.location = torch.zeros(3, device=self.points.device, dtype=self.points.dtype)
            if self.features is not None:
                empty_node.features = torch.zeros(self.feature_dim, device=self.features.device, dtype=self.features.dtype)
    
    def _find_sibling_with_data(self, empty_node: TreeNode, layer: LayerInfo) -> Optional[TreeNode]:
        """
        Find a sibling node that has data (non-empty point_indices).
        For leaf nodes, we look for the nearest sibling in the same layer.
        For internal nodes, we look for the nearest sibling in the same layer.
        """
        # First, try to find the nearest sibling in the same layer
        best_sibling = None
        min_distance = float('inf')
        
        for node in layer.nodes:
            if (node != empty_node and 
                node.point_indices is not None and 
                node.point_indices.numel() > 0):
                
                # Calculate distance based on local_id (closer in tree structure)
                distance = abs(node.local_id - empty_node.local_id)
                if distance < min_distance:
                    min_distance = distance
                    best_sibling = node
        
        if best_sibling is not None:
            return best_sibling
        
        # If no sibling in same layer has data, look for any node with data
        for layer_info in self.layers:
            for node in layer_info.nodes:
                if (node.point_indices is not None and 
                    node.point_indices.numel() > 0):
                    return node
        
        return None
    
    def _find_actual_sibling_with_data(self, empty_node: TreeNode) -> Optional[TreeNode]:
        """
        Find the actual sibling node (sharing the same parent) that has data.
        This is more specific than _find_sibling_with_data and looks for the true sibling.
        """
        if empty_node.parent_id is None:
            # Root node has no siblings
            return None
        
        parent_node = self.all_nodes[empty_node.parent_id]
        
        # Look for siblings (other children of the same parent)
        for sibling_id in parent_node.children_ids:
            if sibling_id != empty_node.node_id:
                sibling_node = self.all_nodes[sibling_id]
                if (sibling_node.point_indices is not None and 
                    sibling_node.point_indices.numel() > 0):
                    return sibling_node
        
        # If no direct sibling has data, look for any node with data
        return self._find_sibling_with_data(empty_node, self.layers[empty_node.layer_id])
    
    # ==================== Core Access Methods ====================
    
    def get_layer_features(self, layer_id: int) -> torch.Tensor:
        """Get all features for a specific layer."""
        layer = self.layers[layer_id]
        features = []
        
        for node in layer.nodes:
            if node.features is not None:
                features.append(node.features)
            else:
                # Create zero features if not available
                zero_features = torch.zeros(self.feature_dim, device=self.points.device, dtype=self.points.dtype)
                features.append(zero_features)
        
        return torch.stack(features, dim=0)  # [num_nodes, feature_dim]
    
    def get_layer_locations(self, layer_id: int) -> torch.Tensor:
        """Get all locations for a specific layer."""
        layer = self.layers[layer_id]
        locations = []
        
        for node in layer.nodes:
            if node.location is not None:
                locations.append(node.location)
            else:
                zero_location = torch.zeros(3, device=self.points.device, dtype=self.points.dtype)
                locations.append(zero_location)
        
        return torch.stack(locations, dim=0)  # [num_nodes, 3]
    
    def get_parent_child_features(self, parent_layer_id: int) -> torch.Tensor:
        """
        Get features grouped by parent-child relationships.
        
        Args:
            parent_layer_id: Layer ID of parent nodes
            
        Returns:
            torch.Tensor: Shape [num_parents, num_children_per_parent, feature_dim]
        """
        if parent_layer_id >= len(self.layers):
            raise ValueError(f"Parent layer {parent_layer_id} has no children")
        
        parent_layer = self.layers[parent_layer_id]
        child_layer = self.layers[parent_layer_id - 1]
        
        grouped_features = []
        
        for parent_node in parent_layer.nodes:
            child_features = []
            
            for child_id in parent_node.children_ids:
                child_node = self.all_nodes[child_id]
                if child_node.features is not None:
                    child_features.append(child_node.features)
                else:
                    zero_features = torch.zeros(self.feature_dim, device=self.points.device, dtype=self.points.dtype)
                    child_features.append(zero_features)
            
            if child_features:
                parent_child_features = torch.stack(child_features, dim=0)  # [num_children, feature_dim]
                grouped_features.append(parent_child_features)
        
        if grouped_features:
            return torch.stack(grouped_features, dim=0)  # [num_parents, num_children, feature_dim]
        else:
            return torch.zeros(0, 0, self.feature_dim, device=self.points.device, dtype=self.points.dtype)
    
    def get_ancestor_descendant_features(self, source_layer_id: int, hops: int) -> torch.Tensor:
        """
        Get features grouped by ancestor-descendant relationships across multiple levels.
        
        Args:
            source_layer_id: Layer ID of descendant nodes
            hops: Number of levels to go up (1=direct parent, 2=grandparent, etc.)
            
        Returns:
            torch.Tensor: Shape [num_ancestors, 2^hops, feature_dim]
        """
        if hops <= 0:
            raise ValueError("hops must be positive")
        
        ancestor_layer_id = source_layer_id + hops
        if ancestor_layer_id >= len(self.layers):
            raise ValueError(f"Ancestor layer {ancestor_layer_id} exceeds tree depth")
        
        source_layer = self.layers[source_layer_id]
        ancestor_layer = self.layers[ancestor_layer_id]
        
        grouped_features = []
        
        for ancestor_local_id, ancestor_node in enumerate(ancestor_layer.nodes):
            # Calculate which descendants belong to this ancestor
            # Each ancestor at position i has descendants at positions [i*2^hops, (i+1)*2^hops)
            start_descendant_id = ancestor_local_id * (2 ** hops)
            end_descendant_id = start_descendant_id + (2 ** hops)
            
            descendant_features = []
            for desc_local_id in range(start_descendant_id, end_descendant_id):
                if desc_local_id < len(source_layer.nodes):
                    descendant_node = source_layer.nodes[desc_local_id]
                    if descendant_node.features is not None:
                        descendant_features.append(descendant_node.features)
                    else:
                        zero_features = torch.zeros(self.feature_dim, device=self.points.device, dtype=self.points.dtype)
                        descendant_features.append(zero_features)
                else:
                    # Handle case where we don't have enough descendants
                    zero_features = torch.zeros(self.feature_dim, device=self.points.device, dtype=self.points.dtype)
                    descendant_features.append(zero_features)
            
            if descendant_features:
                ancestor_descendant_features = torch.stack(descendant_features, dim=0)  # [2^hops, feature_dim]
                grouped_features.append(ancestor_descendant_features)
        
        if grouped_features:
            return torch.stack(grouped_features, dim=0)  # [num_ancestors, 2^hops, feature_dim]
        else:
            return torch.zeros(0, 2**hops, self.feature_dim, device=self.points.device, dtype=self.points.dtype)
    
    def update_layer_features(self, layer_id: int, new_features: torch.Tensor):
        """Update features for all nodes in a specific layer."""
        layer = self.layers[layer_id]
        
        if new_features.shape[0] != len(layer.nodes):
            raise ValueError(f"Expected {len(layer.nodes)} features, got {new_features.shape[0]}")
        
        for i, node in enumerate(layer.nodes):
            node.features = new_features[i]
    
    def reset_to_base_features(self, new_features: Optional[torch.Tensor] = None, feature_dim: Optional[int] = None):
        """
        Reset the tree to use new base features, recomputing all node features from scratch.
        This allows you to process different initial features with different networks.
        
        Args:
            new_features: New base features [N, C]. If None, uses self.features
        """
        if feature_dim is not None:
            self.feature_dim = feature_dim
        if new_features is not None:
            if new_features.shape[0] != self.num_points:
                raise ValueError(f"Expected {self.num_points} features, got {new_features.shape[0]}")
            if new_features.shape[1] != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} feature dimensions, got {new_features.shape[1]}")
            self.features = new_features
        
        # Recompute all node features from the base features
        self._compute_node_data()
        print(f"✓ Tree reset to use new base features. Shape: {self.features.shape}")
    
    def copy_with_new_features(self, new_features: torch.Tensor) -> 'HierarchicalTree':
        """
        Create a copy of this tree with new base features but same structure.
        This is useful when you want to process different features with different networks
        without modifying the original tree.
        
        Args:
            new_features: New base features [N, C]
            
        Returns:
            New HierarchicalTree instance with same structure but new features
        """
        if new_features.shape[0] != self.num_points:
            raise ValueError(f"Expected {self.num_points} features, got {new_features.shape[0]}")
        if new_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature dimensions, got {new_features.shape[1]}")
        
        # Create new tree with same structure but new features
        new_tree = HierarchicalTree.__new__(HierarchicalTree)
        new_tree.points = self.points
        new_tree.features = new_features
        new_tree.num_points = self.num_points
        new_tree.feature_dim = self.feature_dim
        new_tree.max_depth = self.max_depth
        
        # Copy the tree structure (this is the expensive part, but we're reusing it)
        new_tree.layers = self.layers
        new_tree.all_nodes = self.all_nodes
        
        # Recompute node data with new features
        new_tree._compute_node_data()
        print(f"✓ Created tree copy with new features. Shape: {new_features.shape}")
        
        return new_tree
    
    # ==================== Utility Methods ====================
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the tree structure."""
        info = {
            'max_depth': self.max_depth,
            'num_layers': len(self.layers),
            'num_points': self.num_points,
            'feature_dim': self.feature_dim,
            'layers': []
        }
        
        for layer in self.layers:
            layer_info = {
                'layer_id': layer.layer_id,
                'num_nodes': layer.num_nodes,
                'has_features': layer.nodes[0].features is not None if layer.nodes else False
            }
            info['layers'].append(layer_info)
        
        return info
    
    def print_tree_structure(self):
        """Print the tree structure for debugging."""
        print(f"\nHierarchical Tree Structure:")
        print(f"Depth: {self.max_depth}, Points: {self.num_points}")
        
        for layer in self.layers:
            print(f"Layer {layer.layer_id}: {layer.num_nodes} nodes")
            if layer.nodes and layer.nodes[0].features is not None:
                print(f"  Features: {layer.nodes[0].features.shape}")
            if layer.nodes and layer.nodes[0].location is not None:
                print(f"  Locations: {layer.nodes[0].location.shape}")

    def to(self, device: torch.device, in_place: bool = True):
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device (e.g., torch.device('cpu') or torch.device('cuda'))
            in_place: If True and moving from CUDA to CPU, release CUDA memory after transfer
        
        Returns:
            self (for chaining)
        """
        # Check if we're moving from CUDA to CPU for memory release
        was_on_cuda = self.points.is_cuda
        moving_to_cpu = device == 'cpu'
        moving_to_cuda = device == 'cuda'
        should_release_memory = in_place and was_on_cuda and moving_to_cpu
        
        # Move main tensors (use non_blocking only when moving to CUDA for async transfer)
        self.points = self.points.to(device, non_blocking=moving_to_cuda)
        if self.features is not None:
            self.features = self.features.to(device, non_blocking=moving_to_cuda)
        
        # Move node tensors
        for layer in self.layers:
            for node in layer.nodes:
                if node.point_indices is not None:
                    node.point_indices = node.point_indices.to(device, non_blocking=moving_to_cuda)
                if node.location is not None:
                    node.location = node.location.to(device, non_blocking=moving_to_cuda)
                if node.features is not None:
                    node.features = node.features.to(device, non_blocking=moving_to_cuda)
        
        # Release CUDA memory if requested
        if should_release_memory:
            # Synchronize CUDA operations to ensure all transfers are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()  # Collect Python garbage first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache
        
        return self