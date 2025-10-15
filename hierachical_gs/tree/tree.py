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
        for layer in tqdm(self.layers, desc="Computing node data, time decreasing exponentially"):
            for node in layer.nodes:
                if node.point_indices is not None and node.point_indices.numel() > 0:
                    # Compute location (mean of assigned points)
                    node.location = self.points[node.point_indices].mean(dim=0)
                    
                    # Compute features if available
                    if self.features is not None:
                        node_features = self.features[node.point_indices]
                        node.features = node_features.mean(dim=0)  # Simple mean pooling
                else:
                    # Empty node
                    node.location = torch.zeros(3, device=self.points.device, dtype=self.points.dtype)
                    if self.features is not None:
                        node.features = torch.zeros(self.feature_dim, device=self.features.device, dtype=self.features.dtype)
    
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
        if parent_layer_id >= len(self.layers) - 1:
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

