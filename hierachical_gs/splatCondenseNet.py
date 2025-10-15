"""
SplatCondenseNet implementation for the new HierarchicalTree structure.
This implementation is clean, efficient, and leverages the new tree's capabilities.
"""

import torch
import torch.nn as nn
from tree.tree import HierarchicalTree
from splatLayer.splat_self_attention import SplatMeanPooling, SplatAttentionPooling
from tqdm import trange


class SplatCondenseNet(nn.Module):
    """
    Efficient hierarchical feature condensation using the new tree structure.
    
    Strategy:
    - Layers 1-2: Use mean pooling (2 children per parent, attention not useful)
    - Layer 3+: Use attention pooling (8+ children per parent, attention becomes useful)
    """
    
    def __init__(self, tree: HierarchicalTree, attention_start_layer: int = 3, hops_for_attention: int = 3):
        super(SplatCondenseNet, self).__init__()
        
        self.tree = tree
        self.attention_start_layer = attention_start_layer
        self.hops_for_attention = hops_for_attention
        
        # Initialize pooling modules
        self.mean_pooling = SplatMeanPooling()
        self.attention_pooling = SplatAttentionPooling()
        
        print(f"SplatCondenseNetNew initialized:")
        print(f"  - Attention starts at layer {attention_start_layer}")
        print(f"  - Using {hops_for_attention} hops for attention (2^{hops_for_attention} = {2**hops_for_attention} children per parent)")
    
    def forward(self, tree: HierarchicalTree = None):
        """
        Process features hierarchically using mean pooling for early layers and attention for later layers.
        
        Args:
            tree: Optional tree to process (uses self.tree if None)
            
        Returns:
            The processed tree with updated features
        """
        if tree is None:
            tree = self.tree
        
        tree_info = tree.get_tree_info()
        num_layers = tree_info['num_layers']
        
        print(f"\nProcessing tree with {num_layers} layers...")
        
        # Process each layer bottom-up (skip layer 0 - leaves)
        for current_layer_id in trange(1, num_layers, desc="Processing layers"):
            print(f"\nProcessing layer {current_layer_id}...")
            
            if current_layer_id < self.attention_start_layer:
                # Use mean pooling for early layers (direct parent-child relationship)
                self._process_with_mean_pooling(tree, current_layer_id)
            else:
                # Use attention pooling for later layers (multiple hops)
                self._process_with_attention_pooling(tree, current_layer_id)
        
        print(f"\n✓ Hierarchical processing completed!")
        return tree
    
    def _process_with_mean_pooling(self, tree: HierarchicalTree, parent_layer_id: int):
        """
        Process features using mean pooling (direct parent-child relationship).
        """
        # Get parent-child features: [num_parents, 2, feature_dim]
        parent_child_features = tree.get_parent_child_features(parent_layer_id)
        
        print(f"  Layer {parent_layer_id}: Mean pooling with {parent_child_features.shape}")
        print(f"    - {parent_child_features.shape[0]} parents")
        print(f"    - {parent_child_features.shape[1]} children per parent")
        
        # Apply mean pooling to each parent's children
        processed_features = []
        for parent_idx in range(parent_child_features.shape[0]):
            children_features = parent_child_features[parent_idx]  # [2, feature_dim]
            pooled_feature = self.mean_pooling(children_features)  # [feature_dim]
            processed_features.append(pooled_feature)
        
        # Stack and update parent features
        new_parent_features = torch.stack(processed_features, dim=0)  # [num_parents, feature_dim]
        tree.update_layer_features(parent_layer_id, new_parent_features)
        
        print(f"    ✓ Updated {new_parent_features.shape[0]} parent features")
    
    def _process_with_attention_pooling(self, tree: HierarchicalTree, parent_layer_id: int):
        """
        Process features using attention pooling (multiple hops to get more children per parent).
        """
        # Calculate source layer (where we collect features from)
        source_layer_id = max(0, parent_layer_id - self.hops_for_attention)
        
        # Get ancestor-descendant features: [num_ancestors, 2^hops, feature_dim]
        ancestor_descendant_features = tree.get_ancestor_descendant_features(source_layer_id, self.hops_for_attention)
        
        print(f"  Layer {parent_layer_id}: Attention pooling with {ancestor_descendant_features.shape}")
        print(f"    - Source layer: {source_layer_id}")
        print(f"    - Hops: {self.hops_for_attention}")
        print(f"    - {ancestor_descendant_features.shape[0]} ancestors")
        print(f"    - {ancestor_descendant_features.shape[1]} descendants per ancestor")
        
        # Apply attention pooling to each ancestor's descendants
        processed_features = []
        for ancestor_idx in range(ancestor_descendant_features.shape[0]):
            descendants_features = ancestor_descendant_features[ancestor_idx]  # [2^hops, feature_dim]
            pooled_feature = self.attention_pooling(descendants_features)  # [feature_dim]
            processed_features.append(pooled_feature)
        
        # Stack and update ancestor features
        new_ancestor_features = torch.stack(processed_features, dim=0)  # [num_ancestors, feature_dim]
        
        # Map back to the correct layer (this might need adjustment based on actual tree structure)
        target_layer_id = source_layer_id + self.hops_for_attention
        if target_layer_id == parent_layer_id:
            tree.update_layer_features(parent_layer_id, new_ancestor_features)
            print(f"    ✓ Updated {new_ancestor_features.shape[0]} ancestor features")
        else:
            print(f"    ⚠ Warning: Layer mismatch - expected {parent_layer_id}, got {target_layer_id}")
    
    def get_processing_summary(self) -> dict:
        """Get a summary of the processing strategy."""
        tree_info = self.tree.get_tree_info()
        num_layers = tree_info['num_layers']
        
        summary = {
            'total_layers': num_layers,
            'attention_start_layer': self.attention_start_layer,
            'hops_for_attention': self.hops_for_attention,
            'children_per_parent_attention': 2 ** self.hops_for_attention,
            'processing_strategy': {}
        }
        
        for layer_id in range(1, num_layers):
            if layer_id < self.attention_start_layer:
                summary['processing_strategy'][layer_id] = {
                    'method': 'mean_pooling',
                    'children_per_parent': 2,
                    'description': 'Direct parent-child relationship'
                }
            else:
                summary['processing_strategy'][layer_id] = {
                    'method': 'attention_pooling',
                    'children_per_parent': 2 ** self.hops_for_attention,
                    'description': f'Multi-level ancestor-descendant relationship ({self.hops_for_attention} hops)'
                }
        
        return summary


def create_example_condense_net():
    """Create an example SplatCondenseNet for testing."""
    from tree.tree import create_example_tree
    
    # Create tree
    tree = create_example_tree()
    
    # Create condense net
    condense_net = SplatCondenseNet(tree, attention_start_layer=3, hops_for_attention=3)
    
    # Print processing summary
    summary = condense_net.get_processing_summary()
    print("\nProcessing Summary:")
    for layer_id, strategy in summary['processing_strategy'].items():
        print(f"  Layer {layer_id}: {strategy['method']} - {strategy['children_per_parent']} children/parent")
    
    return tree, condense_net


if __name__ == "__main__":
    # Test the new implementation
    tree, condense_net = create_example_condense_net()
    
    print("\n" + "="*60)
    print("Testing SplatCondenseNetNew...")
    print("="*60)
    
    # Process the tree
    processed_tree = condense_net()
    
    # Print final results
    print("\nFinal layer features:")
    tree_info = processed_tree.get_tree_info()
    for layer_info in tree_info['layers']:
        layer_id = layer_info['layer_id']
        if layer_info['has_features']:
            features = processed_tree.get_layer_features(layer_id)
            print(f"Layer {layer_id}: {features.shape}")
        else:
            print(f"Layer {layer_id}: No features")
