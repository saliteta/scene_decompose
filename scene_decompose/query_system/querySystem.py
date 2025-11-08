from .database import Database, FeatureDatabase
from typing import Union, List
import torch
from abc import abstractmethod
try:
    # Try absolute import first (when run as module)
    from scene_decompose.hierachical_gs.splatLayer.splat_image_attention import SplatImageAttention
except ImportError:
    # Fall back to relative import (when imported as a module)
    from ..hierachical_gs.splatLayer.splat_image_attention import SplatImageAttention

class QuerySystem:
    def __init__(self,database: Database):
        self.database = database
        self.num_layers = database.num_layers

    @property
    def input_description(self):
        return self.database.input_description

    @property
    def output_description(self):
        return self.database.output_description

    @property
    def avaliable_flags(self):
        """
            for verification in the upper system
        """
        return self.database.get_actual_types()

    
    def verbose(self) -> str:
        return f"Query System: {self.database.name}\n" \
            f"Input Description: {self.input_description}\n" \
            f"Output Description: {self.output_description}\n" \
            f"Availiable Flags: {self.database.get_readable_types()}\n"

    @abstractmethod
    def query(self, query: Union[str, torch.Tensor, None]) -> Union[List[torch.Tensor], None]:
        """
        Query the system with a single text. Torch.Tensor means query with image feature.
        Maybe other types of query will be supported in the future.
        Args:
            query: Union[str, torch.Tensor]
        Returns:
            determined by the flags of the query system
        """
        pass


class LayerQuerySystem(QuerySystem):
    def __init__(self, database: Database):
        super().__init__(database)

    def query(self, query_text: Union[str, None] = None)-> Union[List[torch.Tensor], None]:
        """
            Query the system with a single text.
            The example of text is: "00" or "001"
            00 means query from top to bottom, layer 2, node 0 and all its decendents
            001 means query from top to bottom, layer 3, node 1 and all its decendents
            We need to return a list of torch tensors determine the index range of the nodes in each layer
            The length of the list is the number of layers.
            Higher layer has no index, therefore return None
            Lower layer has index, therefore return the index range of the nodes in the layer

            Args:
                query_text: the text to query the system
            Returns:
                A list of torch tensors determine the index range of the nodes in each layer
        """
        if query_text is not None:
            assert isinstance(query_text, str), f"Query text must be a string or None, but got {type(query_text)}"
            assert all(c in "01" for c in query_text), f"Query text must only contain '0' and '1', but got {query_text}"
            assert len(query_text) < self.num_layers, f"Query text length must be less than the number of layers, but got {len(query_text)}"
        else:
            return None
        highest_layer_number = len(query_text)
        layer_index = int(query_text, 2)
        index_range = []
        for layer in range(highest_layer_number):
            index_range.append(None)
        for layer in range(highest_layer_number, self.num_layers):
            index_range.append(torch.tensor([layer_index << (layer - highest_layer_number), ((layer_index+1) << (layer - highest_layer_number)) - 1]))
        return index_range



class FeatureQuerySystem(LayerQuerySystem):
    def __init__(self, database: FeatureDatabase):
        super().__init__(database)
        self.splat_image_attention = SplatImageAttention()

    
    def query(self, query_image: Union[torch.Tensor, str, None] = None, layer_level: int = 5, token_level: int = 5)-> Union[List[torch.Tensor], None]:
        """
        if query image is None or str is given, we run layer query system
        Query the system with a single image.
        We will get the splat's token in the shape of 
        2**layer_level * 2**token_level * C
        Args:
            query_image: the image to query the system [K, C]
            layer_level: the level of the layer to query the system
            token_level: the level of the token to query the system
        Returns:
            A list of torch tensors determine the attention response of the features in layer_level 
            in the shape of 2**layer_level

        We are supposed to have 2**token_level == K
        """
        if query_image is None or isinstance(query_image, str):
            print(f"\033[92m[FeatureQuerySystem] Query image is None or str, running layer query system\033[0m")
            return super().query(query_image)
        K,C = query_image.shape
        assert layer_level + token_level <= self.num_layers, f"Layer level and token level must be less than the number of layers, but got {layer_level} + {token_level} > {self.num_layers}"
        assert K == 2**token_level, f"Query image must have K tokens, but got {K}, Query level is {token_level}"
        assert C == self.database.features[layer_level].shape[1],\
             f"Query image must have C channels, but got {C}, \
                Feature dim is {self.database.features[layer_level].shape[1]} for layer {layer_level}"
        
        splat_block_features:torch.Tensor = self._prepare_splat_block_features(layer_level, token_level)
        attn_scores:torch.Tensor = self.splat_image_attention(splat_block_features, query_image)
        assert attn_scores.shape == (2**layer_level,), f"Attention scores must have 2**layer_level elements, but got {attn_scores.shape}"
        return attn_scores
    

    def _prepare_splat_block_features(self, layer_level: int, token_level: int) -> torch.Tensor:
        """
        Prepare the splat block features in the shape of 2**layer_level * 2**token_level * C
        Args:
            layer_level: the level of the layer to prepare the features
            token_level: the level of the token to prepare the features
        Returns:
            The splat block features in the shape of 2**layer_level * 2**token_level * C
        """

        block_number = 2 ** layer_level
        token_number = 2 ** token_level
        flat_features:torch.Tensor = self.database.features[-(layer_level+token_level+1)] # [N, C]
        block_features:torch.Tensor = flat_features.view(block_number, token_number, -1) # [2**layer_level, 2**token_level, C]
        return block_features.to("cuda")