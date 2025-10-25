
from gsplat_ext import GaussianPrimitive, GaussianRenderer
import torch
from typing import Union, Dict, List

class HierachicalPrimitive:
    """
    A hierachical primitive is a primitive that is composed of multiple primitives.
    The primitives are organized in a tree structure, where the root is the parent primitive and the leaves are the child primitives.
    During Visualization, it is hard to display tree structure,
    We displayed our layered structure. 

    Notice that we cannot save the source data of the primitives, so it is hard to train the model.
    But it is OK to visualize the model. 

    """

    def __init__(self, with_feature: bool = False):
        self.with_feature = with_feature
        self.source: Dict[str, List[torch.Tensor]] = {} # Dict[str, List[torch.Tensor]]
        self.source["geometry"]: List[torch.Tensor] = [] # List[torch.Tensor] (mean(3) | quat(4) | scale(3) | opacity(1) | Color features(F))
        self.source["color"]: List[torch.Tensor] = [] # List[torch.Tensor]
        if with_feature:
            self.source["feature"]: List[torch.Tensor] = [] # List[torch.Tensor]

    def add_primitive(self, splat_content: Union[torch.Tensor, None] = None, feature: Union[torch.Tensor, None] = None):
        """
            Args:
                splat_content: torch.Tensor, the content of the splat
                We follow hierachcal structure to add the splat content.
                [ mean(3) | quat(4) | scale(3) | opacity(1) | Color features(F) ]
                feature: Union[torch.Tensor, None], the feature of the splat
            Returns:
                None
        """
        if splat_content is not None:
            splat_content = splat_content.cpu()
            self.source["geometry"].append(splat_content[:, :11])
            self.source["color"].append(splat_content[:, 11:])
        if feature is not None:
            assert self.with_feature, "Feature is not available"
            feature = feature.cpu()
            self.source["feature"].append(feature)
        if splat_content is None and feature is None:
            raise ValueError("Splat content and feature are not available")

    def remove_primitive(self, index: int):
        """
            Args:
                index: int, the index of the primitive to remove
            Returns:
                None
        """
        raise NotImplementedError("Not implemented, not recommended to use this function")

    
    def get_renderer(self, layer_index: int) -> GaussianRenderer:
        """
            Args:
                layer_index: int, the index means which layer of the hierachical primitive to render
            Returns:
                GaussianRenderer, the renderer of the primitive
        """
        # Notice that we are creating a Gaussian Primitive with our source data
        primitive = GaussianPrimitive()
        primitive._geometry["means"] = self.source["geometry"][layer_index][:, :3].cuda()
        primitive._geometry["quats"] = self.source["geometry"][layer_index][:, 3:7].cuda()
        primitive._geometry["scales"] = self.source["geometry"][layer_index][:, 7:10].cuda()
        primitive._geometry["opacities"] = self.source["geometry"][layer_index][:, 10].cuda()
        primitive._color["colors"] = self.source["color"][layer_index].cuda().reshape(primitive.geometry["means"].shape[0], -1, 3)
        if self.with_feature:
            primitive._feature = self.source["feature"][layer_index].cuda()
        return GaussianRenderer(primitive)


    def save_to_file(self, file_path: str):
        """
            Args:
                file_path: str, the path to save the hierachical primitive
            Returns:
                None
        """
        torch.save(self.source, file_path)

    def load_from_file(self, file_path: str):
        """
            Args:
                file_path: str, the path to load the hierachical primitive
            Returns:
                None
        """
        self.source = torch.load(file_path)

    def to(self, device: torch.device):
        """
            Args:
                device: torch.device, the device to move the hierachical primitive to
            Returns:
                None
        """
        self.source["geometry"] = [tensor.to(device) for tensor in self.source["geometry"]]
        self.source["color"] = [tensor.to(device) for tensor in self.source["color"]]
        if self.with_feature:
            self.source["feature"] = [tensor.to(device) for tensor in self.source["feature"]]
        return self
    
    def copy_to(self, device: torch.device)-> "HierachicalPrimitive":
        """
            Args:
                device: torch.device, the device to copy the hierachical primitive to
            Returns:
                HierachicalPrimitive, the copied hierachical primitive
        """
        new_hierachical_primitive = HierachicalPrimitive(with_feature=self.with_feature)
        new_hierachical_primitive.source["geometry"] = [tensor.clone().to(device) if tensor is not None else None for tensor in self.source["geometry"]]
        new_hierachical_primitive.source["color"] = [tensor.clone().to(device) if tensor is not None else None for tensor in self.source["color"]]
        if self.with_feature:
            new_hierachical_primitive.source["feature"] = [tensor.clone().to(device) if tensor is not None else None for tensor in self.source["feature"]]
        return new_hierachical_primitive


    def query_with_list_torch_index(self, index: List[torch.Tensor], invert_index: bool = True)-> "HierachicalPrimitive":
        """
            Args:
                index: torch.Tensor, the index of the primitive to query
            Returns:
                HierachicalPrimitive, the queried hierachical primitive
        """
        # If invert_index is True, reverse the index list so that the indices are processed in reverse order (from back to front).
        if invert_index:
            index = list(reversed(index))
        new_hierachical_primitive = HierachicalPrimitive(with_feature=self.with_feature)
        for i in range(len(index)):
            if index[i] is None:
                new_hierachical_primitive.source["geometry"].append(None)
                new_hierachical_primitive.source["color"].append(None)
                if self.with_feature:
                    new_hierachical_primitive.source["feature"].append(None)
            else:
                new_hierachical_primitive.source["geometry"].append(self.source["geometry"][i][index[i][0]: index[i][1]])
                new_hierachical_primitive.source["color"].append(self.source["color"][i][index[i][0]: index[i][1]])
                if self.with_feature:
                    new_hierachical_primitive.source["feature"].append(self.source["feature"][i][index[i][0]: index[i][1]])
        return new_hierachical_primitive