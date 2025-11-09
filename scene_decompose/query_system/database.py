from dataclasses import dataclass
import torch
from typing import List, Tuple, Type


@dataclass
class Database:
    name: str = "default database"
    input_description: str = "Default binary layer query"
    output_description: str = (
        "List of [None | Torch.Tensor] as a dense index mask to the database, None means no constrains\
        Binary layer means a binary layer mask"
    )
    flags: Tuple[List[str], List[Type]] = (
        ["None", "List[Tensor|None]"],
        [type(None), List[torch.Tensor | None]],
    )
    num_layers: int = 0  # number of layers in the database
    features: List[torch.Tensor | None] | None = None

    def get_readable_types(self) -> List[str]:
        """Return a list of readable type strings"""
        return self.flags[0]  # Returns ["None", "List[Tensor|None]"]

    def get_actual_types(self) -> List[Type]:
        """Return a list of actual Python types"""
        return self.flags[1]  # Returns [type(None), List[torch.Tensor | None]]

    def get_type_info(self) -> dict:
        """Return a dictionary with both readable and actual type information"""
        return {
            "readable_types": self.get_readable_types(),
            "actual_types": self.get_actual_types(),
            "type_mapping": dict(zip(self.flags[0], self.flags[1])),
        }

    def verbose(self) -> str:
        """Return a verbose string of the database"""
        return f"Database: {self.name}\nInput Description: {self.input_description}\nOutput Description: {self.output_description}\nFlags: {self.flags}\nNumber of Layers: {self.num_layers}"


@dataclass
class FeatureDatabase(Database):
    name: str = "Feature Database"
    input_description: str = "Image feature query, use query image button"
    output_description: str = (
        "it should return an attention map and a layer index for compressed response"
    )

    # verify the features are provided and not None
    def __post_init__(self):
        """Validate that features are provided and not None"""
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        # Validate features are provided
        if self.features is None:
            raise ValueError(
                "FeatureDatabase requires 'features' to be provided during initialization. Cannot be None."
            )

        # Validate features is a non-empty list
        if not isinstance(self.features, list) or len(self.features) == 0:
            raise ValueError(
                "FeatureDatabase requires 'features' to be a non-empty list of tensors."
            )

        # Validate all features are tensors (not None)
        for i, feature in enumerate(self.features):
            if feature is None:
                raise ValueError(
                    f"FeatureDatabase requires all features to be tensors. Feature at index {i} is None."
                )
            if not isinstance(feature, torch.Tensor):
                raise ValueError(
                    f"FeatureDatabase requires all features to be torch.Tensor. Feature at index {i} is {type(feature)}."
                )

        # Update num_layers based on actual features
        self.num_layers = len(self.features)

        print(f"✓ FeatureDatabase initialized with {self.num_layers} feature layers")
