"""
Query System for Scene Decomposition
The query system is a pipeline that takes in a query and returns a response.
The response are the following:
Flag + Content: response with content
None: no response
List of [None | Torch.Tensor]: it could be binary mask or attention map
"""
from .querySystem import QuerySystem
from .database import Database, FeatureDatabase

__all__ = [
    "QuerySystem",
    "Database",
    "FeatureDatabase",
]