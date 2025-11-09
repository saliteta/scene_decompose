import torch
from dataclasses import dataclass
from typing import Literal, Dict

"""
    Sometime, we will return a image feature token, and some time, we will return a
    gt camera pose. This is handle by Gradio thread.

    We need to handle different query data in main thread
"""


@dataclass
class QueryData:
    query_type: Literal["image_feature", "gt_camera_pose"] = "image_feature"
    image_feature_token: torch.Tensor | None = None
    gt_camera: Dict[str, torch.Tensor] | None = None
