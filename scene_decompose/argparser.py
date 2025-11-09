# pip install tyro
from dataclasses import dataclass, field
from typing import Optional, Literal
import tyro

# ----- Backbone options -----
BACKBONE_OPTIONS = (
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_base_patch16_224",
    "ViT-B-16",
    "vit_base_patch16_224.dino",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_small_patch14_reg4_dinov2",
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_small_plus_patch16_dinov3.lvd1689m",
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_base_patch16_clip_384",
    "vit_base_patch16_siglip_512.v2_webli",
    "radio_v2.5-b",
    "radio_v2.5-l",
    "radio_v2.5-h",
)

### Detail Check JAFAR Backbone options:
# https://github.com/PaulCouairon/JAFAR/blob/main/jafar.ipynb


# ----- Your dataclasses -----
@dataclass
class DataArgs:
    dir: str
    test_every: int = 1000
    factor: int = 1


@dataclass
class DistillArgs:
    ckpt: str
    method: Literal["2DGS", "3DGS", "DBS", "H_GS"] = "3DGS"
    feature_ckpt: str | None = None


@dataclass
class ModelArgs:
    model_type: Literal["jafar", "openclip"] = "jafar"
    backbone: Literal[
        "ViT-B-16",  # OpenCLIP Backbone
        ######### JAFAR Backbone #########
        "vit_large_patch16_dinov3.lvd1689m",
        "vit_base_patch16_224",
        "vit_base_patch16_224.dino",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_small_patch14_reg4_dinov2",
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_small_plus_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_base_patch16_clip_384",
        "vit_base_patch16_siglip_512.v2_webli",
        "radio_v2.5-b",
        "radio_v2.5-l",
        "radio_v2.5-h",
    ] = "vit_large_patch16_dinov3.lvd1689m"
    model_path: Optional[str] = None


# ----- Wrapper for tyro (nested CLI groups) -----
@dataclass
class Args:
    data: DataArgs
    distill: DistillArgs
    model: ModelArgs


def parse_args() -> tuple[DataArgs, DistillArgs, ModelArgs]:
    """
    Parses CLI into nested dataclasses using tyro and returns
    (data_args, distill_args, model_args).
    """
    args = tyro.cli(Args)  # tyro builds the CLI from the dataclass schema
    return args.data, args.distill, args.model
