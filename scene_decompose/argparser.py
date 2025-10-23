# pip install tyro
from dataclasses import dataclass, field
from typing import Optional
import tyro

# ----- Your dataclasses -----
@dataclass
class DataArgs:
    dir: str
    test_every: int = 1000
    factor: int = 1

@dataclass
class DistillArgs:
    ckpt: str
    method: str = '3DGS'

@dataclass
class ModelArgs:
    model_path: str
    backbone: str = "vit_large_patch16_dinov3.lvd1689m"

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
