'''
This is a general viewer for the splat primitives, it can be used to visualize the splat primitives in the scene.

It should be able to handle the following cases:
1. 2DGS
2. 3DGS
3. DBS


it can be easily extended to handle other splat primitives
it focus on the feature visualization, as well as the attention map visualization
it supports the segmentation using open vocabulary query
'''

from gsplat_ext import HierachicalViewer, HierachicalViewerState
from pathlib import Path
import torch
import viser
import time
import tyro
from dataclasses import dataclass
from typing import Annotated


@dataclass
class Args:
    hierachical_primitive_path: Annotated[str, tyro.conf.arg(aliases=["-s"])]
    port: Annotated[int, tyro.conf.arg(aliases=["-P"])] = 8080

def main(args):
    hierachical_primitive_path = Path(args.hierachical_primitive_path)

    server = viser.ViserServer(port=args.port, verbose=False)

    viewer = HierachicalViewer(
        server=server,
        hierachical_primitive_path=hierachical_primitive_path,
        viewer_state=HierachicalViewerState(render_mode="RGB"),
        with_feature=True

    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)