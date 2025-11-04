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

from pathlib import Path
import viser
import time
import tyro
from dataclasses import dataclass
from typing import Annotated, Literal
from scene_decompose import HierachicalViewer, FeatureQuerySystem, HierachicalViewerState, FeatureDatabase
from scene_decompose import GeneralViewerWrapper, GeneralViewerWrapperState


@dataclass
class Args:
    primitive_type: Annotated[Literal["h_gs", "gs"], tyro.conf.arg(aliases=["-st"])]
    primitive_path: Annotated[str, tyro.conf.arg(aliases=["-s"])]
    splat_method: Annotated[Literal["3DGS", "2DGS", "DBS"], tyro.conf.arg(aliases=["-sm"])] = "3DGS"
    feature_path: Annotated[str, tyro.conf.arg(aliases=["-f"])] = None
    port: Annotated[int, tyro.conf.arg(aliases=["-P"])] = 8080

def main(args: Args):
    primitive_path = Path(args.primitive_path)

    server = viser.ViserServer(port=args.port, verbose=False)

    if args.primitive_type == "h_gs":
        viewer = HierachicalViewer(
            server=server,
            hierachical_primitive_path=primitive_path,
            viewer_state=HierachicalViewerState(render_mode="RGB"),
            database=FeatureDatabase,
            query_system_type=FeatureQuerySystem,
            with_feature=True
        )
    elif args.primitive_type == "gs":
        with_feature = args.feature_path is not None
        viewer = GeneralViewerWrapper(
            server=server,
            primitive_path=primitive_path,
            render_tab_state=GeneralViewerWrapperState(render_mode="RGB"),
            splat_method=args.splat_method,
            with_feature=with_feature,
            feature_path=args.feature_path
        )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)