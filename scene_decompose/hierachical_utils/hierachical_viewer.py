from dataclasses import dataclass
import viser
import copy
import torch
from pathlib import Path
from nerfview import CameraState, RenderTabState
from typing import List, Union, Literal, Tuple, Type
from nerfview import Viewer, RenderTabState
from gsplat_ext import ViewerState
import torch
from tqdm import tqdm
from .image_dashboard import launch_image_dashboard
from .hierachical_primitive import HierachicalPrimitive
from .pca_utils import pca_torch_extract_directions, apply_pca_directions_torch
from ..query_system.database import Database, FeatureDatabase
from ..query_system.querySystem import QuerySystem, LayerQuerySystem, FeatureQuerySystem

@dataclass
class HierachicalViewerState(RenderTabState):
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Feature"] = "RGB"
    layer_index: int = 0



class HierachicalViewer(Viewer):
    def __init__(self, server: viser.ViserServer, 
                       hierachical_primitive_path: Path, 
                       viewer_state: HierachicalViewerState, 
                       database: Type[Database],
                       query_system_type: Type[QuerySystem],
                       port=8080, with_feature: bool = False
                       ):
                       
        self.port = port
        self._dashboard_started = False
        self._dashboard_url = None
        self._dashboard_thread = None
        self.with_feature = with_feature
        self.viewer_state = viewer_state
        self.render_tab_state = copy.deepcopy(viewer_state)
        self.hierachical_primitive_path = hierachical_primitive_path
        self.database_type = database
        self.query_system_type = query_system_type
        
        # Print initialization header
        print("\n" + "="*60)
        print("\033[94m🚀 Initializing HierachicalViewer\033[0m")
        print("="*60)
        
        # Initialize source primitive on CPU
        self.hierachical_primitive_initialization()
        # Setup query system
        self.setup_query_system()
        
        print("\033[92m✓ HierachicalViewer initialization completed!\033[0m")
        print("="*60 + "\n")
        # Caching system
        self._hierachical_primitive_gpu_cache = None
        self._current_query_index, self._system_query_index = None, None  # Internal state for query index
        self.curret_renderer_cache = self.hierachical_primitive.get_renderer(0)
        super().__init__(server, self.render_function, None, 'rendering')
    
    def _on_dashboard_submit(self, image_feature_token: torch.Tensor):
        """
        This is the hook of the dashboard thread holding by JAFAR
        as well as the Gradio thread.

        It is supposed to process the 2D image, returning the feature token of the image.
        As well as display some information of the image in the gradio dashboard.

        And when it return the result, we are supposed to run the image feature query in this method
        I think it would be better to run in another thread.
        """
        # image_feature_token is (F,)
        pass
        # TODO: Implement the image feature query
        # TODO: Implement the run of the image feature query in another thread
        # TODO: Implement the return of the image feature query result

    def hierachical_primitive_initialization(self):
        "set up splat entity"
        print("\033[93m📦 Loading hierarchical primitive...\033[0m")
        self._hierachical_primitive = HierachicalPrimitive(with_feature=self.with_feature)
        self._hierachical_primitive.load_from_file(self.hierachical_primitive_path)
        print("\033[93m🔧 Applying feature PCA...\033[0m")
        self._feature_pca()
        self._hierachical_primitive.to("cpu")
        print("\033[92m✓ Hierarchical primitive loaded successfully!\033[0m")

    def setup_query_system(self):
        """Initialize the query system with the provided database and query system type"""
        if self.with_feature and self.database_type == FeatureDatabase:
            self.database: FeatureDatabase = self.database_type(num_layers=len(self._hierachical_primitive.source["feature"]), features=self._hierachical_primitive.source["feature"]) # initialization of the database
        else:
            self.database:Database = self.database_type(num_layers=len(self._hierachical_primitive.source["geometry"])) # initialization of the database
        self.query_system = self.query_system_type(self.database)
        print("\033[92m✓ Successfully initialized database!\033[0m")
        print(f"\033[96m{self.query_system.verbose()}\033[0m")


    def _update_hierachical_primitive_gpu_cache(self):
        self._current_query_index = self._system_query_index
        if self._system_query_index == None:
            return self._hierachical_primitive.copy_to("cuda")
        else:
            query_list = self.query_system.query(self._system_query_index)
            assert query_list is not None, "Query list is None, should be handled above"
            return self._hierachical_primitive.query_with_list_torch_index(query_list).copy_to("cuda")


    @property
    def hierachical_primitive(self):
        if self._hierachical_primitive_gpu_cache == None or (self._current_query_index != self._system_query_index):
            # inconsistent query index, update the cache
            # or no cache, update the cache
            self._hierachical_primitive_gpu_cache = self._update_hierachical_primitive_gpu_cache()
        return self._hierachical_primitive_gpu_cache


    def reset(self):
        self.render_tab_state = copy.deepcopy(self.viewer_state)
        self.hierachical_primitive_initialization()
        self.setup_query_system()

    def _feature_pca(self):
        if self._hierachical_primitive.with_feature is False:
            return
        else:
            self.feature_pcas: List[torch.Tensor] = []
            for i, feature in enumerate(tqdm(self._hierachical_primitive.source["feature"], desc="Feature PCA")):
                if feature.shape[0] <= 3:
                    print("Feature is too small, skipping")
                    zeros = torch.zeros(feature.shape[0], 3, dtype=feature.dtype, device=feature.device)
                    self.feature_pcas.append(zeros)
                    continue
                if i == 0:
                    projected_features, pc_directions, mean = pca_torch_extract_directions(feature, n_components=3)
                    self.pc_directions = pc_directions
                    self.mean = mean
                else:
                    projected_features = apply_pca_directions_torch(feature, self.pc_directions, self.mean)

                mins   = projected_features.min(dim=0).values    # shape (3,)
                maxs   = projected_features.max(dim=0).values    # shape (3,)
                ranges = maxs - mins
                eps    = 1e-8
                projected_features = (projected_features - mins) / (ranges + eps)
                self.feature_pcas.append(projected_features)

            self._hierachical_primitive.source["feature"] = self.feature_pcas



    def camera_state_parser(self, camera_state: CameraState, render_tab_state: RenderTabState)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = camera_state.c2w
        width = render_tab_state.render_width
        height = render_tab_state.render_height
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to('cuda').unsqueeze(0)
        K = torch.from_numpy(K).float().to('cuda').unsqueeze(0)
        return K, c2w, width, height

    @torch.no_grad()
    def render_function(self, camera_state: CameraState, render_tab_state: ViewerState):
        state = self.render_tab_state
        K, c2w, width, height = self.camera_state_parser(camera_state, self.render_tab_state)
        if state.render_mode == "RGB":
            image = self.curret_renderer_cache.render(K, c2w, width, height, state.render_mode)
        elif state.render_mode == "Feature":
            image = self.curret_renderer_cache.render(K, c2w, width, height, state.render_mode)
        else:
            raise ValueError(f"Unsupported render mode: {state.render_mode}")
        return image.cpu().numpy()


    def _init_rendering_tab(self):
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")
    

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:

            self.render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                (
                    "RGB",
                    "Feature",
                ),
                initial_value=self.render_tab_state.render_mode,
                hint="Render mode to use.",
            )
            self.layer_index_slider = server.gui.add_slider(
                "Layer Index",
                initial_value=self.render_tab_state.layer_index,
                min=0,
                max=len(self.hierachical_primitive.source["geometry"]) - 1,
                step=1,
                hint="Layer index to render.",
            )


            # NEW: Image Query button
            self.image_query_btn = server.gui.add_button("Image Query")
            @self.image_query_btn.on_click
            def _(_evt):
                # Start dashboard once
                if not self._dashboard_started:
                    self._dashboard_thread, self._dashboard_url = launch_image_dashboard(
                        on_submit=self._on_dashboard_submit,
                        pca_directions=self.pc_directions,
                        pca_mean=self.mean,
                        port=7860,
                        server_name="0.0.0.0",
                        open_browser=False,
                    )
                    self._dashboard_started = True

                # Provide a clickable link inside Viser
                server.add_gui_markdown(f"[Open 2D Dashboard]({self._dashboard_url})")

            # NEW: Text Input and Submit Button
            self.text_input = server.gui.add_text(
                "Node Query System",
                initial_value="",
                hint="Enter node query such as 001 here and click submit",
            )
            
            self.text_submit_btn = server.gui.add_button("Query Subnode")
            @self.text_submit_btn.on_click
            def _(_evt):
                text_value = self.text_input.value
                print(f"[Text Input] Submitted node query: '{text_value}'")
                self._system_query_index = text_value
                del self.curret_renderer_cache
                torch.cuda.empty_cache()
                self.curret_renderer_cache = self.hierachical_primitive.get_renderer(self.render_tab_state.layer_index)
                self.rerender(_)

            @self.layer_index_slider.on_update
            def _(_) -> None:
                self.render_tab_state.layer_index = self.layer_index_slider.value
                del self.curret_renderer_cache
                torch.cuda.empty_cache()
                self.curret_renderer_cache = self.hierachical_primitive.get_renderer(self.render_tab_state.layer_index)
                self.rerender(_)

            @self.render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = self.render_mode_dropdown.value
                self.rerender(_)



        self._rendering_tab_handles.update(
            {
                "render_mode_dropdown": self.render_mode_dropdown,
                "layer_index_slider": self.layer_index_slider,
                "backgrounds_slider": self.render_tab_state.backgrounds,
                "image_query_btn": self.image_query_btn,
            }
        )

        super()._populate_rendering_tab()
                


