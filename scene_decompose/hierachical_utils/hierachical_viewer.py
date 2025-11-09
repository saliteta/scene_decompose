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
from gsplat_ext import GaussianPrimitive, GaussianRenderer
from nerfview import apply_float_colormap


@dataclass
class HierachicalViewerState(RenderTabState):
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    layer_index: int = 0
    #### Input value
    render_mode: Literal["RGB", "Feature", "Attention"] = "RGB"
    query_index: str = ""
    #### Internal value for cache update
    current_render_mode: Literal["RGB", "Feature", "Attention"] = "RGB"
    current_query_index: str = ""


class HierachcalPrimitiveCPUCache(HierachicalPrimitive):
    def __init__(self, with_feature: bool = True):
        super().__init__(with_feature=with_feature)
        self.source["attention_scores"]: List[
            torch.Tensor
        ] = []  # List[torch.Tensor] (attention scores)

    def get_renderer(
        self, layer_index: int, attention: bool = False
    ) -> GaussianRenderer:
        primitive = GaussianPrimitive()
        primitive._geometry["means"] = self.source["geometry"][layer_index][
            :, :3
        ].cuda()
        primitive._geometry["quats"] = self.source["geometry"][layer_index][
            :, 3:7
        ].cuda()
        primitive._geometry["scales"] = self.source["geometry"][layer_index][
            :, 7:10
        ].cuda()
        primitive._geometry["opacities"] = self.source["geometry"][layer_index][
            :, 10
        ].cuda()
        primitive._color["colors"] = (
            self.source["color"][layer_index]
            .cuda()
            .reshape(primitive.geometry["means"].shape[0], -1, 3)
        )
        if attention:
            primitive._feature = self.source["attention_scores"][layer_index].cuda()
        else:
            primitive._feature = self.source["feature"][layer_index].cuda()
        return GaussianRenderer(primitive)

    def to(self, device: torch.device):
        super().to(device)
        if len(self.source["attention_scores"]) == 0:
            return self
        self.source["attention_scores"] = [
            tensor.to(device) for tensor in self.source["attention_scores"]
        ]
        return self

    def setup_mode(
        self, mode: Literal["RGB", "Feature", "Attention"]
    ) -> HierachicalPrimitive:
        if mode == "Feature" or mode == "RGB":
            return self
        elif mode == "Attention":
            # Create a lightweight view that reuses existing CPU tensors without cloning
            # Geometry and color are shared; feature is aliased to attention_scores
            new_hierachical_primitive = HierachicalPrimitive(with_feature=True)
            new_hierachical_primitive.source["geometry"] = self.source["geometry"]
            new_hierachical_primitive.source["color"] = self.source["color"]
            new_hierachical_primitive.source["feature"] = self.source[
                "attention_scores"
            ]
            return new_hierachical_primitive
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def set_attention_colors(self, attention_scores: torch.Tensor) -> None:
        """
        Convert attention scores to colors
        The colors will stored at self.source[""]
        Args:
            attention_scores: torch.Tensor (N,)
        Returns:
            None
        """

        # Normalize attention scores to [0, 1]
        if attention_scores.dim() > 1:
            attention_scores = attention_scores.squeeze()
        mins = attention_scores.min()
        maxs = attention_scores.max()
        eps = (
            torch.finfo(attention_scores.dtype).eps
            if attention_scores.is_floating_point()
            else 1e-8
        )
        denom = (maxs - mins) + eps
        attention_scores = (attention_scores - mins) / denom
        attention_scores = attention_scores.clamp(0.0, 1.0)

        propagated_feature = self.propagate_feature(attention_scores.unsqueeze(-1))
        propagated_feature = reversed(propagated_feature)

        for i, feature in enumerate(propagated_feature):
            colors = apply_float_colormap(feature.unsqueeze(-1), "turbo")
            colors = colors.squeeze(1)
            if len(self.source["attention_scores"]) == i:
                self.source["attention_scores"].append(colors)
            else:
                self.source["attention_scores"][i] = colors


class HierachicalViewer(Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        hierachical_primitive_path: Path,
        viewer_state: HierachicalViewerState,
        database: Type[Database],
        query_system_type: Type[QuerySystem],
        port=8080,
        with_feature: bool = False,
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
        print("\n" + "=" * 60)
        print("\033[94m🚀 Initializing HierachicalViewer\033[0m")
        print("=" * 60)

        # Initialize source primitive on CPU
        self.hierachical_primitive_initialization()

        print("\033[92m✓ HierachicalViewer initialization completed!\033[0m")
        print("=" * 60 + "\n")

        # Caching system
        self._hierachical_primitive_gpu_cache = None
        self.curret_renderer_cache = self.hierachical_primitive.get_renderer(0)
        super().__init__(server, self.render_function, None, "rendering")

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
        if isinstance(self.query_system, FeatureQuerySystem):
            attn_scores = self.query_system.query(image_feature_token, 7)
            print(
                f"\033[92m[HierachicalViewer] Attn scores: {attn_scores.shape}\033[0m"
            )
        else:
            raise ValueError(
                f"Query system type {type(self.query_system)} is not supported, feature query system is required"
            )
        print(
            f"\033[92m[HierachicalViewer] Image feature token: {image_feature_token.shape}\033[0m"
        )
        self._hierachical_primitive.set_attention_colors(attn_scores)

    def hierachical_primitive_initialization(self):
        "set up splat entity"
        print("\033[93m📦 Loading hierarchical primitive...\033[0m")
        self._hierachical_primitive = HierachcalPrimitiveCPUCache(with_feature=True)
        self._hierachical_primitive.load_from_file(self.hierachical_primitive_path)
        # Setup query system
        self.setup_query_system()
        print("\033[93m🔧 Applying feature PCA...\033[0m")
        self._feature_pca()
        self._hierachical_primitive.to("cpu")
        print("\033[92m✓ Hierarchical primitive loaded successfully!\033[0m")

    def setup_query_system(self):
        """Initialize the query system with the provided database and query system type"""
        if self.with_feature and self.database_type == FeatureDatabase:
            self.database: FeatureDatabase = self.database_type(
                num_layers=len(self._hierachical_primitive.source["feature"]),
                features=self._hierachical_primitive.source["feature"],
            )  # initialization of the database
        else:
            self.database: Database = self.database_type(
                num_layers=len(self._hierachical_primitive.source["geometry"])
            )  # initialization of the database
        self.query_system = self.query_system_type(self.database)
        print("\033[92m✓ Successfully initialized database!\033[0m")
        print(f"\033[96m{self.query_system.verbose()}\033[0m")

    def _update_hierachical_primitive_gpu_cache(self):
        cpu_aliased = self._hierachical_primitive.setup_mode(
            self.render_tab_state.render_mode
        )
        if self.render_tab_state.query_index == "":
            gpu_cache = cpu_aliased.copy_to("cuda")
        else:
            query_list = self.query_system.query(self.render_tab_state.query_index)
            assert query_list is not None, "Query list is None, should be handled above"
            gpu_cache = cpu_aliased.query_with_list_torch_index(query_list).copy_to(
                "cuda"
            )
        return gpu_cache

    @property
    def hierachical_primitive(self) -> HierachcalPrimitiveCPUCache:
        no_cache_flag = self._hierachical_primitive_gpu_cache == None
        inconsistent_query_flag = (
            self.viewer_state.current_query_index != self.render_tab_state.query_index
        )
        mode_change_flag = (
            self.render_tab_state.render_mode != self.viewer_state.current_render_mode
        )
        if no_cache_flag or inconsistent_query_flag or mode_change_flag:
            # inconsistent query index, update the cache
            # or no cache, update the cache
            self._hierachical_primitive_gpu_cache = (
                self._update_hierachical_primitive_gpu_cache()
            )
            self.viewer_state.current_render_mode = self.render_tab_state.render_mode
            self.viewer_state.current_query_index = self.render_tab_state.query_index
        return self._hierachical_primitive_gpu_cache

    def reset(self):
        self.render_tab_state = copy.deepcopy(self.viewer_state)
        self.setup_query_system()
        self.hierachical_primitive_initialization()

    def _feature_pca(self):
        if self._hierachical_primitive.with_feature is False:
            return
        else:
            self.feature_pcas: List[torch.Tensor] = []
            for i, feature in enumerate(
                tqdm(self._hierachical_primitive.source["feature"], desc="Feature PCA")
            ):
                if feature.shape[0] <= 3:
                    print("Feature is too small, skipping")
                    zeros = torch.zeros(
                        feature.shape[0], 3, dtype=feature.dtype, device=feature.device
                    )
                    self.feature_pcas.append(zeros)
                    continue
                if i == 0:
                    projected_features, pc_directions, mean = (
                        pca_torch_extract_directions(feature, n_components=3)
                    )
                    self.pc_directions = pc_directions
                    self.mean = mean
                else:
                    projected_features = apply_pca_directions_torch(
                        feature, self.pc_directions, self.mean
                    )

                mins = projected_features.min(dim=0).values  # shape (3,)
                maxs = projected_features.max(dim=0).values  # shape (3,)
                ranges = maxs - mins
                eps = 1e-8
                projected_features = (projected_features - mins) / (ranges + eps)
                self.feature_pcas.append(projected_features)

            self._hierachical_primitive.source["feature"] = self.feature_pcas

    def camera_state_parser(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = camera_state.c2w
        width = render_tab_state.render_width
        height = render_tab_state.render_height
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to("cuda").unsqueeze(0)
        K = torch.from_numpy(K).float().to("cuda").unsqueeze(0)
        return K, c2w, width, height

    @torch.no_grad()
    def render_function(self, camera_state: CameraState, render_tab_state: ViewerState):
        state = self.render_tab_state
        K, c2w, width, height = self.camera_state_parser(
            camera_state, self.render_tab_state
        )
        if state.render_mode == "RGB":
            image = self.curret_renderer_cache.render(
                K, c2w, width, height, state.render_mode
            )
        elif state.render_mode == "Feature":
            image = self.curret_renderer_cache.render(
                K, c2w, width, height, state.render_mode
            )
        elif state.render_mode == "Attention":
            image = self.curret_renderer_cache.render(K, c2w, width, height, "Feature")
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
                    "Attention",
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
                    self._dashboard_thread, self._dashboard_url = (
                        launch_image_dashboard(
                            on_submit=self._on_dashboard_submit,
                            pca_directions=self.pc_directions,
                            pca_mean=self.mean,
                            port=7860,
                            server_name="0.0.0.0",
                            open_browser=False,
                        )
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
                self.render_tab_state.query_index = text_value
                del self.curret_renderer_cache
                torch.cuda.empty_cache()
                self.curret_renderer_cache = self.hierachical_primitive.get_renderer(
                    self.render_tab_state.layer_index
                )
                self.rerender(_)

            @self.layer_index_slider.on_update
            def _(_) -> None:
                self.render_tab_state.layer_index = self.layer_index_slider.value
                del self.curret_renderer_cache
                torch.cuda.empty_cache()
                self.curret_renderer_cache = self.hierachical_primitive.get_renderer(
                    self.render_tab_state.layer_index
                )
                self.rerender(_)

            @self.render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = self.render_mode_dropdown.value
                if self.render_tab_state.render_mode != "RGB":
                    del self.curret_renderer_cache
                    torch.cuda.empty_cache()
                    self.curret_renderer_cache = (
                        self.hierachical_primitive.get_renderer(
                            self.render_tab_state.layer_index
                        )
                    )
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
