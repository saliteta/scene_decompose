"""
We are giving a general wrapper for Splat Primitive Viewer.
It support instead of launguage query, we can use image query.
"""
from typing import Type, Optional, Dict
from gsplat_ext import GeneralViewer, Renderer
from viser import ViserServer
from pathlib import Path
from nerfview import  RenderTabState
from dataclasses import dataclass
from typing import Literal
import torch
from ..hierachical_utils.pca_utils import pca_torch_extract_directions
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from nerfview import apply_float_colormap
from .query_data import QueryData
from .image_dashboard import launch_image_dashboard
from nerfview import CameraState

@dataclass
class GeneralViewerWrapperState(RenderTabState):
    current_render_mode: Literal["RGB", "Feature", "Attention"] = "RGB"
    render_mode: Literal["RGB", "Feature", "Attention"] = "RGB"
    pca_cached: bool = False
    current_query: str = ""
    system_query: str = ""

class GeneralViewerWrapper(GeneralViewer):
    def __init__(self, server: ViserServer, 
                       primitive_path: Path, 
                       render_tab_state: GeneralViewerWrapperState, 
                       splat_method: Literal["3DGS", "2DGS", "DBS"] = "3DGS",
                       port=8080, with_feature: bool = False, 
                       feature_path: Optional[Path] = None
                       ):

        self.splat_method = splat_method
        self.with_feature = with_feature
        self.feature_path = feature_path
        self.render_tab_state = render_tab_state
        self.splat_path = primitive_path

        self.splat_method_selection() # get splat and renderer
        self.setup()
        super(GeneralViewer, self).__init__(server, self.render_function, None, 'rendering')


    def setup(self):
        self._splat_cpu_cache = self.splat.to("cpu") # original features to CPU
        del self.splat, self.renderer
        torch.cuda.empty_cache()
        self.pca_features  
        self._cached_renderer = None
        self._attention_scores = None
        self._gt_attention_scores = None
        self._dashboard_started = False
        self._gt_db = None


    def feature_pca(self) -> torch.Tensor:
        features = self._splat_cpu_cache.feature
        features = features.reshape(features.shape[0], -1)
        projected_features, pc_directions, mean = pca_torch_extract_directions(features, n_components=3)
        self.pc_directions = pc_directions
        self.mean = mean
        mins   = projected_features.min(dim=0).values    # shape (3,)
        maxs   = projected_features.max(dim=0).values    # shape (3,)
        ranges = maxs - mins
        eps    = 1e-8
        projected_features = (projected_features - mins) / (ranges + eps)
        return projected_features.to("cuda")

    @property
    def pca_features(self) -> torch.Tensor: # [N, 3] in cuda
        if self.render_tab_state.pca_cached is False:
            self._pca_features = self.feature_pca().to("cuda")
            self.render_tab_state.pca_cached = True
        return self._pca_features

    @property
    def attention_scores(self) -> torch.Tensor: # [N,4] in cuda, first is score, last three are color
        # get attention scores on update and move to CUDA
        if self._attention_scores is None:
            # Get number of splats from CPU cache
            N = self._splat_cpu_cache.geometry["means"].shape[0]
            
            # Randomly initialize attention scores: [N, 4]
            # First column: random attention scores in [0, 1]
            # Last 3 columns: random RGB colors in [0, 1]
            random_scores = torch.rand(N, 1, device="cuda")
            random_colors = torch.rand(N, 3, device="cuda")
            self._attention_scores = torch.cat([random_scores, random_colors], dim=1)
            
            # Warn user
            print("\033[33m[GeneralViewerWrapper] Warning: Attention scores not initialized. "
                  "Using random values for visualization. Please run a query to get proper attention scores.\033[0m")
        
        if self._attention_scores.shape[1] == 1:
            "only attention score, no color, need to convert to color"
            attention_colors = self.attention_score_to_color(self._attention_scores)
            self._attention_scores = torch.cat([self._attention_scores, attention_colors], dim=1)

            print("\033[92m[GeneralViewerWrapper] Attention scores converted to colors successfully!\033[0m")
        
        return self._attention_scores

    def attention_score_to_color(self, attention_scores: torch.Tensor) -> torch.Tensor:
        '''
        Convert attention scores to colors
        Args:
            attention_scores: torch.Tensor (N,)
        Returns:
            colors: torch.Tensor (N, 3)
        '''
        # Scale the attention scores from 0 to 1
        print(f"attention_scores shape: {attention_scores[:10]}")
        attention_scores = attention_scores.clamp(min = attention_scores.mean())
        min_score = attention_scores.min()
        max_score = attention_scores.max()
        eps = torch.finfo(attention_scores.dtype).eps if attention_scores.is_floating_point() else 1e-8
        denom = (max_score - min_score) + eps
        attention_scores = (attention_scores - min_score) / denom + eps
        attention_scores = attention_scores.clamp(0.0, 1.0)
        colors = apply_float_colormap(attention_scores, 'turbo')
        print(f"\033[92m[GeneralViewerWrapper] Attention scores converted to colors successfully!, get {colors.shape} colors\033[0m")
        return colors
    
    @property
    def gt_attention_scores(self) -> torch.Tensor: # [N,4] in cuda, first is score, last three are color
        # get ground truth attention scores on update and move to CUDA
        # If gt database is not provided, return None
        
        # get attention scores on update and move to CUDA
        if self._gt_attention_scores is None:
            # Get number of splats from CPU cache
            N = self._splat_cpu_cache.geometry["means"].shape[0]
            
            # Randomly initialize attention scores: [N, 4]
            # First column: random attention scores in [0, 1]
            # Last 3 columns: random RGB colors in [0, 1]
            random_scores = torch.rand(N, 1, device="cuda")
            random_colors = torch.rand(N, 3, device="cuda")
            self._gt_attention_scores = torch.cat([random_scores, random_colors], dim=1)
            
            # Warn user
            print("\033[33m[GeneralViewerWrapper] Warning: Attention scores not initialized. "
                  "Using random values for visualization. Please run a query to get proper attention scores.\033[0m")
        
        if self._gt_attention_scores.shape[1] == 1:
            "only attention score, no color, need to convert to color"
            attention_colors = self.attention_score_to_color(self._gt_attention_scores)
            self._gt_attention_scores = torch.cat([self._gt_attention_scores, attention_colors], dim=1)

            print("\033[92m[GeneralViewerWrapper] Attention scores converted to colors successfully!\033[0m")
        
        return self._gt_attention_scores
    
    @property 
    def cached_renderer(self) -> Renderer:
        # cached mode changing, or attention update, or something else
        # or cache not initialized
        if self._cached_renderer is None:
            splat = self._splat_cpu_cache.to("cuda") # get a copy of the splat on cuda 
            if self.splat_method == "3DGS":
                self._cached_renderer = GaussianRenderer(splat)
            elif self.splat_method == "2DGS":
                self._cached_renderer = GaussianRenderer2D(splat)
            elif self.splat_method == "DBS":
                self._cached_renderer = BetaSplatRenderer(splat)
            else:
                raise ValueError(f"Invalid splat method: {self.splat_method}")
            
            if self.render_tab_state.render_mode == "Attention":
                self._cached_renderer.primitives._feature = self.attention_scores[:,1:]
            elif self.render_tab_state.render_mode == "gt_Attention":
                self._cached_renderer.primitives._feature = self.gt_attention_scores[:, 1:]
            print("\033[92m[GeneralViewerWrapper] Renderer cached successfully!\033[0m")

        ### default renderer feature is pca features
        if self.render_tab_state.render_mode == "RGB":
            self.render_tab_state.current_render_mode = "RGB"
            return self._cached_renderer
        
        else:
            if self.render_tab_state.render_mode == self.render_tab_state.current_render_mode:
                return self._cached_renderer
            else:
                self.render_tab_state.current_render_mode = self.render_tab_state.render_mode
                if self.render_tab_state.render_mode == "Feature":
                    self._cached_renderer.primitives._feature = self.pca_features
                elif self.render_tab_state.render_mode == "Attention":
                    self._cached_renderer.primitives._feature = self.attention_scores[:, 1:]
                elif self.render_tab_state.render_mode == "gt_Attention":
                    self._cached_renderer.primitives._feature = self.gt_attention_scores[:, 1:]
                else:
                    raise ValueError(f"Invalid render mode: {self.render_tab_state.render_mode}")
                return self._cached_renderer

    def attention_query(self, query_feature: torch.Tensor) -> torch.Tensor:
        """
        Query the attention scores of the query feature
        Args:
            query_feature: torch.Tensor (F,)
        Returns:
            attention_scores: torch.Tensor (N,)
        """
        query_feature = query_feature.to("cuda")
        splat_features = self._splat_cpu_cache.feature.to("cuda")
        attention_scores = torch.matmul(query_feature, splat_features.transpose(0, 1))
        return attention_scores
    
    def gt_attention_query(self, gt_camera_pose: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Query the attention scores of the gt camera pose
        Args:
            gt_camera_pose: Dict containing camera pose, K, height, width
        Returns:
            attention_scores: torch.Tensor (N,) where N is number of Gaussians
            - Values are set for splats visible in the image (based on ids)
            - Remaining values are -1 (indicating not visible)
        """
        camera_pose = gt_camera_pose["camtoworld"]
        K = gt_camera_pose["K"]
        pixels = gt_camera_pose["image"]
        height, width = pixels.shape[0:2]
        features = torch.ones(height, width, 1, device="cuda")
        eps = torch.finfo(features.dtype).eps
        splat_features_per_image, splat_weights_per_image, ids = self.cached_renderer.inverse_render(
            K=K.to("cuda").unsqueeze(0), extrinsic=camera_pose.to("cuda").unsqueeze(0), width=width, height=height, features=features.unsqueeze(0)
        )
        splat_features_per_image /= splat_weights_per_image[..., None] + eps
        splat_features_per_image = splat_features_per_image.unsqueeze(-1)
        
        # Get number of Gaussians
        N = self._splat_cpu_cache.geometry["means"].shape[0]
        
        # Initialize attention scores with -1 for all Gaussians
        attention_scores = torch.full((N,), -1.0, device="cuda", dtype=splat_features_per_image.dtype)
        
        # Replace values for splats that appear in the image (have valid ids)
        # ids contains the indices of splats visible in this camera view
        if ids.numel() > 0:  # Check if there are any visible splats
            valid_mask = ids >= 0  # Filter out invalid IDs (typically -1)
            valid_ids = ids[valid_mask]
            valid_features = splat_features_per_image[valid_mask].flatten()
            
            # Assign attention scores to the corresponding splat indices
            attention_scores[valid_ids] = valid_features
        
        print(f"\033[33m[GeneralViewerWrapper] GT attention scores min: {attention_scores.min()}, GT attention scores max: {attention_scores.max()}\033[0m")
        return attention_scores

    def _on_dashboard_submit(self, query_data: QueryData):
        """
        This is the hook of the dashboard thread holding by Gradio
        It is supposed to process the 2D image, returning the feature token of the image.
        And when it return the result, we are supposed to run the image feature query in this method

        It should also corresponds to gt attn scores
        """
        if query_data.query_type == "image_feature":
            query_feature = query_data.image_feature_token
            self._attention_scores = self.attention_query(query_feature).unsqueeze(-1) # [N, 1]
            print(f"\033[92m[GeneralViewerWrapper] Querying image feature successfully!, get {self._attention_scores.shape} attention scores\033[0m")
        elif query_data.query_type == "gt_camera_pose":
            gt_camera = query_data.gt_camera
            self._gt_attention_scores = self.gt_attention_query(gt_camera).unsqueeze(-1) # [N, 1]
            print(f"\033[92m[GeneralViewerWrapper] Querying gt camera pose successfully!, get {self._gt_attention_scores.shape} attention scores\033[0m")
        else:
            raise ValueError(f"Invalid query type: {query_data.query_type}")


    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:

            self.render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                (
                    "RGB",
                    "Feature",
                    "Attention",
                    "gt_Attention",
                ),
                initial_value=self.render_tab_state.render_mode,
                hint="Render mode to use.",
            )

            self.gt_location = server.gui.add_text(
                "GT Location",
                initial_value="/data1/COLMAP_RESULT/CUHK_LOWER/",
                hint="GT location to use.",
            )

            # NEW: Image Query button
            self.image_query_btn = server.gui.add_button("Image Query")
            @self.image_query_btn.on_click
            def _(_evt):
                self._gt_db = Path(self.gt_location.value)
                print(f"\033[92m[GeneralViewerWrapper] GT database path set to {self._gt_db}\033[0m")
                # Start dashboard once
                if not self._dashboard_started:
                    self._dashboard_thread, self._dashboard_url = launch_image_dashboard(
                        on_submit=self._on_dashboard_submit,
                        pca_directions=self.pc_directions,
                        pca_mean=self.mean,
                        port=7860,
                        server_name="0.0.0.0",
                        open_browser=False,
                        gt_db=self._gt_db,
                    )
                    self._dashboard_started = True

                # Provide a clickable link inside Viser
                server.add_gui_markdown(f"[Open 2D Dashboard]({self._dashboard_url})")


            @self.render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = self.render_mode_dropdown.value
                self.cached_renderer # update
                self.rerender(_)



        self._rendering_tab_handles.update(
            {
                "render_mode_dropdown": self.render_mode_dropdown,
                "image_query_btn": self.image_query_btn,
            }
        )

        super(GeneralViewer, self)._populate_rendering_tab()
                

    @torch.no_grad()
    def render_function(self, camera_state: CameraState, render_tab_state: GeneralViewerWrapperState):
        state = self.render_tab_state
        K, c2w, width, height = self.camera_state_parser(camera_state, self.render_tab_state)
        if state.render_mode == "RGB":
            image = self.cached_renderer.render(K, c2w, width, height, state.render_mode)
        elif state.render_mode == "Feature":
            image = self.cached_renderer.render(K, c2w, width, height, state.render_mode)
        elif state.render_mode == "Attention":
            image = self.cached_renderer.render(K, c2w, width, height, "Feature")
        elif state.render_mode == "gt_Attention":
            image = self.cached_renderer.render(K, c2w, width, height, "Feature")
        else:
            raise ValueError(f"Unsupported render mode: {state.render_mode}")
        return image.cpu().numpy()
