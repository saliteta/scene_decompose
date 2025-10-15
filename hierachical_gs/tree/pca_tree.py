# pca_tree.py
from typing import Optional, Tuple, List
import torch
import numpy as np
from tqdm import trange

from .tree import HierarchicalTree

class PCABinaryTree(HierarchicalTree):
    def __init__(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        max_depth: Optional[int] = None,
        min_points_per_leaf: int = 1,
    ):
        self.min_points_per_leaf = int(min_points_per_leaf)
        super().__init__(points, features, max_depth, min_points_per_leaf)

    # Override only this; base _build_tree will call it.
    def _assign_points_to_leaves(self):
        # Root gets all points
        root = self.layers[self.max_depth].nodes[0]
        root.point_indices = torch.arange(self.num_points, device=self.points.device, dtype=torch.long)

        # Split top-down until leaves
        for layer_id in trange(self.max_depth, 0, -1, desc="Assigning points to leaves, time increasing exponentially"):
            parent_layer = self.layers[layer_id]

            # Clear next layer's memberships
            for parent in parent_layer.nodes:
                for cid in parent.children_ids:
                    self.all_nodes[cid].point_indices = None

            for parent in parent_layer.nodes:
                idx = parent.point_indices
                if idx is None or idx.numel() == 0:
                    continue

                left_id, right_id = parent.children_ids
                n = idx.numel()
                producing_leaves = (layer_id - 1) == 0

                if n < 2:
                    L, R = idx, torch.empty(0, dtype=idx.dtype, device=idx.device)
                else:
                    if producing_leaves and n < 2 * self.min_points_per_leaf:
                        L, R = self._balanced_by_count(idx, prefer_min=self.min_points_per_leaf)
                    else:
                        L, R = self._pca_half_split(idx)
                        if L.numel() == 0 or R.numel() == 0:
                            L, R = self._widest_axis_half_split(idx)
                        if L.numel() == 0 or R.numel() == 0:
                            L, R = self._balanced_by_count(idx)

                self.all_nodes[left_id].point_indices  = L
                self.all_nodes[right_id].point_indices = R

        self._soft_repair_empty_leaves()

    # -------- splitters (location-based PCA) --------

    @torch.no_grad()
    def _pca_half_split(self, point_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pts = self.points[point_indices].to(torch.float64)
        X = pts - pts.mean(dim=0, keepdim=True)
        if X.abs().max().item() == 0.0:
            return self._widest_axis_half_split(point_indices)
        try:
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            axis = Vh[0]
        except Exception:
            return self._widest_axis_half_split(point_indices)
        proj = (X * axis).sum(dim=1)
        order = torch.argsort(proj, stable=True)
        n = order.numel()
        left_n = n // 2
        return point_indices[order[:left_n]], point_indices[order[left_n:]]

    @torch.no_grad()
    def _widest_axis_half_split(self, point_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pts = self.points[point_indices]
        axis = int(torch.argmax(pts.max(0).values - pts.min(0).values).item())
        vals = pts[:, axis]
        order = torch.argsort(vals, stable=True)
        n = order.numel()
        if n < 2:
            return point_indices, torch.empty(0, dtype=point_indices.dtype, device=point_indices.device)
        left_n = n // 2
        L = point_indices[order[:left_n]]
        R = point_indices[order[left_n:]]
        if L.numel() == 0 or R.numel() == 0:
            return self._balanced_by_count(point_indices)
        return L, R

    @torch.no_grad()
    def _balanced_by_count(self, point_indices: torch.Tensor, prefer_min: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        n = point_indices.numel()
        if n < 2:
            return point_indices, torch.empty(0, dtype=point_indices.dtype, device=point_indices.device)
        perm = torch.randperm(n, device=point_indices.device)
        left_n = max(prefer_min, n // 2)
        right_n = n - left_n
        if right_n == 0:
            left_n -= 1
            right_n = 1
        return point_indices[perm[:left_n]], point_indices[perm[left_n:]]

    @torch.no_grad()
    def _soft_repair_empty_leaves(self):
        leaves = self.layers[0].nodes
        empties = [n for n in leaves if n.point_indices is None or n.point_indices.numel() == 0]
        if not empties:
            return
        donors = [n.point_indices for n in leaves if n.point_indices is not None and n.point_indices.numel() > 1]
        if not donors:
            return
        pool = torch.cat(donors, dim=0)
        take_each = max(1, pool.numel() // (2 * len(leaves)))
        for n in empties:
            ridx = torch.randint(0, pool.numel(), (take_each,), device=pool.device)
            n.point_indices = pool[ridx].clone()
