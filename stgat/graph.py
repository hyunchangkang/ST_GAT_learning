# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, Tuple

import torch

from .config import ModelConfig
from .backends import (
    _HAS_TORCH_CLUSTER,
    knn_graph,
    knn,
    _HAS_SCATTER,
    _HAS_SCATTER_MIN,
    scatter_sum,
    scatter_min,
)
from .utils import warp_points_to_frame, _edge_attr, maybe_dropout_edges, cross_stats_mean_dist_logcnt


class GraphBuilder:
    """
    Build edges for flattened nodes across a batch.

    Inputs per batch:
      x:        (N, 11)
      frame_id: (N,) in [0..WINDOW-1]
      batch_id: (N,)
      sensor_id:(N,) {0:LiDAR, 1:Radar1, 2:Radar2}
      pose_by_frame: (B, WINDOW or WINDOW+1, 3) base->world poses for window frames (and optional next)
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @torch.no_grad()
    def _knn_within(self, pts: torch.Tensor, k: int) -> torch.Tensor:
        """Return edge_index (2,E) for kNN within one set."""
        M = pts.size(0)
        if M == 0 or k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=pts.device)

        kk = min(k, max(M - 1, 1))
        if _HAS_TORCH_CLUSTER:
            # knn_graph returns edges as [src, dst] (neighbor -> query) for flow='source_to_target'
            return knn_graph(pts, k=kk, loop=False, flow='source_to_target')
        # fallback: cdist
        d = torch.cdist(pts, pts)
        d.fill_diagonal_(1e9)
        nn_idx = torch.topk(d, k=kk, largest=False, dim=1).indices
        dst = torch.arange(M, device=pts.device).unsqueeze(1).expand(M, kk).reshape(-1)
        src = nn_idx.reshape(-1)
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def _knn_cross(self, src_pts: torch.Tensor, dst_pts: torch.Tensor, k_per_dst: int) -> torch.Tensor:
        """
        Return pairs for kNN from src_pts to each dst_pt:
          edge_index (2,E) with indices in src/dst LOCAL coordinates: [src_idx, dst_idx]
        """
        Ns = src_pts.size(0)
        Nd = dst_pts.size(0)
        if Ns == 0 or Nd == 0 or k_per_dst <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=dst_pts.device)

        kk = min(k_per_dst, Ns)
        if _HAS_TORCH_CLUSTER:
            # torch_cluster.knn(x, y, k) returns [y_index, x_index] (query indices, neighbor indices)
            pair = knn(src_pts, dst_pts, k=kk)  # (2,E) : [dst, src]
            dst = pair[0]
            src = pair[1]
            return torch.stack([src, dst], dim=0)

        # fallback: cdist
        d = torch.cdist(dst_pts, src_pts)  # (Nd, Ns)
        nn = torch.topk(d, k=kk, largest=False, dim=1).indices  # (Nd,kk)
        dst = torch.arange(Nd, device=dst_pts.device).unsqueeze(1).expand(Nd, kk).reshape(-1)
        src = nn.reshape(-1)
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def build(
        self,
        x: torch.Tensor,
        frame_id: torch.Tensor,
        batch_id: torch.Tensor,
        sensor_id: torch.Tensor,
        pose_by_frame: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        device = x.device
        cfg = self.cfg

        edges: Dict[str, Dict[str, torch.Tensor]] = {}
        def init_rel(rel: str):
            edges[rel] = {
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=device),
                "edge_attr": torch.empty((0, cfg.EDGE_DIM), dtype=torch.float32, device=device),
            }
        for rel in ["LL", "R1R1", "R2R2", "L2R1", "R12L", "L2R2", "R22L"]:
            init_rel(rel)

        if batch_id.numel() == 0:
            return edges
        B = int(batch_id.max().item()) + 1

        for b in range(B):
            bmask = (batch_id == b)
            if not bmask.any():
                continue
            idx_b = torch.where(bmask)[0]
            xb = x[idx_b]
            fb = frame_id[idx_b]
            sb = sensor_id[idx_b]

            # Spatial edges within each frame, per sensor
            for f in range(cfg.WINDOW):
                fmask = (fb == f)
                if not fmask.any():
                    continue
                idx_f = torch.where(fmask)[0]  # local indices in idx_b

                for sid, rel, k in [(0, "LL", cfg.K_LIDAR_SPATIAL), (1, "R1R1", cfg.K_RADAR_SPATIAL), (2, "R2R2", cfg.K_RADAR_SPATIAL)]:
                    sm = (sb[idx_f] == sid)
                    if not sm.any():
                        continue
                    loc = idx_f[sm]
                    pts = xb[loc][:, list(cfg.IDX_POS)]
                    ei_local = self._knn_within(pts, k)
                    if ei_local.numel() == 0:
                        continue
                    src_loc = loc[ei_local[0]]
                    dst_loc = loc[ei_local[1]]
                    dt_edge = torch.zeros(src_loc.size(0), device=device)
                    eattr = _edge_attr(xb[src_loc][:, list(cfg.IDX_POS)], xb[dst_loc][:, list(cfg.IDX_POS)], dt_edge)
                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src_loc], idx_b[dst_loc]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

            # Temporal edges between adjacent frames only: (0->1), (1->2), (2->3)
            if cfg.TEMPORAL_ADJ_ONLY:
                pairs = [(0, 1), (1, 2), (2, 3)]
            else:
                pairs = [(i, j) for i in range(cfg.WINDOW) for j in range(cfg.WINDOW) if i < j]

            # Pose availability check
            has_pose = pose_by_frame is not None and pose_by_frame.dim() == 3 and pose_by_frame.size(0) > b and pose_by_frame.size(1) >= cfg.WINDOW

            for f0, f1 in pairs:
                idx0 = torch.where(fb == f0)[0]
                idx1 = torch.where(fb == f1)[0]
                if idx0.numel() == 0 or idx1.numel() == 0:
                    continue

                pose0 = pose1 = None
                if has_pose:
                    pose0 = pose_by_frame[b, f0]
                    pose1 = pose_by_frame[b, f1]

                for sid, rel in [(0, "LL"), (1, "R1R1"), (2, "R2R2")]:
                    a0 = idx0[sb[idx0] == sid]
                    a1 = idx1[sb[idx1] == sid]
                    if a0.numel() == 0 or a1.numel() == 0:
                        continue

                    pts0 = xb[a0][:, list(cfg.IDX_POS)]
                    pts1 = xb[a1][:, list(cfg.IDX_POS)]

                    # Warp src (f0) points into dst (f1) frame before kNN
                    pts0_warp = pts0
                    if pose0 is not None and pose1 is not None:
                        pts0_warp = warp_points_to_frame(pts0, pose_src=pose0, pose_dst=pose1)

                    # kNN from src-set to each dst point: gives edges src->dst
                    ei = self._knn_cross(pts0_warp, pts1, cfg.K_TEMPORAL)  # local src in a0, local dst in a1
                    if ei.numel() == 0:
                        continue

                    src = a0[ei[0]]
                    dst = a1[ei[1]]

                    # dt_edge: |dt_src - dt_dst| is the frame-to-frame time gap
                    dt_edge = (xb[src, cfg.IDX_DT] - xb[dst, cfg.IDX_DT]).abs()

                    src_xy = pts0_warp[ei[0]]
                    dst_xy = pts1[ei[1]]
                    eattr = _edge_attr(src_xy, dst_xy, dt_edge)


                    # Distance gate + per-dst fallback (vectorized when torch_scatter is available)
                    # - Keep all edges within radius.
                    # - If a dst node has zero in-radius edges, keep its nearest edge (fallback) to avoid disconnects.
                    radius = cfg.TEMPORAL_RADIUS_LIDAR if sid == 0 else cfg.TEMPORAL_RADIUS_RADAR
                    if radius is not None and radius > 0:
                        # squared distance (avoid sqrt for speed)
                        dist2 = ((src_xy - dst_xy) ** 2).sum(dim=1)
                        r2 = float(radius * radius)
                        dst_local = ei[1]  # local indices in a1
                        keep_in = dist2 <= r2

                        num_dst = int(dst_local.max().item()) + 1
                        keep_final = keep_in.clone()

                        if _HAS_SCATTER:
                            # dst nodes that already have at least one in-radius edge
                            any_keep = scatter_sum(keep_in.float(), dst_local, dim=0, dim_size=num_dst) > 0
                            need_fb = ~any_keep
                            if need_fb.any():
                                if _HAS_SCATTER_MIN:
                                    _, argmin = scatter_min(dist2, dst_local, dim=0, dim_size=num_dst)
                                    fb = argmin[need_fb]
                                    fb = fb[fb >= 0]
                                    if fb.numel() > 0:
                                        keep_final[fb] = True
                                else:
                                    # scatter_min unavailable: fall back to a small python loop over dst nodes needing fallback
                                    for j in torch.where(need_fb)[0].tolist():
                                        mj = (dst_local == j)
                                        if not mj.any():
                                            continue
                                        dj = dist2[mj]
                                        local_argmin = torch.argmin(dj)
                                        edge_idx = torch.where(mj)[0][local_argmin]
                                        keep_final[edge_idx] = True
                        else:
                            # No scatter backend: python loop (correct but slower)
                            keep_final = torch.zeros_like(keep_in)
                            for j in range(num_dst):
                                mj = (dst_local == j)
                                if not mj.any():
                                    continue
                                kij = keep_in & mj
                                if kij.any():
                                    keep_final |= kij
                                else:
                                    dj = dist2[mj]
                                    local_argmin = torch.argmin(dj)
                                    edge_idx = torch.where(mj)[0][local_argmin]
                                    keep_final[edge_idx] = True

                        src = src[keep_final]
                        dst = dst[keep_final]
                        dt_edge = dt_edge[keep_final]
                        eattr = eattr[keep_final]

                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src], idx_b[dst]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

            # Cross edges: only on last frame in the window (current)
            f = cfg.WINDOW - 1
            idx_f = torch.where(fb == f)[0]
            if idx_f.numel() > 0:
                l = idx_f[sb[idx_f] == 0]
                r1 = idx_f[sb[idx_f] == 1]
                r2 = idx_f[sb[idx_f] == 2]

                def add_cross(rel: str, src_loc: torch.Tensor, dst_loc: torch.Tensor, k: int):
                    if src_loc.numel() == 0 or dst_loc.numel() == 0 or k <= 0:
                        return
                    ptsS = xb[src_loc][:, list(cfg.IDX_POS)]
                    ptsD = xb[dst_loc][:, list(cfg.IDX_POS)]
                    ei = self._knn_cross(ptsS, ptsD, k)  # [src, dst] local indices
                    if ei.numel() == 0:
                        return
                    # radius gate (post-filter)
                    src_xy = ptsS[ei[0]]
                    dst_xy = ptsD[ei[1]]
                    dist = torch.sqrt(((src_xy - dst_xy) ** 2).sum(dim=1) + 1e-12)
                    keep = dist <= cfg.CROSS_RADIUS
                    if not keep.any():
                        return
                    ei = ei[:, keep]
                    src_xy = src_xy[keep]
                    dst_xy = dst_xy[keep]

                    src = src_loc[ei[0]]
                    dst = dst_loc[ei[1]]
                    dt_edge = torch.zeros(src.size(0), device=device)
                    eattr = _edge_attr(src_xy, dst_xy, dt_edge)
                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src], idx_b[dst]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

                # LiDAR <-> Radar1
                add_cross("L2R1", l, r1, cfg.K_CROSS_L2R)
                add_cross("R12L", r1, l, cfg.K_CROSS_R2L)
                # LiDAR <-> Radar2
                add_cross("L2R2", l, r2, cfg.K_CROSS_L2R)
                add_cross("R22L", r2, l, cfg.K_CROSS_R2L)

        return edges


