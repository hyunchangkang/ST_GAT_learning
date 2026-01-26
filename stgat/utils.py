# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .backends import _HAS_SCATTER, scatter_max, scatter_sum


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bound_log_sigma(raw: torch.Tensor, min_log: float, max_log: float, mode: str = "sigmoid") -> torch.Tensor:
    """Map unconstrained raw -> [min_log, max_log].

    mode='sigmoid': smooth bounded mapping (recommended; avoids zero-gradient clamps).
    mode='clamp'  : hard clamp (compatibility with older checkpoints).
    """
    if max_log <= min_log:
        return torch.full_like(raw, min_log)
    if mode == "clamp":
        return torch.clamp(raw, min_log, max_log)
    return min_log + (max_log - min_log) * torch.sigmoid(raw)


def safe_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Edge-wise softmax over incoming edges per destination node (dst).

    scores: (E, H) or (E,)
    index : (E,) dst indices
    """
    if scores.numel() == 0:
        return scores

    if scores.dim() == 1:
        scores = scores.unsqueeze(-1)

    if _HAS_SCATTER and scatter_max is not None and scatter_sum is not None:
        m, _ = scatter_max(scores, index, dim=0, dim_size=num_nodes)
        scores2 = scores - m[index]
        exp = torch.exp(scores2)
        denom = scatter_sum(exp, index, dim=0, dim_size=num_nodes)
        out = exp / (denom[index] + 1e-12)
        return out.squeeze(-1) if out.size(1) == 1 else out

    out = torch.empty_like(scores)
    for n in torch.unique(index).tolist():
        mask = index == n
        s = scores[mask]
        s = s - s.max(dim=0, keepdim=True).values
        e = torch.exp(s)
        out[mask] = e / (e.sum(dim=0, keepdim=True) + 1e-12)
    return out.squeeze(-1) if out.size(1) == 1 else out


def se2_inv(xytheta: torch.Tensor) -> torch.Tensor:
    x, y, th = xytheta
    c, s = torch.cos(th), torch.sin(th)
    xi = -(c * x + s * y)
    yi = -(-s * x + c * y)
    thi = -th
    return torch.stack([xi, yi, thi])


def se2_apply(xytheta: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    x, y, th = xytheta
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # (2,2)
    return (pts_xy @ R.T) + torch.stack([x, y])


def warp_points_to_frame(pts_xy: torch.Tensor, pose_src: torch.Tensor, pose_dst: torch.Tensor) -> torch.Tensor:
    """pts_xy in src base frame -> dst base frame, using base->world poses."""
    world = se2_apply(pose_src, pts_xy)
    dst_inv = se2_inv(pose_dst)
    return se2_apply(dst_inv, world)


def _edge_attr(src_xy: torch.Tensor, dst_xy: torch.Tensor, dt_edge: torch.Tensor) -> torch.Tensor:
    dxy = src_xy - dst_xy
    dist = torch.sqrt((dxy**2).sum(dim=1) + 1e-12)
    return torch.cat([dxy, dist.unsqueeze(1), dt_edge.unsqueeze(1)], dim=1)


def maybe_dropout_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, p: float, training: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (not training) or p <= 0.0 or edge_index.numel() == 0:
        return edge_index, edge_attr
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p
    return edge_index[:, keep], edge_attr[keep]


def cross_stats_mean_dist_logcnt(
    N: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-node cross-support summary (Radar-side).

    Returns:
      mean_dist: (N,1) mean cross-edge distance into each destination node
      log_cnt  : (N,1) log(1 + in_degree) for each destination node
    """
    if edge_index is None or edge_index.numel() == 0:
        mean_dist = torch.zeros((N, 1), device=device, dtype=dtype)
        log_cnt = torch.zeros((N, 1), device=device, dtype=dtype)
        return mean_dist, log_cnt

    dst = edge_index[1]  # destination nodes (E,)
    dist = edge_attr[:, 2].to(dtype)  # (E,) edge_attr = [dx, dy, dist, dt]
    ones = torch.ones_like(dist)

    cnt = torch.zeros((N,), device=device, dtype=dtype)
    sumd = torch.zeros((N,), device=device, dtype=dtype)
    cnt.index_add_(0, dst, ones)
    sumd.index_add_(0, dst, dist)

    mean = (sumd / (cnt + 1e-6)).unsqueeze(1)
    logc = torch.log1p(cnt).unsqueeze(1)
    return mean, logc


def extract_zip_if_needed(data_zip: Optional[str], data_root: str) -> str:
    if data_zip is None:
        return data_root
    data_root = str(Path(data_root).resolve())
    p = Path(data_root)
    if p.exists() and any(p.glob("*.txt")):
        return data_root
    p.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip, "r") as z:
        z.extractall(p)
    subs = [d for d in p.iterdir() if d.is_dir()]
    if len(subs) == 1:
        return str(subs[0].resolve())
    return data_root


def read_txt_np(path: Path, expected_cols: int, dtype=np.float32) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != expected_cols:
        raise ValueError(f"Unexpected columns in {path.name}: got {arr.shape[1]}, expected {expected_cols}")
    return arr


def assign_to_odom_frames(t_pts: np.ndarray, t_odom: np.ndarray) -> np.ndarray:
    F = t_odom.shape[0]
    idx = np.searchsorted(t_odom, t_pts, side="left")
    idx1 = np.clip(idx, 0, F - 1)
    idx0 = np.clip(idx - 1, 0, F - 1)
    d0 = np.abs(t_pts - t_odom[idx0])
    d1 = np.abs(t_pts - t_odom[idx1])
    use0 = d0 <= d1
    out = np.where(use0, idx0, idx1)
    return out.astype(np.int64)


def sort_by_frame(frame_idx: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(frame_idx, kind="mergesort")
    frame_sorted = frame_idx[order]
    data_sorted = data[order]
    F = int(frame_sorted.max()) + 1 if frame_sorted.size > 0 else 0
    counts = np.bincount(frame_sorted, minlength=F)
    offsets = np.zeros(F + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return data_sorted, offsets
