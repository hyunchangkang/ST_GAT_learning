# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch

from .config import ModelConfig


def mask_inputs_for_mu(cfg: ModelConfig, x_in: torch.Tensor, frame_id: torch.Tensor, sensor_id: torch.Tensor) -> torch.Tensor:
    """
    A-style anti-copy masking for CURRENT-frame supervision.

    - For LiDAR nodes at the last window frame: mask (x,y) so mu-head cannot trivially copy targets.
    - For Radar nodes at the last window frame: mask v_r so mu-head cannot trivially copy targets.

    Edges should be built from the unmasked x_raw (geometry).
    """
    t_idx = cfg.WINDOW - 1
    is_t = (frame_id == t_idx)

    mL = is_t & (sensor_id == 0)
    if mL.any():
        x_in[mL, cfg.IDX_POS[0]] = 0.0
        x_in[mL, cfg.IDX_POS[1]] = 0.0

    mR = is_t & (sensor_id != 0)
    if mR.any():
        x_in[mR, cfg.IDX_VR] = 0.0

    return x_in


def normalize_node_features(cfg: ModelConfig, x: torch.Tensor) -> torch.Tensor:
    """Normalize selected raw values for stable training.

    NOTE:
      - Graph construction (kNN / radius gates / warping) should still use METRIC coordinates.
      - Therefore, we normalize only the features fed into the network and the supervision targets,
        while building edges from the original (metric) x.
    """
    x = x.clone()

    # Positions (x,y)
    if cfg.POS_NORM and cfg.POS_NORM != 1.0:
        x[:, cfg.IDX_POS[0]] = x[:, cfg.IDX_POS[0]] / cfg.POS_NORM
        x[:, cfg.IDX_POS[1]] = x[:, cfg.IDX_POS[1]] / cfg.POS_NORM

    # LiDAR intensity
    if cfg.INTENSITY_NORM and cfg.INTENSITY_NORM != 1.0:
        x[:, cfg.IDX_INTENSITY] = x[:, cfg.IDX_INTENSITY] / cfg.INTENSITY_NORM

    # Radar v_r
    if cfg.VR_NORM and cfg.VR_NORM != 1.0:
        x[:, cfg.IDX_VR] = x[:, cfg.IDX_VR] / cfg.VR_NORM

    # Radar SNR
    if cfg.SNR_NORM and cfg.SNR_NORM != 1.0:
        x[:, cfg.IDX_SNR] = x[:, cfg.IDX_SNR] / cfg.SNR_NORM

    return x


def normalize_edges_inplace(cfg: ModelConfig, edges: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Normalize edge attributes derived from (dx,dy,dist) so they match normalized position scale."""
    if not cfg.POS_NORM or cfg.POS_NORM == 1.0:
        return edges
    for rel, ed in edges.items():
        ea = ed.get("edge_attr", None)
        if ea is None or ea.numel() == 0:
            continue
        # edge_attr = [dx, dy, dist, dt] -> normalize dx/dy/dist only
        ed["edge_attr"][:, 0:3] = ed["edge_attr"][:, 0:3] / cfg.POS_NORM
    return edges

