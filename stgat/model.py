# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import ModelConfig
from .gat_layers import EdgeAwareGAT, RelationalGATBlock
from .utils import bound_log_sigma, maybe_dropout_edges, cross_stats_mean_dist_logcnt


class DOGMSTGATUncertaintyNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.enc_lidar = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))
        self.enc_r1    = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))
        self.enc_r2    = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))

        self.st1_LL = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st1_R1 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st1_R2 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)

        self.st2_L2R1 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_R12L = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_L2R2 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_R22L = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)

        self.adap_back = nn.Linear(cfg.HIDDEN_DIM * cfg.NUM_HEADS, cfg.HIDDEN_DIM)
        out_dim = cfg.HIDDEN_DIM * cfg.NUM_HEADS

        # Mean heads (mu): do NOT need access to the current supervised measurement values
        self.head_lidar_mu = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 2))
        self.head_r1_mu    = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 1))
        self.head_r2_mu    = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 1))

        # Uncertainty heads (log_sigma): may condition on raw current measurements (passed separately as x_raw)
        # LiDAR: add raw (x,y) -> +2 dims
        self.head_lidar_sigma = nn.Sequential(nn.Linear(out_dim + 2, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 2))
        # Radar: add raw (v_r, snr) -> +2 dims
        self.head_r1_sigma    = nn.Sequential(nn.Linear(out_dim + 4, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 1))
        self.head_r2_sigma    = nn.Sequential(nn.Linear(out_dim + 4, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 1))

    def cross_alpha(self, global_step: int) -> float:
        cfg = self.cfg
        if global_step < cfg.CROSS_WARMUP_STEPS:
            return 0.0
        t = global_step - cfg.CROSS_WARMUP_STEPS
        if t >= cfg.CROSS_RAMP_STEPS:
            return cfg.CROSS_ALPHA_MAX
        return cfg.CROSS_ALPHA_MAX * (t / max(cfg.CROSS_RAMP_STEPS, 1))

    def forward(
        self,
        x: torch.Tensor,
        frame_id: torch.Tensor,
        batch_id: torch.Tensor,
        sensor_id: torch.Tensor,
        edges: Dict[str, Dict[str, torch.Tensor]],
        global_step: int = 0,
        x_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        N = x.size(0)
        device = x.device
        min_log_lidar = float(math.log(max(cfg.MIN_SIGMA_LIDAR / max(cfg.POS_NORM, 1e-12), 1e-12)))
        min_log_radar = float(math.log(max(cfg.MIN_SIGMA_RADAR / max(cfg.VR_NORM, 1e-12), 1e-12)))
        if x_raw is None:
            x_raw = x

        h = torch.zeros((N, cfg.HIDDEN_DIM), device=device, dtype=x.dtype)
        mL  = (sensor_id == 0)
        mR1 = (sensor_id == 1)
        mR2 = (sensor_id == 2)
        if mL.any():
            h[mL]  = self.enc_lidar(x[mL])
        if mR1.any():
            h[mR1] = self.enc_r1(x[mR1])
        if mR2.any():
            h[mR2] = self.enc_r2(x[mR2])

        # Stage1: intra-sensor ST graph
        h_st1 = torch.zeros((N, cfg.HIDDEN_DIM * cfg.NUM_HEADS), device=device, dtype=x.dtype)
        hL = self.st1_LL(h, edges["LL"]["edge_index"], edges["LL"]["edge_attr"]) if mL.any() else h_st1
        h1 = self.st1_R1(h, edges["R1R1"]["edge_index"], edges["R1R1"]["edge_attr"]) if mR1.any() else h_st1
        h2 = self.st1_R2(h, edges["R2R2"]["edge_index"], edges["R2R2"]["edge_attr"]) if mR2.any() else h_st1
        h_st1[mL]  = hL[mL]
        h_st1[mR1] = h1[mR1]
        h_st1[mR2] = h2[mR2]

        # Stage2: cross-sensor residual (only affects last window frame nodes through cross edges)
        alpha = self.cross_alpha(global_step)

        eL2R1_i, eL2R1_a = maybe_dropout_edges(edges["L2R1"]["edge_index"], edges["L2R1"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eR12L_i, eR12L_a = maybe_dropout_edges(edges["R12L"]["edge_index"], edges["R12L"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eL2R2_i, eL2R2_a = maybe_dropout_edges(edges["L2R2"]["edge_index"], edges["L2R2"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eR22L_i, eR22L_a = maybe_dropout_edges(edges["R22L"]["edge_index"], edges["R22L"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)

        h_base = self.adap_back(h_st1)
        h_cross = torch.zeros_like(h_st1)

        if alpha > 0.0:
            if eL2R1_i.numel() > 0:
                upd = self.st2_L2R1(h_base, eL2R1_i, eL2R1_a)
                h_cross[mR1] += upd[mR1]
            if eR12L_i.numel() > 0:
                upd = self.st2_R12L(h_base, eR12L_i, eR12L_a)
                h_cross[mL] += upd[mL]
            if eL2R2_i.numel() > 0:
                upd = self.st2_L2R2(h_base, eL2R2_i, eL2R2_a)
                h_cross[mR2] += upd[mR2]
            if eR22L_i.numel() > 0:
                upd = self.st2_R22L(h_base, eR22L_i, eR22L_a)
                h_cross[mL] += upd[mL]

        h_final = h_st1 + (alpha * h_cross)
        # Cross-support summaries for Radar sigma heads (mean cross distance, log in-degree)
        mean_d_r1, logc_r1 = cross_stats_mean_dist_logcnt(N, eL2R1_i, eL2R1_a, device, x.dtype)
        mean_d_r2, logc_r2 = cross_stats_mean_dist_logcnt(N, eL2R2_i, eL2R2_a, device, x.dtype)

        # Heads (A-style): mu from h_final; log_sigma from [h_final, raw_measurements]
        lidar_mu = torch.zeros((N, 2), device=device, dtype=x.dtype)
        lidar_log_sigma = torch.full((N, 2), min_log_lidar, device=device, dtype=x.dtype)

        r1_mu = torch.zeros((N, 1), device=device, dtype=x.dtype)
        r1_log_sigma = torch.full((N, 1), min_log_radar, device=device, dtype=x.dtype)

        r2_mu = torch.zeros((N, 1), device=device, dtype=x.dtype)
        r2_log_sigma = torch.full((N, 1), min_log_radar, device=device, dtype=x.dtype)

        if mL.any():
            lidar_mu[mL] = self.head_lidar_mu(h_final[mL]).to(x.dtype)
            sigma_in = torch.cat([h_final[mL], x_raw[mL][:, list(cfg.IDX_POS)]], dim=1)
            lidar_log_sigma[mL] = bound_log_sigma(self.head_lidar_sigma(sigma_in), min_log_lidar, cfg.MAX_LOG_SIGMA, cfg.SIGMA_BOUND_MODE).to(x.dtype)

        if mR1.any():
            r1_mu[mR1] = self.head_r1_mu(h_final[mR1]).to(x.dtype)
            sigma_in = torch.cat([h_final[mR1], x_raw[mR1][:, [cfg.IDX_VR, cfg.IDX_SNR]], mean_d_r1[mR1], logc_r1[mR1]], dim=1)
            r1_log_sigma[mR1] = bound_log_sigma(self.head_r1_sigma(sigma_in), min_log_radar, cfg.MAX_LOG_SIGMA, cfg.SIGMA_BOUND_MODE).to(x.dtype)

        if mR2.any():
            r2_mu[mR2] = self.head_r2_mu(h_final[mR2]).to(x.dtype)
            sigma_in = torch.cat([h_final[mR2], x_raw[mR2][:, [cfg.IDX_VR, cfg.IDX_SNR]], mean_d_r2[mR2], logc_r2[mR2]], dim=1)
            r2_log_sigma[mR2] = bound_log_sigma(self.head_r2_sigma(sigma_in), min_log_radar, cfg.MAX_LOG_SIGMA, cfg.SIGMA_BOUND_MODE).to(x.dtype)

        return {
            "h_final": h_final,
            "lidar_mu": lidar_mu,
            "lidar_log_sigma": lidar_log_sigma,
            "r1_mu": r1_mu,
            "r1_log_sigma": r1_log_sigma,
            "r2_mu": r2_mu,
            "r2_log_sigma": r2_log_sigma,
            "frame_id": frame_id,
            "batch_id": batch_id,
            "sensor_id": sensor_id,
            "cross_alpha": torch.tensor(alpha, device=device, dtype=x.dtype),
        }


