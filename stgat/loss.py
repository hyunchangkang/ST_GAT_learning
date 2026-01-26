# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class UncertaintyLoss(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def gaussian_nll_vec2(self, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (2.0 * log_sigma + (target - mu) ** 2 / (var + 1e-12))
        return nll.sum(dim=1)

    def gaussian_nll_scalar(self, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (2.0 * log_sigma + (target - mu) ** 2 / (var + 1e-12))
        return nll.squeeze(1)

    @torch.no_grad()
    def _soft_assoc(self, pred_xy: torch.Tensor, gt_xy: torch.Tensor, topk: int, tau: float, gate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        M = pred_xy.size(0)
        G = gt_xy.size(0)
        device = pred_xy.device
        if M == 0 or G == 0:
            return torch.empty((0, 2), device=device), torch.empty((0,), dtype=torch.bool, device=device)

        # NOTE: For training, M and G are typically <= 512; cdist here is acceptable.
        d = torch.cdist(pred_xy, gt_xy)
        k = min(topk, G)
        nn_d, nn_idx = torch.topk(d, k=k, largest=False, dim=1)
        min_d = nn_d[:, 0]
        valid = min_d <= gate
        w = torch.softmax(-(nn_d ** 2) / max(tau ** 2, 1e-6), dim=1)
        cand = gt_xy[nn_idx]
        gt_expect = (w.unsqueeze(-1) * cand).sum(dim=1)
        return gt_expect, valid

    def forward(
        self,
        out_t: Dict[str, torch.Tensor],
        x_t: torch.Tensor,
        frame_id_t: torch.Tensor,
        batch_id_t: torch.Tensor,
        sensor_id_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Current-frame (t) supervision.

        We supervise predictions for nodes that belong to the last window frame (frame_id == WINDOW-1)
        against their own measurements at that same frame.

        NOTE: If the mean heads can trivially copy the target from inputs (e.g., LiDAR x/y in node features,
        Radar v_r in node features), the residuals can collapse toward zero and sigma can collapse as well.
        If that happens, consider masking the supervised fields for the last frame when feeding the model,
        or switching to an innovation / masked-modelling objective.
        """
        cfg = self.cfg
        device = x_t.device

        t_idx = cfg.WINDOW - 1
        is_t = (frame_id_t == t_idx)

        mL  = (sensor_id_t == 0) & is_t
        mR1 = (sensor_id_t == 1) & is_t
        mR2 = (sensor_id_t == 2) & is_t

        B = int(batch_id_t.max().item()) + 1 if batch_id_t.numel() > 0 else 0

        lidar_losses: List[torch.Tensor] = []
        r1_losses: List[torch.Tensor] = []
        r2_losses: List[torch.Tensor] = []

        for b in range(B):
            idxL  = torch.where((batch_id_t == b) & mL)[0]
            idxR1 = torch.where((batch_id_t == b) & mR1)[0]
            idxR2 = torch.where((batch_id_t == b) & mR2)[0]

            # LiDAR position loss (mu + sigma)
            if idxL.numel() > 0:
                mu = out_t["lidar_mu"][idxL]
                logs = out_t["lidar_log_sigma"][idxL]
                gt_xy = x_t[idxL][:, list(cfg.IDX_POS)]

                # (A) mean loss: train mu explicitly (copy is prevented by masking in the training loop)
                mse = ((gt_xy - mu) ** 2).sum(dim=1).mean()

                # (A) sigma loss: NLL with mu detached so sigma gradients don't steer mu
                nll = self.gaussian_nll_vec2(mu.detach(), logs, gt_xy).mean()

                lidar_losses.append(cfg.MU_LOSS_WEIGHT * mse + cfg.SIGMA_NLL_WEIGHT * nll)

            # Radar1 vr loss (mu + sigma)
            if idxR1.numel() > 0:
                mu = out_t["r1_mu"][idxR1]
                logs = out_t["r1_log_sigma"][idxR1]
                gt_vr = x_t[idxR1][:, cfg.IDX_VR].unsqueeze(1)

                mse = ((gt_vr - mu) ** 2).mean()
                nll = self.gaussian_nll_scalar(mu.detach(), logs, gt_vr).mean()

                r1_losses.append(cfg.MU_LOSS_WEIGHT * mse + cfg.SIGMA_NLL_WEIGHT * nll)

            # Radar2 vr loss (mu + sigma)
            if idxR2.numel() > 0:
                mu = out_t["r2_mu"][idxR2]
                logs = out_t["r2_log_sigma"][idxR2]
                gt_vr = x_t[idxR2][:, cfg.IDX_VR].unsqueeze(1)

                mse = ((gt_vr - mu) ** 2).mean()
                nll = self.gaussian_nll_scalar(mu.detach(), logs, gt_vr).mean()

                r2_losses.append(cfg.MU_LOSS_WEIGHT * mse + cfg.SIGMA_NLL_WEIGHT * nll)

        loss_l = torch.stack(lidar_losses).mean() if len(lidar_losses) else torch.tensor(0.0, device=device)
        loss_r1 = torch.stack(r1_losses).mean() if len(r1_losses) else torch.tensor(0.0, device=device)
        loss_r2 = torch.stack(r2_losses).mean() if len(r2_losses) else torch.tensor(0.0, device=device)

        reg = (
            (out_t["lidar_log_sigma"] ** 2).mean()
            + (out_t["r1_log_sigma"] ** 2).mean()
            + (out_t["r2_log_sigma"] ** 2).mean()
        )

        total = loss_l + cfg.RADAR_LOSS_WEIGHT * (loss_r1 + loss_r2) + cfg.REG_LAMBDA * reg
        return {
            "total": total,
            "loss_lidar": loss_l.detach(),
            "loss_r1": loss_r1.detach(),
            "loss_r2": loss_r2.detach(),
            "reg": reg.detach(),
        }
