# -*- coding: utf-8 -*-

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch

from .config import ModelConfig
from .graph import GraphBuilder
from .loss import UncertaintyLoss
from .model import DOGMSTGATUncertaintyNet
from .preprocess import mask_inputs_for_mu, normalize_node_features, normalize_edges_inplace
from .backends import _HAS_TORCH_CLUSTER
import numpy as np


def train_one_epoch(
    cfg: ModelConfig,
    model: DOGMSTGATUncertaintyNet,
    graph_builder: GraphBuilder,
    loss_fn: UncertaintyLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    accum_steps: int = 1,
) -> Tuple[int, Dict[str, float]]:
    model.train()
    use_amp = bool(cfg.AMP and device.type == "cuda")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            autocast_ctx = torch.cuda.amp.autocast()
    else:
        autocast_ctx = nullcontext()
    

    # Initialize lists to accumulate losses
    epoch_total_losses: List[float] = []
    epoch_lidar_losses: List[float] = []
    epoch_r1_losses: List[float] = []
    epoch_r2_losses: List[float] = []

    for it, bt in enumerate(loader):
        (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt

        x = x.to(device); frame_id = frame_id.to(device); batch_id = batch_id.to(device); sensor_id = sensor_id.to(device)
        pose_by_frame = pose_by_frame.to(device)
        x_metric = x
        x_norm = normalize_node_features(cfg, x_metric)

        # A-style masking is applied in NORMALIZED space.
        x_in = mask_inputs_for_mu(cfg, x_norm.clone(), frame_id, sensor_id)

        # Build edges from METRIC geometry, then normalize edge attributes (dx,dy,dist) for the network.
        edges = graph_builder.build(x_metric, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)
        edges = normalize_edges_inplace(cfg, edges)

        x_raw = x_norm
        if (it % max(accum_steps, 1)) == 0:
            optimizer.zero_grad(set_to_none=True)

       
        with autocast_ctx:
            out = model(x_in, x_raw=x_raw, frame_id=frame_id, batch_id=batch_id, sensor_id=sensor_id, edges=edges, global_step=global_step)
            losses = loss_fn(out, x_raw, frame_id, batch_id, sensor_id)
            total = losses["total"]
            loss_scaled = total / float(max(accum_steps, 1))

        scaler.scale(loss_scaled).backward()

        do_step = ((it + 1) % max(accum_steps, 1) == 0) or ((it + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        if it % 20 == 0:
            print(
                f"[Train E{epoch:02d} I{it:04d}] "
                f"loss={total.item():.4f} "
                f"L={losses['loss_lidar'].item():.4f} "
                f"R1={losses['loss_r1'].item():.4f} "
                f"R2={losses['loss_r2'].item():.4f} "
                f"alpha={out['cross_alpha'].item():.3f} "
                f"knn_backend={'torch_cluster' if _HAS_TORCH_CLUSTER else 'cdist'}"
            )
            # Removed per-iteration wandb.log
        
        # Accumulate losses
        epoch_total_losses.append(total.item())
        epoch_lidar_losses.append(losses['loss_lidar'].item())
        epoch_r1_losses.append(losses['loss_r1'].item())
        epoch_r2_losses.append(losses['loss_r2'].item())

            # Calculate average losses for the epoch
            # train_eval.py 내부, train_one_epoch() 끝부분

        def mean(xs): 
            return float(np.mean(xs)) if len(xs) else 0.0

        avg_losses = {
            # === legacy keys (W&B 대시보드 호환) ===
            "train_loss": mean(epoch_total_losses),
            "train_loss_lidar": mean(epoch_lidar_losses),
            "train_loss_r1": mean(epoch_r1_losses),
            "train_loss_r2": mean(epoch_r2_losses),

            # === keep epoch-suffixed keys too (있어도 무해) ===
            "train_loss_epoch": mean(epoch_total_losses),
            "train_loss_lidar_epoch": mean(epoch_lidar_losses),
            "train_loss_r1_epoch": mean(epoch_r1_losses),
            "train_loss_r2_epoch": mean(epoch_r2_losses),
        }


    return global_step, avg_losses


@torch.no_grad()
def eval_one_epoch(
    cfg: ModelConfig,
    model: DOGMSTGATUncertaintyNet,
    graph_builder: GraphBuilder,
    loss_fn: UncertaintyLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = []
    Ls, R1s, R2s = [], [], []
    for bt in loader:
        (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt
        x = x.to(device); frame_id = frame_id.to(device); batch_id = batch_id.to(device); sensor_id = sensor_id.to(device)
        pose_by_frame = pose_by_frame.to(device)
        x_metric = x
        x_norm = normalize_node_features(cfg, x_metric)
        x_in = mask_inputs_for_mu(cfg, x_norm.clone(), frame_id, sensor_id)

        edges = graph_builder.build(x_metric, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)
        edges = normalize_edges_inplace(cfg, edges)

        x_raw = x_norm
        out = model(x_in, x_raw=x_raw, frame_id=frame_id, batch_id=batch_id, sensor_id=sensor_id, edges=edges, global_step=10**9)  # alpha=max
        losses = loss_fn(out, x_raw, frame_id, batch_id, sensor_id)

        totals.append(losses["total"].item())
        Ls.append(losses["loss_lidar"].item())
        R1s.append(losses["loss_r1"].item())
        R2s.append(losses["loss_r2"].item())

    def mean(xs): return float(np.mean(xs)) if len(xs) else 0.0
    return {"total": mean(totals), "lidar": mean(Ls), "r1": mean(R1s), "r2": mean(R2s)}


