# stgat/infer_export.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import MultiRunTextWindowDataset, collate_fn
from .graph import GraphBuilder
from .preprocess import mask_inputs_for_mu, normalize_node_features, normalize_edges_inplace
from .backends import _HAS_TORCH_CLUSTER


@torch.no_grad()
def infer_export(
    *,
    cfg: ModelConfig,
    model: torch.nn.Module,
    graph_builder: GraphBuilder,
    data_root: str,
    version: int,
    out_path: str,
    cache_dir: Optional[str] = None,
    device: torch.device,
    full_points: bool = False,
    max_frames: Optional[int] = None,
) -> str:
    """
    Export per-frame sigma predictions into NPZ at `out_path`.

    NPZ keys (used by visualize.py):
      - F, pose
      - lidar_out: [x, y, sigx, sigy]
      - lidar_off: frame offsets (len F+1)
      - r1_out: [x, y, vr, sigv, snr]
      - r1_off
      - r2_out: [x, y, vr, sigv, snr]
      - r2_off

    Also stores t_odom, twist, meta for completeness.
    """

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    # full_points=True면 dataset downsample cap을 비활성화(0이면 downsample 안 함: data.py의 _downsample 로직)
    old_lcap = int(getattr(cfg, "LIDAR_CAP_PER_FRAME", 720))
    old_rcap = int(getattr(cfg, "RADAR_CAP_PER_FRAME", 32))
    if full_points:
        cfg.LIDAR_CAP_PER_FRAME = 0
        cfg.RADAR_CAP_PER_FRAME = 0

    try:
        dataset = MultiRunTextWindowDataset(
            cfg=cfg,
            data_root=str(data_root),
            versions=[int(version)],
            cache_dir=cache_dir,
            max_windows_per_run=None,
            seed=42,
        )
        # 단일 버전만 넣었으므로 runs[0]이 해당 run
        run = dataset.runs[0]
        F = int(run["F"])
        pose = run["pose"].astype(np.float32)
        t_odom = run["t_odom"].astype(np.float32)
        twist = run["twist"].astype(np.float32)

        # 추론 가능한 프레임 시작 (WINDOW-1)
        f_start = max(int(cfg.WINDOW) - 1, 0)
        stop_f = F if (max_frames is None) else min(F, int(max_frames))

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        # frame별로 저장했다가 마지막에 순서대로 concat + offsets 생성
        lidar_frames = [None] * F  # each: (Ni,4)
        r1_frames = [None] * F     # each: (Ni,5)
        r2_frames = [None] * F     # each: (Ni,5)

        model.eval()

        for it, bt in enumerate(loader):
            # dataset.index[it] = (run_i, t_end)
            _, t_end = dataset.index[it]
            f = int(t_end)
            if f < f_start:
                continue
            if f >= stop_f:
                break

            (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt
            x = x.to(device)
            frame_id = frame_id.to(device)
            batch_id = batch_id.to(device)
            sensor_id = sensor_id.to(device)
            pose_by_frame = pose_by_frame.to(device)

            # 동일 파이프라인(학습과 동일):
            x_metric = x
            x_norm = normalize_node_features(cfg, x_metric)
            x_in = mask_inputs_for_mu(cfg, x_norm.clone(), frame_id, sensor_id)

            edges = graph_builder.build(x_metric, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)
            edges = normalize_edges_inplace(cfg, edges)

            out = model(
                x_in,
                x_raw=x_norm,
                frame_id=frame_id,
                batch_id=batch_id,
                sensor_id=sensor_id,
                edges=edges,
                global_step=0,
            )

            # export는 "마지막 프레임(local frame == WINDOW-1)"만 해당 t_end 프레임의 결과로 사용
            last_f = int(cfg.WINDOW) - 1
            m_last = (frame_id == last_f)

            # metric 원본 값으로 export
            x_np = x_metric.detach().cpu().numpy()
            sid_np = sensor_id.detach().cpu().numpy()

            # sigma는 log_sigma -> sigma
            lidar_sig = torch.exp(out["lidar_log_sigma"]).detach().cpu().numpy()  # (N,2)
            r1_sigv = torch.exp(out["r1_log_sigma"]).detach().cpu().numpy()      # (N,1)
            r2_sigv = torch.exp(out["r2_log_sigma"]).detach().cpu().numpy()      # (N,1)

            # LiDAR: [x,y,sigx,sigy]
            idxL = np.where((sid_np == 0) & (m_last.detach().cpu().numpy()))[0]
            if idxL.size > 0:
                xy = x_np[idxL][:, 0:2]
                sx = lidar_sig[idxL][:, 0:1]
                sy = lidar_sig[idxL][:, 1:2]
                lidar_frames[f] = np.concatenate([xy, sx, sy], axis=1).astype(np.float32)

            # Radar1: [x,y,vr,sigv,snr]
            idxR1 = np.where((sid_np == 1) & (m_last.detach().cpu().numpy()))[0]
            if idxR1.size > 0:
                xy = x_np[idxR1][:, 0:2]
                vr = x_np[idxR1][:, cfg.IDX_VR:cfg.IDX_VR + 1]
                snr = x_np[idxR1][:, cfg.IDX_SNR:cfg.IDX_SNR + 1]
                sv = r1_sigv[idxR1]
                r1_frames[f] = np.concatenate([xy, vr, sv, snr], axis=1).astype(np.float32)

            # Radar2: [x,y,vr,sigv,snr]
            idxR2 = np.where((sid_np == 2) & (m_last.detach().cpu().numpy()))[0]
            if idxR2.size > 0:
                xy = x_np[idxR2][:, 0:2]
                vr = x_np[idxR2][:, cfg.IDX_VR:cfg.IDX_VR + 1]
                snr = x_np[idxR2][:, cfg.IDX_SNR:cfg.IDX_SNR + 1]
                sv = r2_sigv[idxR2]
                r2_frames[f] = np.concatenate([xy, vr, sv, snr], axis=1).astype(np.float32)

        # offsets 생성 (len F+1)
        def build_out_and_off(frames, dim):
            off = np.zeros((F + 1,), dtype=np.int64)
            chunks = []
            for f in range(F):
                n = 0 if (frames[f] is None) else int(frames[f].shape[0])
                off[f + 1] = off[f] + n
                if n > 0:
                    chunks.append(frames[f])
            out = np.concatenate(chunks, axis=0) if len(chunks) else np.zeros((0, dim), dtype=np.float32)
            return out.astype(np.float32), off

        lidar_out, lidar_off = build_out_and_off(lidar_frames, 4)
        r1_out, r1_off = build_out_and_off(r1_frames, 5)
        r2_out, r2_off = build_out_and_off(r2_frames, 5)

        np.savez_compressed(
            str(out_path_p),
            F=int(F),
            pose=pose,
            t_odom=t_odom,
            twist=twist,
            lidar_out=lidar_out,
            lidar_off=lidar_off,
            r1_out=r1_out,
            r1_off=r1_off,
            r2_out=r2_out,
            r2_off=r2_off,
            meta=np.array([f"torch_cluster={_HAS_TORCH_CLUSTER}", f"full_points={bool(full_points)}"], dtype=object),
        )

        return str(out_path_p)

    finally:
        # cfg 복원
        cfg.LIDAR_CAP_PER_FRAME = old_lcap
        cfg.RADAR_CAP_PER_FRAME = old_rcap
