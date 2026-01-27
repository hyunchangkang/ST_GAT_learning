# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import yaml

import matplotlib as mpl
import matplotlib.pyplot as plt


def robust_range(values: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> Tuple[float, float]:
    """Robust min/max for colormap scaling."""
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(values, p_lo))
    hi = float(np.percentile(values, p_hi))
    if not np.isfinite(lo):
        lo = float(np.min(values))
    if not np.isfinite(hi):
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def resolve_config_path(user_path: Optional[str]) -> str:
    if user_path:
        return user_path
    for c in ["params.yaml", "params_tw.yaml"]:
        if Path(c).exists():
            return c
    return "params_tw.yaml"


def slice_frame(arr: np.ndarray, off: np.ndarray, f: int) -> np.ndarray:
    s = int(off[f])
    e = int(off[f + 1])
    if e <= s:
        return arr[0:0]
    return arr[s:e]


def transform_to_global(xy_local: np.ndarray, pose_xy_yaw: np.ndarray) -> np.ndarray:
    """
    pose_xy_yaw: [x, y, yaw]
    xy_local: (N,2) in ego/local frame
    """
    x0 = float(pose_xy_yaw[0])
    y0 = float(pose_xy_yaw[1])
    yaw = float(pose_xy_yaw[2]) if pose_xy_yaw.shape[0] >= 3 else 0.0

    c = np.cos(yaw)
    s = np.sin(yaw)
    X = x0 + c * xy_local[:, 0] - s * xy_local[:, 1]
    Y = y0 + s * xy_local[:, 0] + c * xy_local[:, 1]
    return np.stack([X, Y], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML path (default: params.yaml or params_tw.yaml)")
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config)
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    version = str(cfg.get("inference_version", "v11"))
    out_root = Path(str(cfg.get("inference_save_dir", "output/inference_results")))
    run_dir = out_root / str(version)

    npz_path = run_dir / "sigma_export.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}. Run inference first.")

    # 필터 기본값: “모든 프레임/모든 포인트”가 목적이므로 기본 OFF
    sampling_rate = int(cfg.get("viz_sampling_rate", 1))          # 1이면 모든 프레임
    near_vehicle_dist = float(cfg.get("viz_near_vehicle_dist", 1e18))  # 사실상 무한대
    window_size = int(cfg.get("window_size", 4))

    z = np.load(str(npz_path))
    F = int(z["F"])
    pose = z["pose"].astype(np.float32)

    lidar_out = z["lidar_out"].astype(np.float32)  # [x,y,sigx,sigy]
    lidar_off = z["lidar_off"].astype(np.int64)
    r1_out = z["r1_out"].astype(np.float32)        # [x,y,vr,sigv,snr]
    r1_off = z["r1_off"].astype(np.int64)
    r2_out = z["r2_out"].astype(np.float32)
    r2_off = z["r2_off"].astype(np.int64)

    # NOTE: NPZ는 일반적으로 WINDOW-1 이후 프레임만 채워지고, 그 이전 프레임은 off가 0으로 유지됩니다.
    #       (WINDOW 이전 프레임까지 “추론 결과”를 만들려면 padding 규칙 정의가 필요하며, 그건 로직 변경입니다.)

    # ------------------------------------------------------------------
    # Pass 1: collect sigma for scaling (필터 적용 전/후 일관되게)
    # ------------------------------------------------------------------
    lidar_sig_all: List[np.ndarray] = []
    radar_sig_all: List[np.ndarray] = []

    f_start = 0  # “모든 프레임” 표시 요구 → 0부터 루프 (비어있는 프레임은 자연히 안 찍힘)

    for fidx in range(f_start, F):
        if sampling_rate > 1 and (fidx % sampling_rate != 0):
            continue

        cur_pose = pose[fidx]
        ego_xy = cur_pose[:2]

        L = slice_frame(lidar_out, lidar_off, fidx)
        if L.size > 0:
            xy_w = transform_to_global(L[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = np.sqrt(L[keep, 2] ** 2 + L[keep, 3] ** 2).astype(np.float32)
                lidar_sig_all.append(sig)

        R1 = slice_frame(r1_out, r1_off, fidx)
        if R1.size > 0:
            xy_w = transform_to_global(R1[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                radar_sig_all.append(R1[keep, 3].astype(np.float32))

        R2 = slice_frame(r2_out, r2_off, fidx)
        if R2.size > 0:
            xy_w = transform_to_global(R2[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                radar_sig_all.append(R2[keep, 3].astype(np.float32))

    lidar_sig_all = np.concatenate(lidar_sig_all, axis=0) if len(lidar_sig_all) else np.zeros((0,), dtype=np.float32)
    radar_sig_all = np.concatenate(radar_sig_all, axis=0) if len(radar_sig_all) else np.zeros((0,), dtype=np.float32)

    lidar_vmin, lidar_vmax = robust_range(lidar_sig_all, 5.0, 95.0)
    radar_vmin, radar_vmax = robust_range(radar_sig_all, 5.0, 95.0)

    # 빨-주-노, 초-파-보 그라데이션 colormap (matplotlib 내장 의존 최소화)
    cmap_lidar = mpl.colors.LinearSegmentedColormap.from_list("lidar_roy", ["red", "orange", "yellow"])
    cmap_radar = mpl.colors.LinearSegmentedColormap.from_list("radar_gbp", ["green", "blue", "purple"])

    norm_lidar = mpl.colors.Normalize(vmin=lidar_vmin, vmax=lidar_vmax, clip=True)
    norm_radar = mpl.colors.Normalize(vmin=radar_vmin, vmax=radar_vmax, clip=True)

    # ------------------------------------------------------------------
    # Pass 2: plot (컬러바 2개를 오른쪽에 배치)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 10), facecolor="black")
    gs = fig.add_gridspec(1, 3, width_ratios=[30, 1.2, 1.2], wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    cax_l = fig.add_subplot(gs[0, 1])
    cax_r = fig.add_subplot(gs[0, 2])

    ax.set_facecolor("black")
    for a in (cax_l, cax_r):
        a.set_facecolor("black")

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    if pose.size > 0:
        ax.plot(pose[:, 0], pose[:, 1], color="white", linewidth=1.0, alpha=0.9, label="trajectory")

    # 포인트 크기: 기존 대비 1/2
    s_lidar = 1.5   # 기존 3 → 1.5
    s_radar = 5.0   # 기존 10 → 5

    did_L = did_R1 = did_R2 = False

    for fidx in range(f_start, F):
        if sampling_rate > 1 and (fidx % sampling_rate != 0):
            continue

        cur_pose = pose[fidx]
        ego_xy = cur_pose[:2]

        L = slice_frame(lidar_out, lidar_off, fidx)
        if L.size > 0:
            xy_w = transform_to_global(L[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = np.sqrt(L[keep, 2] ** 2 + L[keep, 3] ** 2).astype(np.float32).ravel()
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_lidar, norm=norm_lidar,
                    s=s_lidar, alpha=0.95,
                    linewidths=0.0,
                    label=("LiDAR" if not did_L else None),
                )
                did_L = True

        R1 = slice_frame(r1_out, r1_off, fidx)
        if R1.size > 0:
            xy_w = transform_to_global(R1[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = R1[keep, 3].astype(np.float32).ravel()
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_radar, norm=norm_radar,
                    s=s_radar, alpha=0.95,
                    linewidths=0.0,
                    label=("Radar1" if not did_R1 else None),
                )
                did_R1 = True

        R2 = slice_frame(r2_out, r2_off, fidx)
        if R2.size > 0:
            xy_w = transform_to_global(R2[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = R2[keep, 3].astype(np.float32).ravel()
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_radar, norm=norm_radar,
                    s=s_radar, alpha=0.95,
                    linewidths=0.0,
                    label=("Radar2" if not did_R2 else None),
                )
                did_R2 = True

    ax.set_title(f"GAT sigma on actual points (NPZ): {version}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True, color="gray", alpha=0.25)

    # 컬러바(오른쪽 2개)
    sm_l = mpl.cm.ScalarMappable(norm=norm_lidar, cmap=cmap_lidar)
    sm_l.set_array([])
    cb_l = fig.colorbar(sm_l, cax=cax_l)
    cb_l.set_label("LiDAR σ (sqrt(sigx^2+sigy^2))", color="white")
    cb_l.ax.tick_params(colors="white")
    for spine in cb_l.ax.spines.values():
        spine.set_edgecolor("white")

    sm_r = mpl.cm.ScalarMappable(norm=norm_radar, cmap=cmap_radar)
    sm_r.set_array([])
    cb_r = fig.colorbar(sm_r, cax=cax_r)
    cb_r.set_label("Radar σ_v", color="white")
    cb_r.ax.tick_params(colors="white")
    for spine in cb_r.ax.spines.values():
        spine.set_edgecolor("white")

    leg = ax.legend(loc="upper right", fontsize=9, facecolor="black", framealpha=0.6)
    for t in leg.get_texts():
        t.set_color("white")

    out_png = run_dir / "gat_visualization.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[OK] saved: {out_png}")
    print(f"[sigma scale] LiDAR vmin/vmax = {lidar_vmin:.6f}/{lidar_vmax:.6f} | Radar vmin/vmax = {radar_vmin:.6f}/{radar_vmax:.6f}")


if __name__ == "__main__":
    main()
