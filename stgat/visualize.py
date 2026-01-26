# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl


def transform_to_global(pts_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    """SE(2) base->world transform. pts_xy: (N,2) in base frame."""
    x, y, yaw = float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (pts_xy @ R.T) + np.array([x, y], dtype=np.float32)


def slice_frame(arr: np.ndarray, off: np.ndarray, f: int) -> np.ndarray:
    s = int(off[f]); e = int(off[f + 1])
    return arr[s:e] if e > s else arr[0:0]


def robust_range(values: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> tuple[float, float]:
    """Robust vmin/vmax via percentiles (prevents outliers from washing out colormap)."""
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


def _resolve_config_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    for c in ["params.yaml", "params_tw.yaml"]:
        if Path(c).exists():
            return c
    return "params_tw.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML path (default: params.yaml or params_tw.yaml)")
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    version = str(cfg.get("inference_version", "v11"))
    out_root = Path(str(cfg.get("inference_save_dir", "output/inference_results")))
    run_dir = out_root / str(version)

    npz_path = run_dir / "sigma_export.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}. Run: python infer.py")

    # Optional YAML overrides
    sampling_rate = int(cfg.get("viz_sampling_rate", 10))
    near_vehicle_dist = float(cfg.get("viz_near_vehicle_dist", 40.0))
    window_size = int(cfg.get("window_size", 4))

    # Load NPZ
    z = np.load(str(npz_path))
    F = int(z["F"])
    pose = z["pose"].astype(np.float32)

    lidar_out = z["lidar_out"].astype(np.float32)  # [x,y,sigx,sigy]
    lidar_off = z["lidar_off"].astype(np.int64)
    r1_out = z["r1_out"].astype(np.float32)        # [x,y,vr,sigv,snr]
    r1_off = z["r1_off"].astype(np.int64)
    r2_out = z["r2_out"].astype(np.float32)
    r2_off = z["r2_off"].astype(np.int64)

    # Frame range
    f_start = max(window_size - 1, 0)

    # ------------------------------------------------------------------
    # Pass 1: collect sigma values for robust scaling
    # ------------------------------------------------------------------
    lidar_sig_collect = []
    radar_sig_collect = []

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
                lidar_sig_collect.append(np.sqrt(L[keep, 2] ** 2 + L[keep, 3] ** 2))

        R1 = slice_frame(r1_out, r1_off, fidx)
        if R1.size > 0:
            xy_w = transform_to_global(R1[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                radar_sig_collect.append(R1[keep, 3])

        R2 = slice_frame(r2_out, r2_off, fidx)
        if R2.size > 0:
            xy_w = transform_to_global(R2[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                radar_sig_collect.append(R2[keep, 3])

    lidar_sig_all = np.concatenate(lidar_sig_collect, axis=0) if lidar_sig_collect else np.zeros((0,), dtype=np.float32)
    radar_sig_all = np.concatenate(radar_sig_collect, axis=0) if radar_sig_collect else np.zeros((0,), dtype=np.float32)

    lidar_vmin, lidar_vmax = robust_range(lidar_sig_all, 5.0, 95.0)
    radar_vmin, radar_vmax = robust_range(radar_sig_all, 5.0, 95.0)

    # Colormaps (LiDAR: red->yellow, Radar: green->blue->purple)
    cmap_lidar = plt.get_cmap("autumn")
    cmap_radar = mpl.colors.LinearSegmentedColormap.from_list(
        "GnBuPu_custom", ["green", "blue", "purple"]
    )

    # IMPORTANT: clip=False to avoid saturation turning everything into the same end color
    norm_lidar = mpl.colors.Normalize(vmin=lidar_vmin, vmax=lidar_vmax, clip=False)
    norm_radar = mpl.colors.Normalize(vmin=radar_vmin, vmax=radar_vmax, clip=False)

    # ------------------------------------------------------------------
    # Pass 2: plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    if pose.size > 0:
        ax.plot(pose[:, 0], pose[:, 1], color="white", linewidth=1.0, alpha=0.9, label="trajectory")

    did_L = did_R1 = did_R2 = False

    for fidx in range(f_start, F):
        if sampling_rate > 1 and (fidx % sampling_rate != 0):
            continue

        cur_pose = pose[fidx]
        ego_xy = cur_pose[:2]

        # LiDAR
        L = slice_frame(lidar_out, lidar_off, fidx)
        if L.size > 0:
            xy_w = transform_to_global(L[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = np.sqrt(L[keep, 2] ** 2 + L[keep, 3] ** 2)
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_lidar, norm=norm_lidar,
                    s=3, alpha=0.9,
                    label=("LiDAR" if not did_L else None)
                )
                did_L = True

        # Radar1
        R1 = slice_frame(r1_out, r1_off, fidx)
        if R1.size > 0:
            xy_w = transform_to_global(R1[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = R1[keep, 3]
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_radar, norm=norm_radar,
                    s=10, alpha=0.95,
                    label=("Radar1" if not did_R1 else None)
                )
                did_R1 = True

        # Radar2
        R2 = slice_frame(r2_out, r2_off, fidx)
        if R2.size > 0:
            xy_w = transform_to_global(R2[:, 0:2], cur_pose)
            d = np.linalg.norm(xy_w - ego_xy[None, :], axis=1)
            keep = d <= near_vehicle_dist
            if np.any(keep):
                sig = R2[keep, 3]
                ax.scatter(
                    xy_w[keep, 0], xy_w[keep, 1],
                    c=sig, cmap=cmap_radar, norm=norm_radar,
                    s=10, alpha=0.95,
                    label=("Radar2" if not did_R2 else None)
                )
                did_R2 = True

    ax.set_title(f"GAT sigma on actual points (NPZ): {version}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True, color="gray", alpha=0.25)

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
