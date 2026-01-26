import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

def plot_results():
    # ==========================================
    # 1. Configuration
    # ==========================================
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_vers = cfg.get('inference_version', "v2")
    save_dir = "/mnt/samsung_ssd/hyunchang/inference_results"

    print(f"[Plot] Target Version: {target_vers}")
    print(f"[Plot] Reading data from: {save_dir}")

    # Normalization ranges (used in inference.py as well)
    MIN_SIG_L = float(cfg.get('min_sigma_lidar_m', 0.03))
    MAX_SIG_L = float(cfg.get('max_sigma_lidar_m', 0.2))
    MIN_SIG_R = float(cfg.get('min_sigma_radar_v', 0.05))
    MAX_SIG_R = float(cfg.get('max_sigma_radar_v', 1.5))

    def to_sigma_norm(sig, mn, mx):
        denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        return np.clip((sig - mn) / denom, 0.0, 1.0)

    # ==========================================
    # 2. Load Trajectory (Odom)
    # ==========================================
    odom_path = os.path.join(cfg['data_root'], f"odom_filtered_{target_vers}.txt")
    if os.path.exists(odom_path):
        odom_df = pd.read_csv(odom_path, sep=r'\s+', header=None,
                              names=['t', 'x', 'y', 'yaw', 'v', 'w'])
        print(f"[*] Odom Loaded: {len(odom_df)} points")
    else:
        print(f"[!] Warning: Odom file not found at {odom_path}")
        odom_df = pd.DataFrame({'x': [], 'y': []})

    # ==========================================
    # 3. Load Inference Results
    # ==========================================
    results = {}
    sensors = ['lidar', 'radar1', 'radar2']

    for sensor in sensors:
        file_path = os.path.join(save_dir, f"Result_{sensor}_{target_vers}.txt")
        if not os.path.exists(file_path):
            print(f"[!] Warning: Result file not found: {file_path}")
            results[sensor] = None
            continue

        print(f"[Plot] Loading {sensor} data...")
        df = pd.read_csv(file_path, sep=r'\s+')
        df.columns = [c.strip() for c in df.columns]

        # Ensure sigma column exists for fallback
        if sensor.startswith('radar'):
            if 'sig_vr' in df.columns and 'sigma' not in df.columns:
                df['sigma'] = df['sig_vr']  # convenience
        # lidar: new format already has 'sigma' (meters)

        # Ensure sigma_norm exists (preferred)
        if 'sigma_norm' not in df.columns:
            if 'sigma' not in df.columns:
                print(f"[!] Error: No sigma/sigma_norm column in {sensor}. Columns: {df.columns}")
                results[sensor] = None
                continue

            if sensor == 'lidar':
                df['sigma_norm'] = to_sigma_norm(df['sigma'].values, MIN_SIG_L, MAX_SIG_L)
                print("    -> sigma_norm computed from sigma (LiDAR) using yaml min/max")
            else:
                df['sigma_norm'] = to_sigma_norm(df['sigma'].values, MIN_SIG_R, MAX_SIG_R)
                print("    -> sigma_norm computed from sigma (Radar) using yaml min/max")

        results[sensor] = df
        print(f"    -> Loaded {len(df)} points | Columns: {list(df.columns)}")

    # ==========================================
    # 4. Visualization Settings
    # ==========================================
    SAMPLING_RATE = 5
    cmap = plt.get_cmap('jet')  # 0: blue, 1: red

    fig, ax = plt.subplots(figsize=(20, 15))

    # ==========================================
    # 5. Plotting
    # ==========================================
    print("[Plot] Generating Confidence Map...")

    # Trajectory
    if not odom_df.empty:
        ax.plot(
            odom_df['x'], odom_df['y'],
            c='black', linewidth=1.5, linestyle='--', alpha=0.7,
            label='Trajectory', zorder=10
        )

    def sensor_label_with_range(sensor_key):
        # Auto-legend text with normalization range
        if sensor_key == 'lidar':
            return f"LiDAR (σ_p in m, norm: [{MIN_SIG_L:.3f}, {MAX_SIG_L:.3f}])"
        elif sensor_key == 'radar1':
            return f"Radar 1 (σ_v in m/s, norm: [{MIN_SIG_R:.3f}, {MAX_SIG_R:.3f}])"
        elif sensor_key == 'radar2':
            return f"Radar 2 (σ_v in m/s, norm: [{MIN_SIG_R:.3f}, {MAX_SIG_R:.3f}])"
        return sensor_key

    def plot_sensor(key, marker, zorder, size):
        df = results.get(key, None)
        if df is None or df.empty:
            return

        df_sub = df.iloc[::SAMPLING_RATE].copy()

        # sigma_norm: 0 (small sigma) -> weight high -> RED
        # sigma_norm: 1 (large sigma) -> weight low  -> BLUE
        sigma_norm = np.clip(df_sub['sigma_norm'].values, 0.0, 1.0)
        confidence = 1.0 - sigma_norm  # 1: red, 0: blue
        colors = cmap(confidence)

        ax.scatter(
            df_sub['x'], df_sub['y'],
            c=colors,
            s=size,
            marker=marker,
            alpha=0.8,
            label=sensor_label_with_range(key),
            zorder=zorder,
            edgecolors='none'
        )

    # Radar first (bigger)
    plot_sensor('radar1', '^', 2, 15.0)
    plot_sensor('radar2', 'v', 2, 15.0)
    # LiDAR (smaller)
    plot_sensor('lidar', 'o', 3, 5.0)

    # ==========================================
    # 6. Finalize
    # ==========================================
    ax.set_title(
        f'Unified Confidence Map ({target_vers})\nRed: High Weight (Small σ, σ_norm→0) | Blue: Low Weight (Large σ, σ_norm→1)',
        fontsize=16
    )
    ax.set_xlabel('Global X (m)', fontsize=12)
    ax.set_ylabel('Global Y (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle=':')

    lgnd = ax.legend(loc='upper right', fontsize=11)
    for handle in getattr(lgnd, "legend_handles", []):
        try:
            handle._sizes = [30]
        except Exception:
            pass

    # Colorbar: confidence (1 - sigma_norm)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Weight Proxy = 1 - σ_norm', fontsize=12)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low (Blue)', 'Medium', 'High (Red)'])

    output_path = os.path.join(save_dir, f"confidence_map_{target_vers}_UnifiedScale_WithRanges.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Visualization Saved: {output_path}")

    print("[Plot] Displaying interactive plot... (Close the window to finish)")
    plt.show()

if __name__ == "__main__":
    plot_results()
