import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

# ==========================================
# 0. Global Visualization Toggle
# ==========================================
SHOW_LIDAR = False  # Set to False if you want to plot Radar only

def plot_results():
    # ==========================================
    # 1. Configuration
    # ==========================================
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_vers = cfg.get('inference_version', "v11")
    save_dir = cfg.get('inference_save_dir', "/mnt/samsung_ssd/hyunchang/inference_results")

    # Load checkpoint info (matching inference.py logic)
    ckpt_epoch = int(cfg.get('inference_checkpoint', 0))
    model_suffix = "best" if ckpt_epoch == 0 else f"ep{ckpt_epoch}"

    print(f"[Plot] Target Version: {target_vers}")
    print(f"[Plot] Show LiDAR: {SHOW_LIDAR} | Suffix: {model_suffix}")

    MIN_SIG_R = float(cfg.get('min_sigma_radar_v', 0.10))
    MAX_SIG_R = float(cfg.get('max_sigma_radar_v', 5.0))

    # ==========================================
    # 2. Load Trajectory (Odom)
    # ==========================================
    odom_path = os.path.join(cfg['data_root'], f"odom_filtered_{target_vers}.txt")
    if os.path.exists(odom_path):
        odom_df = pd.read_csv(
            odom_path, sep=r'\s+', header=None,
            names=['t', 'x', 'y', 'yaw', 'v', 'w']
        )
        print(f"[*] Odom Loaded: {len(odom_df)} points")
    else:
        print(f"[!] Warning: Odom file not found at {odom_path}")
        odom_df = pd.DataFrame({'x': [], 'y': []})

    # ==========================================
    # 3. Load Inference Results
    # ==========================================
    results = {}
    sensors = ['radar1', 'radar2']
    if SHOW_LIDAR:
        sensors.insert(0, 'lidar')

    for sensor in sensors:
        # Use Result2 naming convention as per your latest script
        file_path = os.path.join(save_dir, f"Result2_{sensor}_{target_vers}_{model_suffix}.txt")
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: Result file not found: {file_path}")
            results[sensor] = None
            continue

        print(f"[Plot] Loading {sensor} data...")
        df = pd.read_csv(file_path, sep=r'\s+')

        # Different requirements for LiDAR and Radar
        if sensor == 'lidar':
            required = {'x', 'y', 'sigma_norm'}
        else:
            required = {'x', 'y', 'velocity', 'sigma_norm'}

        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            print(f"[Error] {file_path} is missing columns: {sorted(list(missing))}")
            results[sensor] = None
            continue

        results[sensor] = df
        print(f"    -> Loaded {len(df)} points")

    # ==========================================
    # 4. Visualization Settings
    # ==========================================
    SAMPLING_RATE = 1
    cmap = plt.get_cmap('jet')

    SIGMA_MARKER_SIZE = 25.0
    VELOCITY_MARKER_SIZE = SIGMA_MARKER_SIZE * 4.0

    fig, ax = plt.subplots(figsize=(20, 15))

    # Trajectory
    if not odom_df.empty:
        ax.plot(
            odom_df['x'], odom_df['y'],
            c='black', linewidth=2.0, linestyle='--', alpha=0.6,
            label='Trajectory', zorder=10
        )

    # Precompute global speed max for Radar
    speed_max = 3.0
    # for key in ['radar1', 'radar2']:
    #     df = results.get(key, None)
    #     if df is not None and not df.empty:
    #         df_sub = df.iloc[::SAMPLING_RATE]
    #         sp = np.abs(df_sub['velocity'].to_numpy(dtype=float))
    #         if sp.size:
    #             speed_max = max(speed_max, float(np.nanmax(sp)))

    if speed_max <= 0.0: speed_max = 1.0

    # ==========================================
    # 5. Plotting Function
    # ==========================================
    def plot_sensor_data(key: str, marker: str, zorder: int):
        df = results.get(key, None)
        if df is None or df.empty:
            return

        df_sub = df.iloc[::SAMPLING_RATE].copy()

        # (A) Velocity Plotting (Radar Only)
        if key != 'lidar':
            speed = np.abs(df_sub['velocity'].to_numpy(dtype=float))
            speed = np.clip(speed, 0.0, speed_max)
            speed_norm = speed / speed_max
            vel_colors = cmap(speed_norm)

            ax.scatter(
                df_sub['x'], df_sub['y'],
                c=vel_colors, s=VELOCITY_MARKER_SIZE,
                marker=marker, alpha=0.85, label='_nolegend_',
                zorder=zorder, edgecolors='none'
            )

        # (B) Sigma Confidence Plotting (All Sensors)
        sigma_norm = np.clip(df_sub['sigma_norm'].to_numpy(dtype=float), 0.0, 1.0)
        confidence = 1.0 - sigma_norm
        sig_colors = cmap(confidence)

        label_name = f"{key.capitalize()} (σ range in config)"
        
        ax.scatter(
            df_sub['x'], df_sub['y'],
            c=sig_colors, s=SIGMA_MARKER_SIZE,
            marker=marker, alpha=0.90, label=label_name,
            zorder=zorder + 1, edgecolors='none'
        )

    # Execute plotting
    if SHOW_LIDAR:
        plot_sensor_data('lidar', 'o', 3)
    plot_sensor_data('radar1', '^', 2)
    plot_sensor_data('radar2', 'v', 2)

    # ==========================================
    # 6. Finalize
    # ==========================================
    ax.set_title(
        f"Physics-Guided Sensor Map ({target_vers}, {model_suffix})\n"
        f"Radar Outer: Speed | Inner: Confidence | LiDAR: Confidence Only",
        fontsize=16
    )
    ax.set_xlabel('Global X (m)', fontsize=12); ax.set_ylabel('Global Y (m)', fontsize=12)
    ax.axis('equal'); ax.grid(True, alpha=0.3, linestyle=':')

    ax.legend(loc='upper right', fontsize=12)

    # Colorbars
    sm_sigma = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar_sigma = fig.colorbar(sm_sigma, ax=ax, shrink=0.6, pad=0.02)
    cbar_sigma.set_label('Sigma Confidence (1 - Norm σ)', fontsize=12)
    cbar_sigma.set_ticks([0, 0.5, 1])
    cbar_sigma.set_ticklabels(['Uncertain (Blue)', 'Medium', 'Certain (Red)'])

    sm_vel = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=speed_max))
    cbar_vel = fig.colorbar(sm_vel, ax=ax, shrink=0.6, pad=0.08)
    cbar_vel.set_label('Radar Speed |v| (m/s)', fontsize=12)

    output_path = os.path.join(save_dir, f"sensor_fusion_map_{target_vers}_{model_suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Saved Map: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_results()