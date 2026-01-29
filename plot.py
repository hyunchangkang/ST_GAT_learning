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

    target_vers = cfg.get('inference_version', "v11")
    save_dir = "/mnt/samsung_ssd/hyunchang/inference_results"

    print(f"[Plot] Target Version: {target_vers}")

    # Ranges for Legend Labeling
    MIN_SIG_L = float(cfg.get('min_sigma_lidar_m', 0.03))
    MAX_SIG_L = float(cfg.get('max_sigma_lidar_m', 0.2))
    
    # Radar is now Velocity Sigma (m/s)
    MIN_SIG_R = float(cfg.get('min_sigma_radar_v', 0.10))
    MAX_SIG_R = float(cfg.get('max_sigma_radar_v', 5.0))

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
        # inference.py saves: x y sigma sigma_norm
        
        # Ensure sigma_norm exists
        if 'sigma_norm' not in df.columns:
             print(f"[!] Error: 'sigma_norm' not found in {file_path}")
             results[sensor] = None
             continue

        results[sensor] = df
        print(f"    -> Loaded {len(df)} points")

    # ==========================================
    # 4. Visualization Settings
    # ==========================================
    SAMPLING_RATE = 5
    cmap = plt.get_cmap('jet')  # Blue(0) to Red(1)

    fig, ax = plt.subplots(figsize=(20, 15))

    # ==========================================
    # 5. Plotting
    # ==========================================
    print("[Plot] Generating Physics-Guided Confidence Map...")

    # Trajectory
    if not odom_df.empty:
        ax.plot(
            odom_df['x'], odom_df['y'],
            c='black', linewidth=2.0, linestyle='--', alpha=0.6,
            label='Trajectory', zorder=10
        )

    def sensor_label(sensor_key):
        if sensor_key == 'lidar':
            return f"LiDAR (σ_pos [m], range: {MIN_SIG_L}~{MAX_SIG_L})"
        elif 'radar' in sensor_key:
            return f"{sensor_key.capitalize()} (σ_vel [m/s], range: {MIN_SIG_R}~{MAX_SIG_R})"
        return sensor_key

    def plot_sensor(key, marker, zorder, size):
        df = results.get(key, None)
        if df is None or df.empty:
            return

        df_sub = df.iloc[::SAMPLING_RATE].copy()

        # [Color Logic]
        # sigma_norm: 0.0 (Min Sigma, Confident) -> Red
        # sigma_norm: 1.0 (Max Sigma, Uncertain) -> Blue
        # Jet colormap: 0=Blue, 1=Red
        
        # We want: High Sigma (1.0) -> Blue (0.0 in cmap)
        #          Low Sigma (0.0) -> Red (1.0 in cmap)
        # So: confidence = 1.0 - sigma_norm
        
        sigma_norm = np.clip(df_sub['sigma_norm'].values, 0.0, 1.0)
        confidence = 1.0 - sigma_norm 
        colors = cmap(confidence)

        ax.scatter(
            df_sub['x'], df_sub['y'],
            c=colors,
            s=size,
            marker=marker,
            alpha=0.8,
            label=sensor_label(key),
            zorder=zorder,
            edgecolors='none'
        )

    plot_sensor('lidar', 'o', 3, 5.0)
    plot_sensor('radar1', '^', 2, 30.0)
    plot_sensor('radar2', 'v', 2, 30.0)

    # ==========================================
    # 6. Finalize
    # ==========================================
    ax.set_title(
        f'Physics-Guided Uncertainty Map ({target_vers})\n'
        f'RED = High Confidence (Low σ) | BLUE = Low Confidence (High σ / Ghost / Lateral Move)',
        fontsize=16
    )
    ax.set_xlabel('Global X (m)', fontsize=12)
    ax.set_ylabel('Global Y (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle=':')

    lgnd = ax.legend(loc='upper right', fontsize=12)
    for handle in getattr(lgnd, "legend_handles", []):
        try:
            handle._sizes = [40]
        except Exception:
            pass

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Confidence Level (1 - Normalized σ)', fontsize=12)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Uncertain (Blue)', 'Medium', 'Certain (Red)'])

    output_path = os.path.join(save_dir, f"ConfidenceMap_{target_vers}_Physics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Saved Map: {output_path}")
    
    plt.show() # 서버 환경일 경우 주석 처리 필요할 수 있음

if __name__ == "__main__":
    plot_results()