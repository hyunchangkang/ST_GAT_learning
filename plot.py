import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np
from matplotlib.lines import Line2D

# ==========================================
# USER CONTROL PARAMETERS (Time Window)
# ==========================================
START_TIME_LIMIT = 1.0  # Seconds
END_TIME_LIMIT   = 80.0  # Seconds

def plot_results():
    # ==========================================
    # 1. Configuration
    # ==========================================
    config_path = "config/params.yaml"
    if not os.path.exists(config_path):
        print(f"[!] Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_vers = cfg.get('inference_version', "v11")
    save_dir = cfg.get('inference_save_dir', "/mnt/samsung_ssd/hyunchang/inference_results")
    
    # Load checkpoint info to match file naming convention
    ckpt_epoch = int(cfg.get('inference_checkpoint', 0))
    model_suffix = "best" if ckpt_epoch == 0 else f"ep{ckpt_epoch}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"[Plot] Target Version: {target_vers}")
    print(f"[Plot] Target Suffix: {model_suffix}")
    print(f"[Plot] Time Window: {START_TIME_LIMIT}s ~ {END_TIME_LIMIT}s")

    # Physical ranges for labeling logic
    MIN_SIG_L = float(cfg.get('min_sigma_lidar_m', 0.04))
    MAX_SIG_L = float(cfg.get('max_sigma_lidar_m', 1.0))
    MIN_SIG_R = float(cfg.get('min_sigma_radar_v', 0.10))
    MAX_SIG_R = float(cfg.get('max_sigma_radar_v', 3.0))

    # ==========================================
    # 2. Load & Filter Trajectory (Odom)
    # ==========================================
    odom_path = os.path.join(cfg.get('data_root', './'), f"odom_filtered_{target_vers}.txt")
    if os.path.exists(odom_path):
        odom_df = pd.read_csv(odom_path, sep=r'\s+', header=None,
                              names=['t', 'x', 'y', 'yaw', 'v', 'w'])
        
        # [Implementation] Filtering Odom by Time Window
        odom_mask = (odom_df['t'] >= START_TIME_LIMIT) & (odom_df['t'] <= END_TIME_LIMIT)
        odom_df = odom_df[odom_mask].reset_index(drop=True)
        
        print(f"[*] Odom Loaded and Filtered: {len(odom_df)} points")
    else:
        print(f"[!] Warning: Odom file not found at {odom_path}")
        odom_df = pd.DataFrame({'x': [], 'y': [], 't': []})

    # ==========================================
    # 3. Load & Filter Inference Results
    # ==========================================
    results = {}
    sensors = ['lidar', 'radar1', 'radar2']

    for sensor in sensors:
        # Dynamically construct file path
        file_path = os.path.join(save_dir, f"Result_{sensor}_{target_vers}_{model_suffix}.txt")
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: Result file not found: {file_path}")
            results[sensor] = None
            continue

        df = pd.read_csv(file_path, sep=r'\s+')
        if 'sigma_norm' not in df.columns:
            print(f"[!] Error: 'sigma_norm' not found in {file_path}")
            results[sensor] = None
            continue

        # [Implementation] Filtering Sensor Results by Time Window
        if 't' in df.columns:
            sensor_mask = (df['t'] >= START_TIME_LIMIT) & (df['t'] <= END_TIME_LIMIT)
            df = df[sensor_mask].reset_index(drop=True)

        results[sensor] = df
        print(f"    -> Filtered {sensor}: {len(df)} points")

    # ==========================================
    # 4. Custom Legend Elements
    # ==========================================
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label='LiDAR',
               markerfacecolor='black', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='^', color='black', label='Radar1',
               markerfacecolor='black', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='v', color='black', label='Radar2',
               markerfacecolor='black', markersize=6, linestyle='None')
    ]

    SAMPLING_RATE = 2
    cmap = plt.get_cmap('jet')

    # ==========================================
    # 5. Figure 1: Confidence Map
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(15, 15))
    print("[Plot] Generating Confidence Map...")

    if not odom_df.empty:
        ax1.plot(odom_df['x'], odom_df['y'], c='black', linewidth=2.0, 
                 linestyle='--', alpha=0.6, label='Trajectory', zorder=10)

    def plot_confidence(key, marker, zorder, size):
        df = results.get(key, None)
        if df is None or df.empty: return
        df_sub = df.iloc[::SAMPLING_RATE].copy()
        
        sigma_norm = np.clip(df_sub['sigma_norm'].values, 0.0, 1.0)
        confidence = 1.0 - sigma_norm 
        colors = cmap(confidence)
        
        ax1.scatter(df_sub['x'], df_sub['y'], c=colors, s=size, marker=marker,
                    alpha=1, zorder=zorder, edgecolors='none')

    plot_confidence('lidar', 'o', 3, 10.0)
    plot_confidence('radar1', '^', 2, 30.0)
    plot_confidence('radar2', 'v', 2, 30.0)

    # Title includes the time range for clarity
    ax1.set_title(f'Uncertainty Map (Epoch: {model_suffix})\nRange: {START_TIME_LIMIT}s ~ {END_TIME_LIMIT}s', fontsize=18)
    ax1.set_xlabel('Global X (m)', fontsize=12); ax1.set_ylabel('Global Y (m)', fontsize=12)
    ax1.axis('equal'); ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig1.colorbar(sm, ax=ax1, shrink=1)
    cbar.set_label('Confidence Level', fontsize=12)

    output_path1 = os.path.join(save_dir, f"ConfidenceMap_{target_vers}_{model_suffix}_filt.png")
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')

    # ==========================================
    # 6. Figure 2: Sensor Source Map
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(20, 15))
    print("[Plot] Generating Sensor Source Map...")

    if not odom_df.empty:
        ax2.plot(odom_df['x'], odom_df['y'], c='black', linewidth=2.0, 
                 linestyle='--', alpha=0.6, label='Trajectory', zorder=10)

    sensor_colors = {'lidar': 'blue', 'radar1': 'red', 'radar2': 'green'}

    def plot_source(key, marker, zorder, size):
        df = results.get(key, None)
        if df is None or df.empty: return
        df_sub = df.iloc[::SAMPLING_RATE].copy()
        
        ax2.scatter(df_sub['x'], df_sub['y'], c=sensor_colors[key], s=size, 
                    marker=marker, alpha=1, zorder=zorder, edgecolors='none')

    # plot_source('lidar', 'o', 3, 10.0)
    # plot_source('radar1', '^', 2, 30.0)
    # plot_source('radar2', 'v', 2, 30.0)

    ax2.set_title(f'Sensor Source Map ({target_vers}, {model_suffix})\n'
                  f'Range: {START_TIME_LIMIT}s ~ {END_TIME_LIMIT}s | BLUE=LiDAR, RED=Radar1, GREEN=Radar2', fontsize=16)
    ax2.set_xlabel('Global X (m)', fontsize=12); ax2.set_ylabel('Global Y (m)', fontsize=12)
    ax2.axis('equal'); ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)

    output_path2 = os.path.join(save_dir, f"SensorSourceMap_{target_vers}_{model_suffix}_filt.png")
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')

    print(f"[*] Process Complete.\n    1. {output_path1}\n    2. {output_path2}")
    plt.show()

if __name__ == "__main__":
    plot_results()