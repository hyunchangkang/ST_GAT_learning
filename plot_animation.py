import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import os
import numpy as np
from matplotlib.colors import Normalize

# ==========================================
# [사용자 설정] 퍼포먼스 튜닝
# ==========================================
CUSTOM_X_LIM = None
CUSTOM_Y_LIM = [-10, 10]

# [최적화] 데이터 솎아내기 (Downsampling)
# 1이면 모든 점 표시, 3이면 3개 중 1개만 표시 (데이터 1/3로 감소 -> 속도 3배 향상)
POINT_STRIDE = 3 

# [최적화] 재생 속도 (ms)
# 10~20ms 추천 (Blit 켜면 이 속도 감당 가능)
ANIMATION_INTERVAL = 20 

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
def load_data():
    config_path = "config/params.yaml"
    if not os.path.exists(config_path):
        print(f"[!] Error: Config file not found at {config_path}")
        return None, None, None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_vers = cfg.get('inference_version', "v11")
    save_dir = cfg.get('inference_save_dir', "/mnt/samsung_ssd/hyunchang/inference_results")
    
    ckpt_epoch = int(cfg.get('inference_checkpoint', 0))
    model_suffix = "best" if ckpt_epoch == 0 else f"ep{ckpt_epoch}"

    print(f"[Plot] Target Version: {target_vers}")

    # 1) Odom 로드
    odom_path = os.path.join(cfg.get('data_root', './'), f"odom_filtered_{target_vers}.txt")
    if os.path.exists(odom_path):
        odom_df = pd.read_csv(odom_path, sep=r'\s+', header=None,
                              names=['t', 'x', 'y', 'yaw', 'v', 'w'])
        odom_df = odom_df.sort_values('t').reset_index(drop=True)
        print(f"[*] Odom Loaded: {len(odom_df)} points")
    else:
        print(f"[!] Critical: Odom file not found at {odom_path}")
        return None, None, None

    # 2) Inference 결과 로드
    results = {}
    sensors = ['lidar', 'radar1', 'radar2']

    for sensor in sensors:
        file_path = os.path.join(save_dir, f"Result_{sensor}_{target_vers}_{model_suffix}.txt")
        
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, sep=r'\s+')
        if 't' not in df.columns:
            continue

        df['confidence'] = 1.0 - np.clip(df['sigma_norm'], 0.0, 1.0)
        results[sensor] = df
        print(f"    -> Loaded {sensor}: {len(df)} points")
        
    return odom_df, results, target_vers

# ==========================================
# 2. Animation Logic (Optimized)
# ==========================================
def run_animation():
    odom_df, results, target_vers = load_data()
    
    if odom_df is None or not results: 
        print("[Error] Data loading failed.")
        return

    main_sensor = 'lidar' if 'lidar' in results else list(results.keys())[0]
    unique_timestamps = sorted(results[main_sensor]['t'].unique())
    
    # Figure 설정
    fig, ax = plt.subplots(figsize=(12, 12))
    # 제목 업데이트를 위해 text 객체 사용 (set_title은 blit에서 느림)
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Global X (m)', fontsize=12)
    ax.set_ylabel('Global Y (m)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('auto')

    # Plot 요소
    traj_line, = ax.plot([], [], 'k--', linewidth=2.0, alpha=0.6, label='Trajectory')
    robot_point, = ax.plot([], [], 'ko', markersize=10, label='Robot', zorder=20)
    
    scatters = {}
    cmap = plt.get_cmap('jet') 
    
    if 'lidar' in results:
        scatters['lidar'] = ax.scatter([], [], c=[], cmap=cmap, s=10, marker='o', 
                                       alpha=1.0, edgecolors='none', label='LiDAR', vmin=0, vmax=1)
    if 'radar1' in results:
        scatters['radar1'] = ax.scatter([], [], c=[], cmap=cmap, s=30, marker='^', 
                                        alpha=1.0, edgecolors='none', label='Radar1', vmin=0, vmax=1)
    if 'radar2' in results:
        scatters['radar2'] = ax.scatter([], [], c=[], cmap=cmap, s=30, marker='v', 
                                        alpha=1.0, edgecolors='none', label='Radar2', vmin=0, vmax=1)
    
    ax.legend(loc='upper right', fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Confidence Level (1.0 - Sigma)', fontsize=12)

    # 축 범위 설정
    margin = 5.0
    if CUSTOM_X_LIM:
        ax.set_xlim(CUSTOM_X_LIM)
    else:
        x_min, x_max = odom_df['x'].min(), odom_df['x'].max()
        for df in results.values():
            x_min = min(x_min, df['x'].min())
            x_max = max(x_max, df['x'].max())
        ax.set_xlim(x_min - margin, x_max + margin)

    if CUSTOM_Y_LIM:
        ax.set_ylim(CUSTOM_Y_LIM)
    else:
        y_min, y_max = odom_df['y'].min(), odom_df['y'].max()
        for df in results.values():
            y_min = min(y_min, df['y'].min())
            y_max = max(y_max, df['y'].max())
        ax.set_ylim(y_min - margin, y_max + margin)

    # 데이터 준비
    sensor_data_groups = {k: df for k, df in results.items()}
    history = {k: {'x': [], 'y': [], 'c': []} for k in scatters.keys()}
    
    odom_times = odom_df['t'].values
    odom_x = odom_df['x'].values
    odom_y = odom_df['y'].values

    def init():
        traj_line.set_data([], [])
        robot_point.set_data([], [])
        title_text.set_text(f'Uncertainty Map Animation ({target_vers})')
        for sc in scatters.values():
            sc.set_offsets(np.empty((0, 2)))
            sc.set_array(np.array([]))
        return [traj_line, robot_point, title_text] + list(scatters.values())

    def update(frame_idx):
        curr_t = unique_timestamps[frame_idx]
        
        # Odom 동기화
        odom_idx = np.searchsorted(odom_times, curr_t, side='right') - 1
        if odom_idx < 0: odom_idx = 0
        
        traj_line.set_data(odom_x[:odom_idx+1], odom_y[:odom_idx+1])
        robot_point.set_data([odom_x[odom_idx]], [odom_y[odom_idx]])
        
        # 센서 데이터 업데이트
        for key, df in sensor_data_groups.items():
            batch = df[np.isclose(df['t'], curr_t, atol=1e-2)]
            
            if not batch.empty:
                # [최적화] Stride 적용: 데이터 솎아내기 (::POINT_STRIDE)
                x_vals = batch['x'].values[::POINT_STRIDE]
                y_vals = batch['y'].values[::POINT_STRIDE]
                c_vals = batch['confidence'].values[::POINT_STRIDE]

                history[key]['x'].extend(x_vals)
                history[key]['y'].extend(y_vals)
                history[key]['c'].extend(c_vals)

            if len(history[key]['x']) > 0:
                pts = np.column_stack((history[key]['x'], history[key]['y']))
                scatters[key].set_offsets(pts)
                scatters[key].set_array(np.array(history[key]['c']))

        title_text.set_text(f'Time: {curr_t:.2f}s | Points: {sum(len(v["x"]) for v in history.values()):,}')
        
        # Blit을 위해 변경된 아티스트 리스트 반환
        return [traj_line, robot_point, title_text] + list(scatters.values())

    # [최적화 핵심] blit=True 사용
    ani = animation.FuncAnimation(fig, update, frames=len(unique_timestamps),
                                  init_func=init, blit=True, 
                                  interval=ANIMATION_INTERVAL, repeat=False)

    print(f"[Info] Optimized Animation Started (Stride={POINT_STRIDE}, Blit=True).")
    plt.show()

if __name__ == "__main__":
    run_animation()