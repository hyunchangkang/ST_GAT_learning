import torch
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch
from scipy.spatial import cKDTree

from src.dataset import get_selected_dataset
from src.model import ST_HGAT
from src.utils import build_graph

def custom_collate(batch):
    return Batch.from_data_list([item[0] for item in batch]), [item[1] for item in batch], [item[2] for item in batch], [item[3] for item in batch]

def inference():
    # 1. Config & Device Setup
    config_path = "config/params.yaml"
    with open(config_path, "r") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Device: {device}")

    # 2. Dataset Setup
    target_vers = cfg.get('inference_version', "v2")
    dataset = get_selected_dataset(cfg['data_root'], [target_vers], cfg['window_size'])
    inner_dataset = dataset.datasets[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # 3. Model Initialization (Fixing TypeError)
    node_types = ['lidar', 'radar1', 'radar2']
    edge_types = [
        ('lidar', 'spatial', 'lidar'), ('lidar', 'temporal', 'lidar'),
        ('radar1', 'spatial', 'radar1'), ('radar1', 'temporal', 'radar1'),
        ('radar2', 'spatial', 'radar2'), ('radar2', 'temporal', 'radar2'),
        ('radar1', 'to', 'lidar'), ('lidar', 'to', 'radar1'),
        ('radar2', 'to', 'lidar'), ('lidar', 'to', 'radar2')
    ]
    metadata = (node_types, edge_types)
    
    # Use dummy batch for dimension check
    dummy_batch = dataset[0][0]
    node_in_dims = {nt: dummy_batch[nt].x.size(1) for nt in node_types}

    model = ST_HGAT(
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers'], 
        heads=cfg['num_heads'], 
        metadata=metadata,
        node_in_dims=node_in_dims
    ).to(device)
    
    model_path = os.path.join(cfg['save_dir'], "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Data Loading (Raw BaseScan & Odom for Global Transformation)
    print(f"[Inference] Loading Raw Data for Version: {target_vers}")
    raw_dfs = {
        'lidar': pd.read_csv(os.path.join(cfg['data_root'], "Basescan", f"LiDARMap_BaseScan_{target_vers}.txt"), sep='\s+', header=None),
        'radar1': pd.read_csv(os.path.join(cfg['data_root'], "Basescan", f"Radar1Map_BaseScan_{target_vers}.txt"), sep='\s+', header=None),
        'radar2': pd.read_csv(os.path.join(cfg['data_root'], "Basescan", f"Radar2Map_BaseScan_{target_vers}.txt"), sep='\s+', header=None)
    }
    odom_df = pd.read_csv(os.path.join(cfg['data_root'], f"odom_filtered_{target_vers}.txt"), sep='\s+', header=None, names=['t', 'x', 'y', 'yaw', 'v', 'w'])

    save_buffers = {'lidar': [], 'radar1': [], 'radar2': []}
    vis_results = {'lidar': [], 'radar1': [], 'radar2': []}
    timestamps = inner_dataset.ts
    scale_pos = 10.0 # Scaling factor used during training

    print(f"[Inference] Processing Frames & Global Mapping...")
    with torch.no_grad():
        # Iterate over loader to handle windowed sequences
        for idx in tqdm(range(len(loader) - 1)):
            # Load specific batch data
            batch_data, _, _, _ = dataset[idx]
            batch_data = batch_data.to(device)
            
            base_t = timestamps[inner_dataset.indices[idx]] # Robot Pose at t=0
            target_ts_plus_1 = timestamps[inner_dataset.indices[idx] + 1] # Target Frame at t+1
        
            # Build Graph and Run Model
            edge_index_dict = build_graph(
                batch_data, cfg['radius_ll'], cfg['radius_rr'], cfg['radius_cross'], 
                cfg.get('temporal_radius_ll', 0.15), cfg.get('temporal_radius_rr', 0.3), device
            )
            _, sig_l, _, sig_r1, _, sig_r2 = model(batch_data.x_dict, edge_index_dict)

            # Global and Inverse Local Transformation Matrices
            T_target = inner_dataset.get_T(target_ts_plus_1) # Pose at t+1
            T_inv_base = np.linalg.inv(inner_dataset.get_T(base_t)) # Inverse Pose at t

            def process(sig, key):
                if sig.size(0) == 0: return
                
                # Filter model outputs for the current time step (dt=0)
                mask = (batch_data[key].x[:, -1] == 0).cpu().numpy()
                if not np.any(mask): return # Prevent IndexError for empty nodes (Frame 174)
                
                sig_f = sig.cpu().numpy()[mask]
                tree = cKDTree(batch_data[key].pos[mask].cpu().numpy()) 

                # Load raw global points for target t+1
                raw_frame = raw_dfs[key][np.round(raw_dfs[key][0], 1) == np.round(target_ts_plus_1, 1)].copy()
                if raw_frame.empty: return
                
                xy_local_raw = raw_frame[[1, 2]].values # Points in local sensor frame at t+1
                xy_h = np.hstack([xy_local_raw, np.ones((len(xy_local_raw), 1))])
                
                # 1. Global Transformation (Mapping)
                xy_global = (T_target @ xy_h.T).T[:, :2]
                
                # 2. Inverse Matching Transformation (Aligning t+1 points to t=0 robot frame)
                xy_local_match = (T_inv_base @ (T_target @ xy_h.T)).T[:, :2]
                
                # 3. Spatial Matching & Uncertainty Recovery
                _, indices = tree.query(xy_local_match / scale_pos, k=1)
                mapped_sig = sig_f[indices] * scale_pos # Scale back to meters (m)
                
                # 4. Store Results
                raw_frame['sigma'] = mapped_sig
                save_buffers[key].append(raw_frame)
                vis_results[key].append({'pos': xy_global, 'sigma': mapped_sig})

            process(sig_l, 'lidar'); process(sig_r1, 'radar1'); process(sig_r2, 'radar2')

    # 5. Save Results to SSD Path
    save_dir = "/mnt/samsung_ssd/hyunchang/inference_results"
    if not os.makedirs(save_dir, exist_ok=True): pass
    
    for key, buffer in save_buffers.items():
        if buffer:
            final_df = pd.concat(buffer)
            save_path = os.path.join(save_dir, f"Predict_Sigma_{key}_{target_vers}.txt")
            final_df.to_csv(save_path, sep=' ', index=False, header=False)
            print(f"[*] Results saved: {save_path}")

    # 6. Global Confidence Map Visualization (Red=Certain, Blue=Uncertain)
    print(f"[Inference] Generating Global Map...")
    fig, ax = plt.subplots(figsize=(20, 15))
    cmap = plt.get_cmap('jet_r') # 0.0 (Red, High Confidence) to 1.0 (Blue, Low Confidence)
    
    LIMIT_LIDAR = 0.5 # Uncertainty Threshold for LiDAR (m)
    LIMIT_RADAR = 2.0 # Uncertainty Threshold for Radar (m)

    # Plot Robot Trajectory
    ax.plot(odom_df['x'], odom_df['y'], c='black', linewidth=1.5, alpha=0.5, label='Robot Trajectory', zorder=1)
    
    def plot_sensor(key, marker, label, zorder, limit_val):
        if not vis_results[key]: return
        pos = np.vstack([d['pos'] for d in vis_results[key]])
        sig = np.concatenate([d['sigma'] for d in vis_results[key]])
        
        # Color mapping: sig -> 0.0 (Red) is Certain, 1.0 (Blue) is Uncertain
        norm_sig = np.clip(sig / limit_val, 0, 1)
        colors = cmap(norm_sig) 
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=1.2, marker=marker, alpha=0.8, label=label, zorder=zorder)

    # Order layers for visualization (Radar first, then LiDAR)
    plot_sensor('radar1', '^', 'Radar 1', 2, LIMIT_RADAR)
    plot_sensor('radar2', 'v', 'Radar 2', 2, LIMIT_RADAR)
    plot_sensor('lidar', '.', 'LiDAR', 3, LIMIT_LIDAR) 
    
    ax.set_title(f'Global Confidence Map (Sequence: {target_vers}) | Red: High Confidence', fontsize=18)
    ax.set_xlabel('Global X (m)', fontsize=14); ax.set_ylabel('Global Y (m)', fontsize=14)
    ax.axis('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', markerscale=5)
    
    # Add Colorbar for uncertainty reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Confidence Score (Red: Confident, Blue: Unconfident)')
    
    plot_save_path = os.path.join(save_dir, f"confidence_map_{target_vers}.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Global map visualization saved: {plot_save_path}")

if __name__ == "__main__":
    inference()