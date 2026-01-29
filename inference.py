import torch
import yaml
import os
import math
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch

# Custom modules
from src.dataset import get_selected_dataset
from src.model import ST_HGAT

def custom_collate(batch):
    return Batch.from_data_list(batch)

def inference():
    # ==========================================
    # 1. Configuration
    # ==========================================
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[Inference] Device: {device}")

    # ==========================================
    # 2. Dataset Setup
    # ==========================================
    target_vers = cfg.get('inference_version', "v11")
    print(f"[Inference] Target Version: {target_vers}")

    dataset = get_selected_dataset(cfg['data_root'], [target_vers], cfg['window_size'])
    inner_dataset = dataset.datasets[0]
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Scaling Factors (Train과 동일)
    SCALE_POSE = 10.0    
    SCALE_RADAR_V = 5.0 

    # [수정] 박사님 요청대로 YAML(params.yaml)의 값을 그대로 사용하도록 복구합니다.
    # 1. LiDAR Limits (m)
    min_l = float(cfg.get('min_sigma_lidar_m', 0.03))
    max_l = float(cfg.get('max_sigma_lidar_m', 0.2))
    L_MIN = 2 * math.log(min_l / SCALE_POSE + 1e-9)
    L_MAX = 2 * math.log(max_l / SCALE_POSE + 1e-9)

    # 2. Radar Limits (m/s)
    min_r = float(cfg.get('min_sigma_radar_v', 0.10))
    max_r = float(cfg.get('max_sigma_radar_v', 5.0)) # 원래 코드의 기본값 5.0 복구
    R_MIN = 2 * math.log(min_r / SCALE_RADAR_V + 1e-9)
    R_MAX = 2 * math.log(max_r / SCALE_RADAR_V + 1e-9)

    print(f"[Inference] Limits -> LiDAR(m): [{min_l}, {max_l}], Radar(m/s): [{min_r}, {max_r}]")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    node_types = ['lidar', 'radar1', 'radar2']
    edge_types = [
        ('lidar', 'spatial', 'lidar'), ('lidar', 'temporal', 'lidar'),
        ('radar1', 'spatial', 'radar1'), ('radar1', 'temporal', 'radar1'),
        ('radar2', 'spatial', 'radar2'), ('radar2', 'temporal', 'radar2'),
        ('radar1', 'to', 'lidar'), ('lidar', 'to', 'radar1'),
        ('radar2', 'to', 'lidar'), ('lidar', 'to', 'radar2')
    ]
    metadata = (node_types, edge_types)

    raw_sample = dataset[0]
    node_in_dims = {nt: raw_sample[nt].x.size(1) for nt in node_types}

    model = ST_HGAT(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        heads=cfg['num_heads'],
        metadata=metadata,
        node_in_dims=node_in_dims,
        radius_ll=float(cfg['radius_ll']) / SCALE_POSE,
        radius_rr=float(cfg['radius_rr']) / SCALE_POSE,
        radius_cross=float(cfg['radius_cross']) / SCALE_POSE,
        temporal_radius_ll=float(cfg.get('temporal_radius_ll', 0.15)) / SCALE_POSE,
        temporal_radius_rr=float(cfg.get('temporal_radius_rr', 0.15)) / SCALE_POSE,
        max_num_neighbors_lidar=int(cfg.get('max_num_neighbors_lidar', 16)),
        max_num_neighbors_radar=int(cfg.get('max_num_neighbors_radar', 32))
    ).to(device)

    # Load Weights
    model_path = os.path.join(cfg['save_dir'], "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Model loaded successfully.")

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    print("[Inference] Running Physics-Guided Inference...")
    result_buffers = {'lidar': [], 'radar1': [], 'radar2': []}
    timestamps = inner_dataset.ts

    def norm01(val, mn, mx):
        denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        return torch.clamp((val - mn) / denom, 0.0, 1.0)

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader)):
            batch_data = batch_data.to(device)

            # [수정] LiDAR 헤드 출력이 1ch이므로 언패킹 제거
            outputs, _ = model(batch_data)
            
            # --- LiDAR Output ---
            if 'lidar' in outputs and batch_data['lidar'].x.size(0) > 0:
                log_var_pos_l = outputs['lidar']
                
                # [핵심] Clamp Log Variance
                log_var_pos_l = torch.clamp(log_var_pos_l, min=L_MIN, max=L_MAX)
                
                sig_norm_scale_l = torch.exp(0.5 * log_var_pos_l)
                sig_phys_l = sig_norm_scale_l * SCALE_POSE # Unit: Meters
                
                # Normalize for Plot (0~1)
                sig_norm_l = norm01(sig_phys_l, min_l, max_l)
                
                curr_mask_l = (batch_data['lidar'].x[:, -1] == 0)
                local_pos_l = batch_data['lidar'].pos
                
                if curr_mask_l.any():
                    base_t = timestamps[inner_dataset.indices[idx]]
                    T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)
                    
                    p_curr = local_pos_l[curr_mask_l]
                    s_phys = sig_phys_l[curr_mask_l]
                    s_norm = sig_norm_l[curr_mask_l]
                    
                    p_curr_m = p_curr * SCALE_POSE
                    ones = torch.ones(p_curr.size(0), 1, device=device)
                    p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                    global_pos = (T_base @ p_curr_h.T).T[:, :2]
                    
                    res = torch.cat([global_pos, s_phys, s_norm], dim=1)
                    result_buffers['lidar'].append(res.cpu().numpy())

            # --- Radar Output ---
            for r_key in ['radar1', 'radar2']:
                if r_key in outputs and batch_data[r_key].x.size(0) > 0:
                    log_var_vel_r = outputs[r_key]
                    
                    log_var_vel_r = torch.clamp(log_var_vel_r, min=R_MIN, max=R_MAX)
                    
                    sig_norm_scale_r = torch.exp(0.5 * log_var_vel_r)
                    sig_phys_r = sig_norm_scale_r * SCALE_RADAR_V # Unit: m/s
                    
                    sig_norm_r = norm01(sig_phys_r, min_r, max_r)
                    
                    curr_mask_r = (batch_data[r_key].x[:, -1] == 0)
                    local_pos_r = batch_data[r_key].pos
                    
                    if curr_mask_r.any():
                        base_t = timestamps[inner_dataset.indices[idx]]
                        T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)
                        
                        p_curr = local_pos_r[curr_mask_r]
                        s_phys = sig_phys_r[curr_mask_r]
                        s_norm = sig_norm_r[curr_mask_r]
                        
                        p_curr_m = p_curr * SCALE_POSE
                        ones = torch.ones(p_curr.size(0), 1, device=device)
                        p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                        global_pos = (T_base @ p_curr_h.T).T[:, :2]
                        
                        res = torch.cat([global_pos, s_phys, s_norm], dim=1)
                        result_buffers[r_key].append(res.cpu().numpy())

    # ==========================================
    # 5. Save Results
    # ==========================================
    save_dir = "/mnt/samsung_ssd/hyunchang/inference_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for key, buffer in result_buffers.items():
        if buffer:
            final_array = np.vstack(buffer)
            save_path = os.path.join(save_dir, f"Result_{key}_{target_vers}.txt")
            header = 'x y sigma sigma_norm'
            np.savetxt(save_path, final_array, fmt='%.4f', delimiter=' ', header=header, comments='')
            print(f"[*] Saved: {save_path} (Shape: {final_array.shape})")

if __name__ == "__main__":
    inference()