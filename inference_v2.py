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


def _norm01(val: torch.Tensor, mn: float, mx: float) -> torch.Tensor:
    denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
    return torch.clamp((val - mn) / denom, 0.0, 1.0)


def _resolve_velocity_index(x_dim: int, cfg: dict) -> int:
    """Resolve which feature column inside batch_data['radar*'].x is velocity.

    Priority:
      1) cfg['radar_velocity_feature_index'] if provided
      2) heuristic fallbacks (common layouts)

    NOTE: inference.py uses x[:, -1] as dt (current-frame mask). We avoid that column.
    """
    dt_idx = x_dim - 1

    idx = cfg.get('radar_velocity_feature_index', None)
    if idx is not None:
        idx = int(idx)
        if idx < 0:
            idx = x_dim + idx
        if idx < 0 or idx >= x_dim:
            raise ValueError(f"radar_velocity_feature_index={cfg['radar_velocity_feature_index']} is out of bounds for x_dim={x_dim}")
        if idx == dt_idx:
            raise ValueError(
                f"radar_velocity_feature_index points to dt column (idx={idx}). "
                f"dt is assumed to be the last feature (x[:, -1]). Choose another index."
            )
        return idx

    # Heuristic candidates (based on common feature layouts in this project)
    # - Many builds use vr at index 6 (0-based) in an 11-D raw vector
    # - Some reduced vectors place velocity around index 3
    candidates = [6, 3, 2, 1, 0, 5, 7, 4]
    for c in candidates:
        if c < x_dim and c != dt_idx:
            return c

    # Absolute fallback
    return 0 if dt_idx != 0 else max(0, x_dim - 2)


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
    # 2. Dataset Setup (single version only)
    # ==========================================
    target_vers = cfg.get('inference_version', "v11")
    print(f"[Inference] Target Version: {target_vers}")

    # IMPORTANT: Pass only [target_vers] so radar/lidar/odom are all the same version.
    dataset = get_selected_dataset(cfg['data_root'], [target_vers], cfg['window_size'])
    inner_dataset = dataset.datasets[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Scaling Factors: Must match loss.py exactly
    SCALE_POSE = 10.0
    SCALE_RADAR_V = 5.0

    # Optional override for velocity scaling (feature -> physical m/s)
    VEL_SCALE = float(cfg.get('radar_velocity_scale', SCALE_RADAR_V))

    # [Stability] Clamping Limits: Derived from YAML for physical consistency
    min_l = float(cfg.get('min_sigma_lidar_m', 0.03))
    max_l = float(cfg.get('max_sigma_lidar_m', 0.2))
    L_MIN = 2 * math.log(min_l / SCALE_POSE + 1e-9)
    L_MAX = 2 * math.log(max_l / SCALE_POSE + 1e-9)

    min_r = float(cfg.get('min_sigma_radar_v', 0.10))
    max_r = float(cfg.get('max_sigma_radar_v', 5.0))
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

    ckpt_epoch = int(cfg.get('inference_checkpoint', 0)) # yaml에서 값 읽기 (기본값 0)

    if ckpt_epoch == 0:
        model_filename = "best_model.pth"
        print("[Inference] Mode: Best Model (Auto)")
    else:
        model_filename = f"model_epoch_{ckpt_epoch}.pth"
        print(f"[Inference] Mode: Specific Epoch ({ckpt_epoch})")

    model_path = os.path.join(cfg['save_dir'], model_filename)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[*] Model loaded successfully from {model_path}")
    else:
        print(f"[Error] Model file not found at {model_path}")
        print("Please check 'inference_checkpoint' in params.yaml or the save directory.")
        return
        
    model.eval()
    # print("[*] Model loaded successfully.")

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    print("[Inference] Running Physics-Guided Inference...")
    result_buffers = {'lidar': [], 'radar1': [], 'radar2': []}
    timestamps = inner_dataset.ts

    # Log velocity feature choice once per radar sensor
    vel_choice_logged = {'radar1': False, 'radar2': False}

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader)):
            batch_data = batch_data.to(device)
            outputs, _ = model(batch_data)

            # --- LiDAR Uncertainty Recovery ---
            if 'lidar' in outputs and batch_data['lidar'].x.size(0) > 0:
                log_var_pos_l = torch.clamp(outputs['lidar'], min=L_MIN, max=L_MAX)

                sig_phys_l = torch.exp(0.5 * log_var_pos_l) * SCALE_POSE
                sig_norm_l = _norm01(sig_phys_l, min_l, max_l)

                # Current frame (dt == 0)
                curr_mask_l = (batch_data['lidar'].x[:, -1].abs() < 0.05)
                local_pos_l = batch_data['lidar'].pos

                if curr_mask_l.any():
                    base_t = timestamps[inner_dataset.indices[idx]]
                    T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)

                    p_curr_m = local_pos_l[curr_mask_l] * SCALE_POSE
                    ones = torch.ones(p_curr_m.size(0), 1, device=device)
                    p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                    global_pos = (T_base @ p_curr_h.T).T[:, :2]

                    res = torch.cat([global_pos, sig_phys_l[curr_mask_l], sig_norm_l[curr_mask_l]], dim=1)
                    result_buffers['lidar'].append(res.cpu().numpy())

            # --- Radar Uncertainty + Velocity Recovery ---
            for r_key in ['radar1', 'radar2']:
                if r_key in outputs and batch_data[r_key].x.size(0) > 0:
                    log_var_vel_r = torch.clamp(outputs[r_key], min=R_MIN, max=R_MAX)

                    sig_phys_r = torch.exp(0.5 * log_var_vel_r) * SCALE_RADAR_V
                    sig_norm_r = _norm01(sig_phys_r, min_r, max_r)

                    curr_mask_r = (batch_data[r_key].x[:, -1].abs() < 0.05)
                    local_pos_r = batch_data[r_key].pos

                    if curr_mask_r.any():
                        base_t = timestamps[inner_dataset.indices[idx]]
                        T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)

                        p_curr_m = local_pos_r[curr_mask_r] * SCALE_POSE
                        ones = torch.ones(p_curr_m.size(0), 1, device=device)
                        p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                        global_pos = (T_base @ p_curr_h.T).T[:, :2]

                        # Velocity extraction from features
                        x_feat = batch_data[r_key].x
                        vel_idx = _resolve_velocity_index(x_feat.size(1), cfg)
                        v_feat = x_feat[curr_mask_r, vel_idx]
                        v_phys = v_feat * VEL_SCALE

                        if not vel_choice_logged[r_key]:
                            vmin = float(v_phys.min().item()) if v_phys.numel() else float('nan')
                            vmax = float(v_phys.max().item()) if v_phys.numel() else float('nan')
                            print(f"[Inference] {r_key}: using velocity feature index={vel_idx}, scale={VEL_SCALE} -> phys v range [{vmin:.3f}, {vmax:.3f}] m/s")
                            vel_choice_logged[r_key] = True

                        # Save: x y velocity sigma sigma_norm
                        res = torch.cat([
                            global_pos,
                            v_phys.unsqueeze(1),
                            sig_phys_r[curr_mask_r],
                            sig_norm_r[curr_mask_r]
                        ], dim=1)
                        result_buffers[r_key].append(res.cpu().numpy())

    # ==========================================
    # 5. Save Results
    #   LiDAR: x y sigma sigma_norm
    #   Radar: x y velocity sigma sigma_norm
    # ==========================================
    save_dir = cfg.get('inference_save_dir', "/mnt/samsung_ssd/hyunchang/inference_results")
    os.makedirs(save_dir, exist_ok=True)

    model_suffix = "best" if ckpt_epoch == 0 else f"ep{ckpt_epoch}"

    for key, buffer in result_buffers.items():
        if not buffer:
            continue

        final_array = np.vstack(buffer)
        save_path = os.path.join(save_dir, f"Result2_{key}_{target_vers}_{model_suffix}.txt")

        if key == 'lidar':
            header = 'x y sigma sigma_norm'
        else:
            header = 'x y velocity sigma sigma_norm'

        np.savetxt(save_path, final_array, fmt='%.4f', delimiter=' ', header=header, comments='')
        print(f"[*] Saved: {save_path} (Shape: {final_array.shape})")


if __name__ == "__main__":
    inference()
