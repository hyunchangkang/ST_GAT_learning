import torch
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch

# Custom modules
from src.dataset import get_selected_dataset
from src.model import ST_HGAT

def custom_collate(batch):
    return Batch.from_data_list(batch)

def apply_checkerboard_masking(batch):
    batch_a = batch.clone()
    batch_b = batch.clone()

    mask_indices_a = {}
    mask_indices_b = {}

    # NOTE: dt bins are {0.0, 0.1, 0.2, 0.3} in dataset
    # keep logic same (no big changes), but this assumes dt==0.0 is exact.
    if 'lidar' in batch.node_types:
        x = batch['lidar'].x
        curr_mask = (x[:, -1] == 0)
        curr_idx = torch.where(curr_mask)[0]
        even_idx = curr_idx[::2]
        odd_idx = curr_idx[1::2]

        batch_a['lidar'].x[even_idx, 0] = 0.0
        batch_a['lidar'].x[even_idx, 1] = 0.0
        mask_indices_a['lidar'] = even_idx

        batch_b['lidar'].x[odd_idx, 0] = 0.0
        batch_b['lidar'].x[odd_idx, 1] = 0.0
        mask_indices_b['lidar'] = odd_idx

    for key in ['radar1', 'radar2']:
        if key in batch.node_types:
            x = batch[key].x
            curr_mask = (x[:, -1] == 0)
            curr_idx = torch.where(curr_mask)[0]
            even_idx = curr_idx[::2]
            odd_idx = curr_idx[1::2]

            batch_a[key].x[even_idx, 2] = 0.0
            mask_indices_a[key] = even_idx

            batch_b[key].x[odd_idx, 2] = 0.0
            mask_indices_b[key] = odd_idx

    final_batch = Batch.from_data_list([batch_a, batch_b])
    return final_batch, mask_indices_a, mask_indices_b

def inference():
    # 1. Config
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[Inference] Device: {device}")

    # 2. Dataset
    target_vers = cfg.get('inference_version', "v2")
    print(f"[Inference] Target Version: {target_vers}")

    dataset = get_selected_dataset(cfg['data_root'], [target_vers], cfg['window_size'])
    inner_dataset = dataset.datasets[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # >>> [FIX-1] Use dataset scaling to stay consistent
    SCALE_POSE = float(getattr(inner_dataset, "scale_pose", 10.0))
    SCALE_RADAR_V = float(getattr(inner_dataset, "scale_radar_v", 5.0))
    print(f"[Inference] SCALE_POSE={SCALE_POSE}, SCALE_RADAR_V={SCALE_RADAR_V}")

    # Sigma limits (physical units) for normalization column
    MIN_SIG_L = float(cfg.get('min_sigma_lidar_m', 0.03))
    MAX_SIG_L = float(cfg.get('max_sigma_lidar_m', 0.2))
    MIN_SIG_R = float(cfg.get('min_sigma_radar_v', 0.05))
    MAX_SIG_R = float(cfg.get('max_sigma_radar_v', 1.5))

    # 3. Model init (match train scaling)
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

    # >>> [FIX-2] Pass temporal radii + neighbors exactly like train (scale_pose applied)
    model = ST_HGAT(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        heads=cfg['num_heads'],
        metadata=metadata,
        node_in_dims=node_in_dims,

        radius_ll=float(cfg['radius_ll']) / SCALE_POSE,
        radius_rr=float(cfg['radius_rr']) / SCALE_POSE,
        radius_cross=float(cfg['radius_cross']) / SCALE_POSE,

        temporal_radius_ll=float(cfg.get('temporal_radius_ll', 0.6)) / SCALE_POSE,
        temporal_radius_rr=float(cfg.get('temporal_radius_rr', 0.6)) / SCALE_POSE,

        max_num_neighbors_lidar=int(cfg.get('max_num_neighbors_lidar', 20)),
        max_num_neighbors_radar=int(cfg.get('max_num_neighbors_radar', 20)),

        min_sigma_lidar_m=MIN_SIG_L,
        max_sigma_lidar_m=MAX_SIG_L,
        min_sigma_radar_v=MIN_SIG_R,
        max_sigma_radar_v=MAX_SIG_R,

        scale_pose=SCALE_POSE,
        scale_radar_v=SCALE_RADAR_V
    ).to(device)

    # Load
    model_path = os.path.join(cfg['save_dir'], "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Model loaded successfully.")

    # 4. Inference loop
    print("[Inference] Running Checkerboard Inference...")
    result_buffers = {'lidar': [], 'radar1': [], 'radar2': []}
    timestamps = inner_dataset.ts

    def norm01(sig_phys, mn, mx):
        denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        return torch.clamp((sig_phys - mn) / denom, 0.0, 1.0)

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader)):
            batch_data = batch_data.to(device)

            parallel_batch, mask_a, mask_b = apply_checkerboard_masking(batch_data)
            mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2 = model(parallel_batch)

            num_l = batch_data['lidar'].x.size(0)
            num_r1 = batch_data['radar1'].x.size(0)
            num_r2 = batch_data['radar2'].x.size(0)

            # Current-frame masks
            curr_mask_l = (batch_data['lidar'].x[:, -1] == 0)
            curr_mask_r1 = (batch_data['radar1'].x[:, -1] == 0)
            curr_mask_r2 = (batch_data['radar2'].x[:, -1] == 0)

            # Merge sigma from checkerboard
            def merge_sigma(sig_all, num_nodes, maskA, maskB, curr_mask):
                global_curr_idx = torch.where(curr_mask)[0]
                if global_curr_idx.numel() == 0:
                    return torch.empty(0, 1, device=device)

                res_dict = {}
                if len(maskA) > 0:
                    for gidx in maskA:
                        res_dict[int(gidx.item())] = sig_all[:num_nodes][gidx]
                if len(maskB) > 0:
                    for gidx in maskB:
                        res_dict[int(gidx.item())] = sig_all[num_nodes:][gidx]

                if len(res_dict) == 0:
                    return torch.empty(0, 1, device=device)

                return torch.stack([res_dict[int(i.item())] for i in global_curr_idx])

            sorted_sig_l = merge_sigma(sig_l, num_l, mask_a.get('lidar', []), mask_b.get('lidar', []), curr_mask_l)
            sorted_sig_r1 = merge_sigma(sig_r1, num_r1, mask_a.get('radar1', []), mask_b.get('radar1', []), curr_mask_r1)
            sorted_sig_r2 = merge_sigma(sig_r2, num_r2, mask_a.get('radar2', []), mask_b.get('radar2', []), curr_mask_r2)

            # Save result
            if idx >= len(inner_dataset.indices):
                break
            base_t = timestamps[inner_dataset.indices[idx]]
            T_base = inner_dataset.get_T(base_t)
            T_base_torch = torch.tensor(T_base, dtype=torch.float, device=device)

            def save_result(sig_normed, key, curr_mask):
                if sig_normed.size(0) == 0:
                    return

                local_pos = batch_data[key].pos[curr_mask]
                ones = torch.ones(local_pos.size(0), 1, device=device)

                # pos is normalized (meters/scale_pose) -> convert back to meters before transform
                local_pos_h = torch.cat([local_pos * SCALE_POSE, ones], dim=1)
                global_pos = (T_base_torch @ local_pos_h.T).T[:, :2]

                # sigma is normalized units -> convert to physical units
                if key == 'lidar':
                    sig_phys = sig_normed * SCALE_POSE  # meters
                    sig01 = norm01(sig_phys, MIN_SIG_L, MAX_SIG_L)
                    res = torch.cat([global_pos, sig_phys, sig01], dim=1)
                else:
                    sig_phys = sig_normed * SCALE_RADAR_V  # m/s
                    sig01 = norm01(sig_phys, MIN_SIG_R, MAX_SIG_R)
                    res = torch.cat([global_pos, sig_phys, sig01], dim=1)

                result_buffers[key].append(res.detach().cpu().numpy())

            save_result(sorted_sig_l, 'lidar', curr_mask_l)
            save_result(sorted_sig_r1, 'radar1', curr_mask_r1)
            save_result(sorted_sig_r2, 'radar2', curr_mask_r2)

    # 5. Save files
    save_dir = "/mnt/samsung_ssd/hyunchang/inference_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Inference] Saving results to {save_dir}...")

    for key, buffer in result_buffers.items():
        if buffer:
            final_array = np.vstack(buffer)
            save_path = os.path.join(save_dir, f"Result_{key}_{target_vers}.txt")

            # >>> [FIX-3] Add sigma_norm(0~1) column for consistent color mapping in plot
            if key == 'lidar':
                header = 'x y sigma sigma_norm'   # sigma: meters
            else:
                header = 'x y sig_vr sigma_norm'  # sig_vr: m/s

            np.savetxt(save_path, final_array, fmt='%.4f', delimiter=' ', header=header, comments='')
            print(f"[*] Saved: {save_path} (Shape: {final_array.shape})")

if __name__ == "__main__":
    inference()
