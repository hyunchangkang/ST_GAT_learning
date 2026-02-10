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
    # Inference 대상 데이터 버전 (예: v11)
    target_vers = cfg.get('inference_version', "v11")
    print(f"[Inference] Target Data Version: {target_vers}")

    # 데이터셋 로드 (학습 때와 동일한 설정)
    dataset = get_selected_dataset(cfg['data_root'], [target_vers], cfg['window_size'])
    inner_dataset = dataset.datasets[0] # 내부 단일 시퀀스 데이터셋 접근 (Timestamp, Odom 접근용)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Scaling Factors: Must match train.py & dataset.py exactly
    SCALE_POSE = 10.0    
    SCALE_RADAR_V = 5.0 

    # [Stability] Clamping Limits (Uncertainty 복원용)
    min_l = float(cfg.get('min_sigma_lidar_m', 0.03))
    max_l = float(cfg.get('max_sigma_lidar_m', 1.0)) 
    L_MIN = 2 * math.log(min_l / SCALE_POSE + 1e-9)
    L_MAX = 2 * math.log(max_l / SCALE_POSE + 1e-9)

    min_r = float(cfg.get('min_sigma_radar_v', 0.10))
    max_r = float(cfg.get('max_sigma_radar_v', 5.0))
    R_MIN = 2 * math.log(min_r / SCALE_RADAR_V + 1e-9)
    R_MAX = 2 * math.log(max_r / SCALE_RADAR_V + 1e-9)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    # 모델 구조 정의 (Metadata)
    node_types = ['lidar', 'radar1', 'radar2']
    edge_types = [
        ('lidar', 'spatial', 'lidar'), ('lidar', 'temporal', 'lidar'),
        ('radar1', 'spatial', 'radar1'), ('radar1', 'temporal', 'radar1'),
        ('radar2', 'spatial', 'radar2'), ('radar2', 'temporal', 'radar2'),
        ('radar1', 'to', 'lidar'), ('lidar', 'to', 'radar1'),
        ('radar2', 'to', 'lidar'), ('lidar', 'to', 'radar2')
    ]
    metadata = (node_types, edge_types)

    # 입력 차원 확인 (Dummy Sample)
    raw_sample = dataset[0]
    node_in_dims = {nt: raw_sample[nt].x.size(1) for nt in node_types}

    # 모델 생성 (학습과 동일한 파라미터)
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

    # [핵심] Checkpoint Selection Logic
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}")

        
    model.eval()

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    # 결과 버퍼: [t, x, y, sigma, sigma_norm] 저장
    result_buffers = {'lidar': [], 'radar1': [], 'radar2': []}
    timestamps = inner_dataset.ts

    # Visualization용 Normalize 함수 (0~1)
    def norm01(val, mn, mx):
        denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        return torch.clamp((val - mn) / denom, 0.0, 1.0)

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader, desc="Inferencing")):
            batch_data = batch_data.to(device)
            outputs, _ = model(batch_data)
            
            # 현재 윈도우의 기준 시간(Base Time) 가져오기 (이것이 raw 데이터의 1열 시간값)
            base_idx = inner_dataset.indices[idx]
            base_t = timestamps[base_idx]

            window_size = cfg.get('window_size', 10) # params.yaml의 값 사용
            last_idx = base_idx + window_size - 1
            
            # # 인덱스 범위 초과 방지
            # if last_idx < len(timestamps):
            #     last_t = timestamps[last_idx]
            # else:
            #     last_t = -1.0 

            # # 50프레임마다 한 번씩 혹은 초반 10프레임 동안 출력하여 확인
            # if idx < 10 or idx % 50 == 0:
            #     print(f"\n[Debug Frame {idx}]")
            #     print(f" - Window Start (base_t): {base_t:.4f}s (Index: {base_idx})")
            #     print(f" - Window End   (last_t): {last_t:.4f}s (Index: {last_idx})")
            #     print(f" - Time Diff: {last_t - base_t:.4f}s")
                
            #     # 현재 mask가 찾고 있는 데이터의 상대 시간 확인 (보통 0.0 근처가 현재 프레임)
            #     sample_rel_t = batch_data['lidar'].x[:, -1].cpu().numpy()
            #     print(f" - Min Rel Time in Batch: {sample_rel_t.min():.4f}")
            #     print(f" - Max Rel Time in Batch: {sample_rel_t.max():.4f}")
            
            # --- LiDAR Processing ---
            if 'lidar' in outputs and batch_data['lidar'].x.size(0) > 0:
                # 1. Uncertainty 복원 (Scaling Factor 적용)
                log_var_l = torch.clamp(outputs['lidar'], min=L_MIN, max=L_MAX)
                sig_phys_l = torch.exp(0.5 * log_var_l) * SCALE_POSE 
                sig_norm_l = norm01(sig_phys_l, min_l, max_l)
                
                # 2. 현재 프레임(t=0) 데이터만 추출 (Masking)
                curr_mask_l = (batch_data['lidar'].x[:, -1].abs() < 0.05)
                
                if curr_mask_l.any():
                    # 3. Global Coordinate Transformation
                    T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)
                    
                    # 로컬 좌표 -> 스케일 복원 -> Homogeneous 좌표
                    p_curr_m = batch_data['lidar'].pos[curr_mask_l] * SCALE_POSE
                    ones = torch.ones(p_curr_m.size(0), 1, device=device)
                    p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                    
                    # Global 좌표로 변환
                    global_pos = (T_base @ p_curr_h.T).T[:, :2]
                    
                    # [수정됨] Time Column 생성 (현재 포인트 개수만큼 base_t 복사)
                    num_points = global_pos.size(0)
                    t_col = torch.full((num_points, 1), base_t, device=device)

                    # [t, Global_X, Global_Y, Sigma_Meter, Sigma_Norm] 순서로 저장
                    res = torch.cat([t_col, global_pos, sig_phys_l[curr_mask_l], sig_norm_l[curr_mask_l]], dim=1)
                    result_buffers['lidar'].append(res.cpu().numpy())

            # --- Radar Processing ---
            for r_key in ['radar1', 'radar2']:
                if r_key in outputs and batch_data[r_key].x.size(0) > 0:
                    log_var_r = torch.clamp(outputs[r_key], min=R_MIN, max=R_MAX)
                    sig_phys_r = torch.exp(0.5 * log_var_r) * SCALE_RADAR_V
                    sig_norm_r = norm01(sig_phys_r, min_r, max_r)
                    
                    curr_mask_r = (batch_data[r_key].x[:, -1].abs() < 0.05)
                    
                    if curr_mask_r.any():
                        T_base = torch.tensor(inner_dataset.get_T(base_t), dtype=torch.float, device=device)
                        
                        p_curr_m = batch_data[r_key].pos[curr_mask_r] * SCALE_POSE
                        ones = torch.ones(p_curr_m.size(0), 1, device=device)
                        p_curr_h = torch.cat([p_curr_m, ones], dim=1)
                        global_pos = (T_base @ p_curr_h.T).T[:, :2]
                        
                        # [수정됨] Time Column 생성
                        num_points = global_pos.size(0)
                        t_col = torch.full((num_points, 1), base_t, device=device)
                        
                        # [t, x, y, sigma, sigma_norm]
                        res = torch.cat([t_col, global_pos, sig_phys_r[curr_mask_r], sig_norm_r[curr_mask_r]], dim=1)
                        result_buffers[r_key].append(res.cpu().numpy())

    # ==========================================
    # 5. Save Results
    # ==========================================
    save_dir = cfg.get('inference_save_dir', "/mnt/samsung_ssd/hyunchang/inference_results")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 파일명에 에폭 정보 추가 (덮어쓰기 방지)
    model_suffix = "best" if ckpt_epoch == 0 else f"ep{ckpt_epoch}"

    for key, buffer in result_buffers.items():
        if buffer:
            final_array = np.vstack(buffer)
            # 예: Result_lidar_v11_ep10.txt
            save_path = os.path.join(save_dir, f"Result_{key}_{target_vers}_{model_suffix}.txt")
            
            # [수정됨] 헤더에 't' 추가
            np.savetxt(save_path, final_array, fmt='%.4f', header='t x y sigma sigma_norm', comments='')
            print(f"[*] Saved: {save_path} (Points: {len(final_array)})")

if __name__ == "__main__":
    inference()