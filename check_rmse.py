import time
import os
import torch
import yaml
import numpy as np
import sys
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from src.dataset import get_selected_dataset
from src.model import ST_HGAT
from src.utils import build_graph

def custom_collate(batch):
    """HeteroData 배치 처리를 위한 collate 함수"""
    hetero_data_list = [item[0] for item in batch]
    gt_l_list = [item[1] for item in batch]
    gt_r1_list = [item[2] for item in batch]
    gt_r2_list = [item[3] for item in batch]
    
    batched_hetero = Batch.from_data_list(hetero_data_list)
    batched_gt_l = torch.cat(gt_l_list, dim=0)
    batched_gt_r1 = torch.cat(gt_r1_list, dim=0)
    batched_gt_r2 = torch.cat(gt_r2_list, dim=0)
    
    return batched_hetero, batched_gt_l, batched_gt_r1, batched_gt_r2

def monitor_rmse():
    # 1. 설정 로드
    config_path = "config/params.yaml"
    if not os.path.exists(config_path):
        print("[Error] Config file not found."); return

    with open(config_path, "r") as f: cfg = yaml.safe_load(f)
    
    # [Tip] 만약 학습 중에 GPU 메모리가 부족하면 아래를 'cpu'로 바꾸세요.
    device = torch.device('cpu')
    
    model_path = os.path.join(cfg['save_dir'], "best_model.pth")
    
    print(f"\n" + "="*50)
    print(f"   [Real-time RMSE Monitor] Activated")
    print(f"   Target: {model_path}")
    print(f"   Device: {device}")
    print(f"="*50)

    # 2. 데이터셋 로드 (최초 1회만 수행)
    print("Loading Validation Dataset...")
    val_versions = cfg.get('val_versions', [cfg['target_versions'][-1]])
    try:
        val_dataset = get_selected_dataset(cfg['data_root'], val_versions, cfg['window_size'])
        # RMSE 계산용이므로 batch_size=1
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
        print(f" -> Validation Frames: {len(val_dataset)}")
    except Exception as e:
        print(f"[Error] Data Load Failed: {e}"); return

    # 3. 모델 초기화
    model = ST_HGAT(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'], heads=cfg['num_heads']).to(device)
    
    last_mod_time = 0

    # 4. 감시 루프 시작
    print("\nWaiting for model update... (Ctrl+C to stop)")
    
    while True:
        try:
            if os.path.exists(model_path):
                # 파일 수정 시간 확인
                current_mod_time = os.path.getmtime(model_path)
                
                # 시간이 이전과 다르면 (즉, 새 모델이 저장되면)
                if current_mod_time > last_mod_time:
                    curr_time_str = time.strftime('%H:%M:%S', time.localtime(current_mod_time))
                    print(f"\n[Update Detected] New model saved at {curr_time_str}")
                    
                    # 파일 쓰기가 완료될 때까지 잠시 대기 (안전장치)
                    time.sleep(1.0)
                    
                    # 모델 로드
                    try:
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()
                    except Exception as e:
                        print(f" -> Model load failed (retrying next loop): {e}")
                        continue

                    # RMSE 계산 시작
                    print(" -> Calculating RMSE on validation set...")
                    lidar_errors = []
                    radar_errors = []
                    
                    try:
                        with torch.no_grad():
                            # tqdm 없이 조용히 계산
                            for batch in val_loader:
                                batch_data, gt_l, gt_r1, gt_r2 = batch
                                batch_data = batch_data.to(device)
                                gt_l, gt_r1, gt_r2 = gt_l.to(device), gt_r1.to(device), gt_r2.to(device)
                                
                                edge_index_dict = build_graph(batch_data, cfg['radius_ll'], cfg['radius_rr'], cfg['radius_cross'], device)
                                mu_l, _, mu_r1, _, mu_r2, _ = model(batch_data.x_dict, edge_index_dict)
                                
                                # (1) LiDAR Position Error
                                if mu_l.size(0) > 0 and gt_l.size(0) > 0:
                                    dist_l = torch.cdist(mu_l, gt_l)
                                    min_dist_l, _ = torch.min(dist_l, dim=1)
                                    lidar_errors.extend(min_dist_l.detach().cpu().numpy().flatten().tolist())

                                # (2) Radar Velocity Error
                                for mu_r, gt_r in [(mu_r1, gt_r1), (mu_r2, gt_r2)]:
                                    if mu_r.size(0) > 0 and gt_r.size(0) > 0:
                                        dist_r = torch.cdist(mu_r, gt_r)
                                        min_dist_r, _ = torch.min(dist_r, dim=1)
                                        radar_errors.extend(min_dist_r.detach().cpu().numpy().flatten().tolist())
                    
                    except Exception as calc_error:
                        print(f" -> RMSE calculation failed: {calc_error}")
                        import traceback
                        traceback.print_exc()
                        continue

                    # 결과 계산 및 출력
                    if len(lidar_errors) > 0:
                         # 리스트 평탄화 (안전장치)
                        if isinstance(lidar_errors[0], list): 
                             lidar_errors = [item for sublist in lidar_errors for item in sublist]
                        if len(radar_errors) > 0 and isinstance(radar_errors[0], list):
                             radar_errors = [item for sublist in radar_errors for item in sublist]

                        rmse_l = np.sqrt(np.mean(np.array(lidar_errors)**2))
                        rmse_r = 0.0
                        if len(radar_errors) > 0:
                            rmse_r = np.sqrt(np.mean(np.array(radar_errors)**2))
                        
                        print("-" * 40)
                        print(f" >> LiDAR Position RMSE : {rmse_l:.4f} m")
                        print(f" >> Radar Velocity RMSE : {rmse_r:.4f} m/s")
                        
                        # 상태 진단 메시지
                        if rmse_l < 0.3:
                            print("    [STATUS] ★★★ Excellent (학습 성공)")
                        elif rmse_l < 0.6:
                            print("    [STATUS] Good (정상 학습 중)")
                        elif rmse_l > 2.0:
                            print("    [STATUS] Warning (오차가 큼. 분산 붕괴 주의)")
                        print("-" * 40)
                    
                    # 시간 갱신 (다음 업데이트 대기)
                    last_mod_time = current_mod_time
                    print("Waiting for next best_model update...")

            # 5초마다 파일 확인 (CPU 부하 최소화)
            time.sleep(5)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_rmse()