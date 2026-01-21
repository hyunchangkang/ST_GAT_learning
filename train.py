import torch
import torch.optim as optim
import yaml
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import wandb

# Project Imports
from src.dataset import get_selected_dataset
from src.model import ST_HGAT
from src.utils import build_graph
from src.loss import SelfSupervisedLoss

def custom_collate(batch):
    hetero_data_list = [item[0] for item in batch]
    gt_l = [item[1] for item in batch]
    gt_r1 = [item[2] for item in batch]
    gt_r2 = [item[3] for item in batch]
    batched_hetero = Batch.from_data_list(hetero_data_list)
    return batched_hetero, gt_l, gt_r1, gt_r2

def train():
    # 1. Config Load
    config_path = "config/params.yaml"
    with open(config_path, "r") as f: cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])

    # WandB Initialization
    wandb.init(project="SensorFusion_ST_HGAT", config=cfg)
    wandb.run.name = f"Predict_{cfg['train_versions']}_Val_{cfg['val_versions']}"

    # 2. Dataset Setup
    print(f"[Data] Loading Train Versions: {cfg['train_versions']}")
    train_dataset = get_selected_dataset(cfg['data_root'], cfg['train_versions'], cfg['window_size'])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True, 
        collate_fn=custom_collate, num_workers=8, pin_memory=True, persistent_workers=True 
    )

    print(f"[Data] Loading Validation Versions: {cfg['val_versions']}")
    val_dataset = get_selected_dataset(cfg['data_root'], cfg['val_versions'], cfg['window_size'])
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], shuffle=False, 
        collate_fn=custom_collate, num_workers=8, pin_memory=True
    )
    
    print(f"[Data] Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    
    # 1. Dataset 로딩 후 샘플 하나 추출 (입력 차원 확인용)
    dummy_batch, _, _, _ = train_dataset[0]

    # 2. Metadata 수동 정의 (GNN 레이어 활성화를 위해 필수)
    node_types = ['lidar', 'radar1', 'radar2']
    edge_types = [
        ('lidar', 'spatial', 'lidar'), ('lidar', 'temporal', 'lidar'),
        ('radar1', 'spatial', 'radar1'), ('radar1', 'temporal', 'radar1'),
        ('radar2', 'spatial', 'radar2'), ('radar2', 'temporal', 'radar2'),
        ('radar1', 'to', 'lidar'), ('lidar', 'to', 'radar1'),
        ('radar2', 'to', 'lidar'), ('lidar', 'to', 'radar2')
    ]
    metadata = (node_types, edge_types)

    # 3. node_in_dims 정의 (이 줄이 모델 생성보다 먼저 와야 합니다)
    # 각 노드 타입별로 입력 피처의 개수를 딕셔너리로 저장합니다.
    node_in_dims = {nt: dummy_batch[nt].x.size(1) for nt in node_types}

    # 4. Model 초기화
    model = ST_HGAT(
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers'], 
        heads=cfg['num_heads'], 
        metadata=metadata, 
        node_in_dims=node_in_dims # 이제 여기서 에러가 나지 않습니다.
    ).to(device)

    # train.py 내 모델 초기화 직후 배치
    # print("\n" + "="*50)
    # print("[Metadata Check] Model expects these edge types:")
    # for etype in metadata[1]:
    #     print(f" - {etype}")

    # print("\n[Build Check] build_graph generates these edge types:")
    
    # build_graph를 한 번 호출해서 키값을 뽑아봅니다.
    test_edges = build_graph(dummy_batch.to(device), 0.1, 0.1, 0.1, 0.15, 0.3, device)
    for etype in test_edges.keys():
        print(f" - {etype}")
    print("="*50 + "\n")
    
    wandb.watch(model, log="all")

    lr = float(cfg.get('learning_rate', 0.0001)) 
    weight_decay = float(cfg.get('weight_decay', 0.001))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # 4. Loss Initialization
    # predictive NLL loss for t+1 frame
    criterion = SelfSupervisedLoss(
        device=device, 
        min_sigma_lidar_m=0.02,   
        min_sigma_radar_v=0.1,  
        dist_threshold=1.0
    ).to(device)

    print(f"[Train] Prediction Mode: Estimating uncertainty for t+1 based on trajectory.")
    
    best_val_loss = float('inf')
    patience = cfg.get('patience', 10)
    counter = 0

    # 5. Training Loop
    for epoch in range(cfg['epochs']):
        # --- [Training Phase] ---
        model.train()
        total_train_loss = 0.0
        total_l, total_r1, total_r2 = 0.0, 0.0, 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")
        
        for batch in loop:
            batch_data = batch[0].to(device)
            # These are now t+1 GT samples
            gt_l = torch.cat([g.to(device) for g in batch[1]], dim=0)
            gt_r1 = torch.cat([g.to(device) for g in batch[2]], dim=0)
            gt_r2 = torch.cat([g.to(device) for g in batch[3]], dim=0)

            # Build Sequential ST-Graph (dt-based filtering) 
            edge_index_dict = build_graph(
                batch_data, 
                cfg['radius_ll'], 
                cfg['radius_rr'], 
                cfg['radius_cross'], 
                cfg.get('temporal_radius_ll', 0.15), # yaml에서 읽어오되 없을 시 기본값 설정
                cfg.get('temporal_radius_rr', 0.3), 
                device
            )
            # print("\n" + "="*30)
            # for key, value in edge_index_dict.items():
            #     print(f"[Edge Check] Type: {key}, Count: {value.size(1)}")
            # print("="*30)
            # --- [에지 확인 디버깅 코드 끝] ---
            
            optimizer.zero_grad()
            
            # Predict t+1 mu and sigma
            mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2 = model(batch_data.x_dict, edge_index_dict)

            # --- [수정 포인트: 현재 시점(t=0) 노드만 필터링] ---
            # 부동 소수점 오차 방지를 위해 == 0.0 대신 abs < 0.05 사용
            mask_l = batch_data['lidar'].x[:, -1].abs() < 0.05
            mask_r1 = batch_data['radar1'].x[:, -1].abs() < 0.05
            mask_r2 = batch_data['radar2'].x[:, -1].abs() < 0.05

            # 필터링된 mu와 sigma (이 텐서들의 size(0)가 이제 0보다 커집니다)
            mu_l_f, sig_l_f = mu_l[mask_l], sig_l[mask_l]
            mu_r1_f, sig_r1_f = mu_r1[mask_r1], sig_r1[mask_r1]
            mu_r2_f, sig_r2_f = mu_r2[mask_r2], sig_r2[mask_r2]

            pos_r1_f = batch_data['radar1'].pos[mask_r1]
            pos_r2_f = batch_data['radar2'].pos[mask_r2]

            # 4. log_v 변환
            def to_log_v(s): return 2 * torch.log(s) if s.size(0) > 0 else s
            log_l, log_r1, log_r2 = to_log_v(sig_l_f), to_log_v(sig_r1_f), to_log_v(sig_r2_f)

            # 5. Criterion 호출 (필터링된 좌표 pos_r1_f, pos_r2_f 전달)
            loss, l_l, l_r1, l_r2 = criterion(
                mu_l_f, log_l, gt_l,
                mu_r1_f, log_r1, gt_r1, pos_r1_f,  # 수정된 부분
                mu_r2_f, log_r2, gt_r2, pos_r2_f   # 수정된 부분
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_l += l_l.item()    
            total_r1 += l_r1.item()  
            total_r2 += l_r2.item()  
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_l = total_l / len(train_loader)     
        avg_train_r1 = total_r1 / len(train_loader)   
        avg_train_r2 = total_r2 / len(train_loader)   

        # --- [Validation Phase] ---
        model.eval()
        total_val_loss = 0.0
        val_l, val_r1, val_r2 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch_data = batch[0].to(device)
                gt_l = torch.cat([g.to(device) for g in batch[1]], dim=0)
                gt_r1 = torch.cat([g.to(device) for g in batch[2]], dim=0)
                gt_r2 = torch.cat([g.to(device) for g in batch[3]], dim=0)

                edge_index_dict = build_graph(
                    batch_data, 
                    cfg['radius_ll'], 
                    cfg['radius_rr'], 
                    cfg['radius_cross'], 
                    cfg.get('temporal_radius_ll', 0.15), # yaml에서 읽어오되 없을 시 기본값 설정
                    cfg.get('temporal_radius_rr', 0.3), 
                    device
                )

                mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2 = model(batch_data.x_dict, edge_index_dict)

                # --- [수정 포인트: 동일한 필터링 적용] ---
                mask_l = batch_data['lidar'].x[:, -1].abs() < 0.05
                mask_r1 = batch_data['radar1'].x[:, -1].abs() < 0.05
                mask_r2 = batch_data['radar2'].x[:, -1].abs() < 0.05

                mu_l_f, sig_l_f = mu_l[mask_l], sig_l[mask_l]
                mu_r1_f, sig_r1_f = mu_r1[mask_r1], sig_r1[mask_r1]
                mu_r2_f, sig_r2_f = mu_r2[mask_r2], sig_r2[mask_r2]
                
                pos_r1_f = batch_data['radar1'].pos[mask_r1]
                pos_r2_f = batch_data['radar2'].pos[mask_r2]
                
                log_l, log_r1, log_r2 = to_log_v(sig_l_f), to_log_v(sig_r1_f), to_log_v(sig_r2_f)
                
                loss, l_l, l_r1, l_r2 = criterion(
                    mu_l_f, log_l, gt_l,
                    mu_r1_f, log_r1, gt_r1, pos_r1_f, 
                    mu_r2_f, log_r2, gt_r2, pos_r2_f
                )

                total_val_loss += loss.item()
                val_l += l_l.item()    
                val_r1 += l_r1.item()  
                val_r2 += l_r2.item()  

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_l = val_l / len(val_loader)
        avg_val_r1 = val_r1 / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        
        # --- [Logging & Checkpoint] ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | "
              f"Train: {avg_train_loss:.4f}  | "
              f"Val: {avg_val_loss:.4f}  | "
              f"LR: {current_lr:.6f}")
        
        wandb.log({
            "Loss/Train_Total": avg_train_loss,
            "Loss/Train_LiDAR": avg_train_l,
            "Loss/Train_Radar1": avg_train_r1,
            "Loss/Train_Radar2": avg_train_r2,
            "Loss/Val_Total": avg_val_loss,
            "Loss/Val_LiDAR": avg_val_l,
            "Loss/Val_Radar1": avg_val_r1,
            "Loss/Val_Radar2": avg_val_r2,
            "Parameters/Learning_Rate": current_lr
        }, step=epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], "best_model.pth"))
            print(f"[*] Best Model Saved (Val Loss: {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"[!] No Improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"[Stop] Early Stopping at Epoch {epoch+1}"); break

    wandb.finish()

if __name__ == "__main__":
    train()