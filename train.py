import torch
import torch.optim as optim
import yaml
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import wandb

# Project Imports
from src.dataset import get_selected_dataset
from src.model import ST_HGAT
from src.loss import SpatiotemporalUncertaintyLoss

def custom_collate(batch):
    return Batch.from_data_list(batch)

def train():
    # ====================================================
    # 1. Configuration & Setup
    # ====================================================
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(cfg["save_dir"]):
        os.makedirs(cfg["save_dir"])

    # Fixed scales
    SCALE_POSE = 10.0
    SCALE_RADAR_V = 5.0

    print(f"[Config] device={device}")
    print(f"[Train] Mode: LiDAR(Anchor) + Radar(Physics-Consistency Sigma)")

    # WandB Init
    wandb.init(project="ST_learning_v3", config=cfg)
    wandb.run.name = f"NewLoss_v3{cfg['train_versions']}"

    # ====================================================
    # 2. Dataset Setup
    # ====================================================
    print(f"[Data] Loading Train Versions: {cfg['train_versions']}")
    train_dataset = get_selected_dataset(cfg["data_root"], cfg["train_versions"], cfg["window_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"[Data] Loading Validation Versions: {cfg['val_versions']}")
    val_dataset = get_selected_dataset(cfg["data_root"], cfg["val_versions"], cfg["window_size"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
    )

    # ====================================================
    # 3. Model Initialization
    # ====================================================
    dummy_batch = train_dataset[0]
    node_types = ["lidar", "radar1", "radar2"]
    edge_types = [
        ("lidar", "spatial", "lidar"), ("lidar", "temporal", "lidar"),
        ("radar1", "spatial", "radar1"), ("radar1", "temporal", "radar1"),
        ("radar2", "spatial", "radar2"), ("radar2", "temporal", "radar2"),
        ("radar1", "to", "lidar"), ("lidar", "to", "radar1"),
        ("radar2", "to", "lidar"), ("lidar", "to", "radar2"),
    ]
    metadata = (node_types, edge_types)
    node_in_dims = {nt: dummy_batch[nt].x.size(1) for nt in node_types}

    # [수정] 모델 생성자 인자를 ST_HGAT 정의에 맞춰 정렬
    model = ST_HGAT(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        heads=cfg["num_heads"],
        metadata=metadata,
        node_in_dims=node_in_dims,
        radius_ll=cfg["radius_ll"] / SCALE_POSE,
        radius_rr=cfg["radius_rr"] / SCALE_POSE,
        radius_cross=cfg["radius_cross"] / SCALE_POSE,
        temporal_radius_ll=cfg.get("temporal_radius_ll", 0.15) / SCALE_POSE,
        temporal_radius_rr=cfg.get("temporal_radius_rr", 0.15) / SCALE_POSE,
        max_num_neighbors_lidar=cfg.get("max_num_neighbors_lidar", 16),
        max_num_neighbors_radar=cfg.get("max_num_neighbors_radar", 32)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    

    criterion = SpatiotemporalUncertaintyLoss(device, cfg)

    with torch.no_grad():
        # LiDAR 중간값 계산 및 주입
        lidar_init_bias = (criterion.L_MIN + criterion.L_MAX) / 2
        model.head_lidar.bias.fill_(lidar_init_bias)
        
        # Radar 중간값 계산 및 주입
        radar_init_bias = (criterion.R_MIN + criterion.R_MAX) / 2
        model.head_radar.bias.fill_(radar_init_bias)

    best_val_loss = float("inf")
    patience = int(cfg.get("patience", 10))
    counter = 0

    # ====================================================
    # 5. Training Loop
    # ====================================================
    for epoch in range(int(cfg["epochs"])):
        model.train()
        total_train_loss = 0.0
        updates = 0
        
        metrics_accum = {
            'loss_lidar': 0.0,
            'lidar_spatial_loss': 0.0,   
            'lidar_intensity_loss': 0.0, 
            'lidar_temporal_loss': 0.0,
            'radar1_loss_total': 0.0, 'radar1_loss_temporal': 0.0, 'radar1_loss_spatial': 0.0,
            'radar2_loss_total': 0.0, 'radar2_loss_temporal': 0.0, 'radar2_loss_spatial': 0.0,
            'lidar_sigma_mean': 0.0,
            'radar1_sigma_mean': 0.0,
            'radar2_sigma_mean': 0.0,
        }

        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}")

        for batch in loop:
            batch = batch.to(device)
            # [수정] dt = batch.dt_sec[0] 제거 (loss.py에서 개별 처리)

            optimizer.zero_grad()
            outputs, edge_index_dict = model(batch)
            
            # [수정] 인자 3개로 호출 (outputs, batch, edge_index_dict)
            loss, log_metrics = criterion(outputs, batch, edge_index_dict)

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            updates += 1
                
            for k in metrics_accum.keys():
                if k in log_metrics:
                    metrics_accum[k] += log_metrics[k]

            loop.set_postfix(loss=f"{loss.item():.4f}")

        # [수정] Division 로직: 업데이트가 없으면 inf로 설정하여 오판 방지
        avg_train_loss = total_train_loss / updates if updates > 0 else float('inf')
        for k in metrics_accum:
            metrics_accum[k] /= max(updates, 1)

        # ------------------------------------------------
        # [Phase 2] Validation
        # ------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        val_updates = 0
        
        val_metrics_accum = {k: 0.0 for k in metrics_accum.keys()}

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs, edge_index_dict = model(batch)
                loss, log_metrics = criterion(outputs, batch, edge_index_dict)


                total_val_loss += loss.item()
                val_updates += 1
                for k in val_metrics_accum.keys():
                    if k in log_metrics:
                        val_metrics_accum[k] += log_metrics[k]

        # [수정] Division 로직: 업데이트가 없으면 inf로 설정
        avg_val_loss = total_val_loss / val_updates if val_updates > 0 else float('inf')
        for k in val_metrics_accum:
            val_metrics_accum[k] /= max(val_updates, 1)

        print(f"Ep {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # ------------------------------------------------
        # [Phase 3] WandB Logging
        # ------------------------------------------------
        log_packet = {
            "Train/Total_Loss": avg_train_loss,
            "Val/Total_Loss": avg_val_loss,
            "LR": optimizer.param_groups[0]["lr"],
            
            # LiDAR metrics
            "Train/LiDAR_loss": metrics_accum['loss_lidar'],
            "Val/LiDAR_loss": val_metrics_accum['loss_lidar'],
            "Train/LiDAR_spatial_loss": metrics_accum['lidar_spatial_loss'],     
            "Val/LiDAR_spatial_loss": val_metrics_accum['lidar_spatial_loss'], 
            "Train/LiDAR_temporal_loss": metrics_accum['lidar_temporal_loss'],     
            "Val/LiDAR_temporal_loss": val_metrics_accum['lidar_temporal_loss'], 
            "Train/LiDAR_intensity_loss": metrics_accum['lidar_intensity_loss'], 
            "Val/LiDAR_intensity_loss": val_metrics_accum['lidar_intensity_loss'], 

            # Radar1 metrics
            "Train/Radar1_loss_total": metrics_accum['radar1_loss_total'],
            "Val/Radar1_loss_total": val_metrics_accum['radar1_loss_total'],
            "Train/Radar1_loss_temporal": metrics_accum['radar1_loss_temporal'],
            "Val/Radar1_loss_temporal": val_metrics_accum['radar1_loss_temporal'],
            "Train/Radar1_loss_spatial": metrics_accum['radar1_loss_spatial'],
            "Val/Radar1_loss_spatial": val_metrics_accum['radar1_loss_spatial'],

            # Radar2 metrics
            "Train/Radar2_loss_total": metrics_accum['radar2_loss_total'],
            "Val/Radar2_loss_total": val_metrics_accum['radar2_loss_total'],
            "Train/Radar2_loss_temporal": metrics_accum['radar2_loss_temporal'],
            "Val/Radar2_loss_temporal": val_metrics_accum['radar2_loss_temporal'],
            "Train/Radar2_loss_spatial": metrics_accum['radar2_loss_spatial'],
            "Val/Radar2_loss_spatial": val_metrics_accum['radar2_loss_spatial'],

            "Train/LiDAR_sigma_mean_scaled": metrics_accum['lidar_sigma_mean'],
            "Val/LiDAR_sigma_mean_scaled": val_metrics_accum['lidar_sigma_mean'],
            "Train/LiDAR_sigma_mean_m": metrics_accum['lidar_sigma_mean'] * 10.0,
            "Val/LiDAR_sigma_mean_m": val_metrics_accum['lidar_sigma_mean'] * 10.0,

            "Train/Radar1_sigma_mean_mps": metrics_accum['radar1_sigma_mean'] * 5.0,
            "Val/Radar1_sigma_mean_mps": val_metrics_accum['radar1_sigma_mean'] * 5.0,
            "Train/Radar2_sigma_mean_mps": metrics_accum['radar2_sigma_mean'] * 5.0,
            "Val/Radar2_sigma_mean_mps": val_metrics_accum['radar2_sigma_mean'] * 5.0,

        }
        
        wandb.log(log_packet, step=epoch)

        # ------------------------------------------------
        # [Phase 4] Save
        # ------------------------------------------------
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best_model.pth"))
            print(f"[*] Best Model Saved (Loss: {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"[!] No Improvement. Patience: {counter}/{patience}")

        save_interval = int(cfg.get("save_interval", 10))

        if (epoch + 1) % save_interval == 0:
            filename = f"model_epoch_{epoch+1}.pth"
            save_path = os.path.join(cfg["save_dir"], filename)
            torch.save(model.state_dict(), save_path)
            print(f"[+] Periodic Checkpoint Saved: {filename}")

        if counter >= patience:
            print("Early Stopping")
            break

    wandb.finish()

if __name__ == "__main__":
    train()