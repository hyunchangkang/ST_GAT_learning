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
    wandb.init(project="SensorFusion_Final", config=cfg)
    wandb.run.name = f"PhysicsLoss_{cfg['train_versions']}"

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
        max_num_neighbors_radar=cfg.get("max_num_neighbors_radar", 32),
        min_sigma_lidar_m=cfg.get("min_sigma_lidar_m"), 
        max_sigma_lidar_m=cfg.get("max_sigma_lidar_m"),
        min_sigma_radar_v=cfg.get("min_sigma_radar_v"), 
        max_sigma_radar_v=cfg.get("max_sigma_radar_v"),
        scale_pose=SCALE_POSE,
        scale_radar_v=SCALE_RADAR_V,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = SpatiotemporalUncertaintyLoss(device, cfg)

    best_val_loss = float("inf")
    patience = int(cfg.get("patience", 10))
    counter = 0

    # ====================================================
    # 5. Training Loop
    # ====================================================
    for epoch in range(int(cfg["epochs"])):
        # ------------------------------------------------
        # [Phase 1] Training
        # ------------------------------------------------
        model.train()
        total_train_loss = 0.0
        updates = 0
        
        # [수정] 박사님이 원하시는 Total Loss까지 포함한 누적 딕셔너리
        metrics_accum = {
            'loss_lidar': 0.0,
            
            'radar1_loss_total': 0.0,
            'radar1_loss_temporal': 0.0,
            'radar1_loss_spatial': 0.0,
            
            'radar2_loss_total': 0.0,
            'radar2_loss_temporal': 0.0,
            'radar2_loss_spatial': 0.0
        }

        loop = tqdm(train_loader, desc=f"Ep {epoch+1} [Train]")

        for batch in loop:
            batch = batch.to(device)
            dt = batch.dt_sec[0] 

            optimizer.zero_grad()
            outputs, edge_index_dict = model(batch)
            loss, log_metrics = criterion(outputs, batch, dt, edge_index_dict)

            if loss.item() != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()
                updates += 1
                
                # 키가 일치하는 것만 누적
                for k in metrics_accum.keys():
                    if k in log_metrics:
                        metrics_accum[k] += log_metrics[k]

            loop.set_postfix(loss=f"{loss.item():.4f}")

        # 평균 계산
        avg_train_loss = total_train_loss / max(updates, 1)
        for k in metrics_accum:
            metrics_accum[k] /= max(updates, 1)

        # ------------------------------------------------
        # [Phase 2] Validation
        # ------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        val_updates = 0
        
        # Validation용 누적 딕셔너리 초기화
        val_metrics_accum = {
            'loss_lidar': 0.0,
            
            'radar1_loss_total': 0.0,
            'radar1_loss_temporal': 0.0,
            'radar1_loss_spatial': 0.0,
            
            'radar2_loss_total': 0.0,
            'radar2_loss_temporal': 0.0,
            'radar2_loss_spatial': 0.0
        }

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                dt = batch.dt_sec[0]
                outputs, edge_index_dict = model(batch)
                loss, log_metrics = criterion(outputs, batch, dt, edge_index_dict)

                if loss.item() != 0.0:
                    total_val_loss += loss.item()
                    val_updates += 1
                    
                    for k in val_metrics_accum.keys():
                        if k in log_metrics:
                            val_metrics_accum[k] += log_metrics[k]

        avg_val_loss = total_val_loss / max(val_updates, 1)
        for k in val_metrics_accum:
            val_metrics_accum[k] /= max(val_updates, 1)

        print(f"Ep {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # ------------------------------------------------
        # [Phase 3] WandB Logging (Requested Format)
        # ------------------------------------------------
        log_packet = {
            "Train/Total_Loss": avg_train_loss,
            "Val/Total_Loss": avg_val_loss,
            "LR": optimizer.param_groups[0]["lr"],
            
            # [LiDAR]
            "Train/loss_lidar": metrics_accum['loss_lidar'],
            "Val/loss_lidar": val_metrics_accum['loss_lidar'],

            # [Radar 1]
            "Train/radar1_loss_total": metrics_accum['radar1_loss_total'],
            "Val/radar1_loss_total": val_metrics_accum['radar1_loss_total'],
            "Train/radar1_loss_temporal": metrics_accum['radar1_loss_temporal'],
            "Val/radar1_loss_temporal": val_metrics_accum['radar1_loss_temporal'],
            "Train/radar1_loss_spatial": metrics_accum['radar1_loss_spatial'],
            "Val/radar1_loss_spatial": val_metrics_accum['radar1_loss_spatial'],

            # [Radar 2]
            "Train/radar2_loss_total": metrics_accum['radar2_loss_total'],
            "Val/radar2_loss_total": val_metrics_accum['radar2_loss_total'],
            "Train/radar2_loss_temporal": metrics_accum['radar2_loss_temporal'],
            "Val/radar2_loss_temporal": val_metrics_accum['radar2_loss_temporal'],
            "Train/radar2_loss_spatial": metrics_accum['radar2_loss_spatial'],
            "Val/radar2_loss_spatial": val_metrics_accum['radar2_loss_spatial'],
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
            if counter >= patience:
                print("Early Stopping")
                break

    wandb.finish()

if __name__ == "__main__":
    train()