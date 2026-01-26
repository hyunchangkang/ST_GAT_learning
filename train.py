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
from src.utils import apply_masking
from src.loss import SelfSupervisedLoss


def custom_collate(batch):
    return Batch.from_data_list(batch)


def train():
    # ====================================================
    # 1. Configuration & Unit Setup
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

    # Masking settings
    mask_ratio = float(cfg.get("mask_ratio", 0.3))

    # LiDAR pos-noise (normalized) tied to radius_ll
    # Recommended: 0.2 * (radius_ll / SCALE_POSE)
    radius_ll_norm = float(cfg["radius_ll"]) / SCALE_POSE
    lidar_pos_noise_std = float(cfg.get("lidar_pos_noise_std", 0.2 * radius_ll_norm))

    print(f"[Config] device={device}")
    print(f"[Masking] mask_ratio={mask_ratio}, lidar_pos_noise_std(norm)={lidar_pos_noise_std:.6f}")

    # WandB
    wandb.init(project="SensorFusion_Masked_HGAT", config=cfg)
    wandb.run.name = f"Masked_{cfg['train_versions']}_mr{mask_ratio}"

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
        ("lidar", "spatial", "lidar"),
        ("lidar", "temporal", "lidar"),
        ("radar1", "spatial", "radar1"),
        ("radar1", "temporal", "radar1"),
        ("radar2", "spatial", "radar2"),
        ("radar2", "temporal", "radar2"),
        ("radar1", "to", "lidar"),
        ("lidar", "to", "radar1"),
        ("radar2", "to", "lidar"),
        ("lidar", "to", "radar2"),
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
        min_sigma_lidar_m=cfg.get("min_sigma_lidar_m", 0.05),
        max_sigma_lidar_m=cfg.get("max_sigma_lidar_m", 2.0),
        min_sigma_radar_v=cfg.get("min_sigma_radar_v", 0.15),
        max_sigma_radar_v=cfg.get("max_sigma_radar_v", 5.0),
        scale_pose=SCALE_POSE,
        scale_radar_v=SCALE_RADAR_V,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ====================================================
    # 4. Loss Initialization
    # ====================================================
    criterion = SelfSupervisedLoss(
        device=device,
        min_sigma_lidar_m=cfg.get("min_sigma_lidar_m", 0.05),
        min_sigma_radar_v=cfg.get("min_sigma_radar_v", 0.15),
    )

    print("[Train] Mode: Masked Reconstruction (Self-Supervised)")

    best_val_loss = float("inf")
    patience = int(cfg.get("patience", 10))
    counter = 0

    # ====================================================
    # 5. Training Loop
    # ====================================================
    for epoch in range(int(cfg["epochs"])):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        total_train_loss = 0.0
        num_updates = 0

        train_dist_error_l = 0.0
        train_abs_error_r = 0.0
        count_l = 0
        count_r = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")

        for batch in loop:
            batch_data = batch.to(device)

            masked_batch, mask_idx, gt_vals = apply_masking(
                batch_data,
                mask_ratio=mask_ratio,
                lidar_pos_noise_std=lidar_pos_noise_std,
            )

            optimizer.zero_grad()

            mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2 = model(masked_batch)

            loss, l_l, l_r1, l_r2 = criterion(
                mu_l, sig_l, gt_vals.get("lidar"), mask_idx.get("lidar"),
                mu_r1, sig_r1, gt_vals.get("radar1"), mask_idx.get("radar1"),
                mu_r2, sig_r2, gt_vals.get("radar2"), mask_idx.get("radar2"),
            )

            if loss.item() != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                num_updates += 1

            with torch.no_grad():
                if "lidar" in mask_idx and mask_idx["lidar"] is not None and len(mask_idx["lidar"]) > 0:
                    idx = mask_idx["lidar"]
                    diff = (mu_l[idx] - gt_vals["lidar"]) * SCALE_POSE
                    train_dist_error_l += torch.norm(diff, dim=1).sum().item()
                    count_l += len(idx)

                for r_key, mu_r, gt_r in [
                    ("radar1", mu_r1, gt_vals.get("radar1")),
                    ("radar2", mu_r2, gt_vals.get("radar2")),
                ]:
                    if r_key in mask_idx and mask_idx[r_key] is not None and len(mask_idx[r_key]) > 0:
                        idx = mask_idx[r_key]
                        diff = (mu_r[idx] - gt_r) * SCALE_RADAR_V
                        train_abs_error_r += torch.abs(diff).sum().item()
                        count_r += len(idx)

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(num_updates, 1)
        avg_train_poserr_m = train_dist_error_l / max(count_l, 1)
        avg_train_radar_mae = train_abs_error_r / max(count_r, 1)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        val_updates = 0

        val_dist_error_l = 0.0
        val_abs_error_r = 0.0
        val_count_l = 0
        val_count_r = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_data = batch.to(device)

                masked_batch, mask_idx, gt_vals = apply_masking(
                    batch_data,
                    mask_ratio=mask_ratio,
                    lidar_pos_noise_std=lidar_pos_noise_std,
                )

                mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2 = model(masked_batch)

                loss, _, _, _ = criterion(
                    mu_l, sig_l, gt_vals.get("lidar"), mask_idx.get("lidar"),
                    mu_r1, sig_r1, gt_vals.get("radar1"), mask_idx.get("radar1"),
                    mu_r2, sig_r2, gt_vals.get("radar2"), mask_idx.get("radar2"),
                )

                if loss.item() != 0.0:
                    total_val_loss += loss.item()
                    val_updates += 1

                if "lidar" in mask_idx and mask_idx["lidar"] is not None and len(mask_idx["lidar"]) > 0:
                    idx = mask_idx["lidar"]
                    diff = (mu_l[idx] - gt_vals["lidar"]) * SCALE_POSE
                    val_dist_error_l += torch.norm(diff, dim=1).sum().item()
                    val_count_l += len(idx)

                for r_key, mu_r, gt_r in [
                    ("radar1", mu_r1, gt_vals.get("radar1")),
                    ("radar2", mu_r2, gt_vals.get("radar2")),
                ]:
                    if r_key in mask_idx and mask_idx[r_key] is not None and len(mask_idx[r_key]) > 0:
                        idx = mask_idx[r_key]
                        diff = (mu_r[idx] - gt_r) * SCALE_RADAR_V
                        val_abs_error_r += torch.abs(diff).sum().item()
                        val_count_r += len(idx)

        avg_val_loss = total_val_loss / max(val_updates, 1)
        avg_val_poserr_m = val_dist_error_l / max(val_count_l, 1)
        avg_val_radar_mae = val_abs_error_r / max(val_count_r, 1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | L_PosErr(m): {avg_train_poserr_m:.3f} | R_MAE(m/s): {avg_train_radar_mae:.3f}"
        )
        print(
            f"       [Val] | Loss: {avg_val_loss:.4f} | L_PosErr(m): {avg_val_poserr_m:.3f} | R_MAE(m/s): {avg_val_radar_mae:.3f}"
        )

        wandb.log(
            {
                "Loss/Train_Total": avg_train_loss,
                "Loss/Val_Total": avg_val_loss,
                "Metric/Train_LiDAR_PosErr_m": avg_train_poserr_m,
                "Metric/Train_Radar_MAE_ms": avg_train_radar_mae,
                "Metric/Val_LiDAR_PosErr_m": avg_val_poserr_m,
                "Metric/Val_Radar_MAE_ms": avg_val_radar_mae,
                "LR": current_lr,
                "Mask/mask_ratio": mask_ratio,
                "Mask/lidar_pos_noise_std_norm": lidar_pos_noise_std,
            },
            step=epoch,
        )

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
                print("[Stop] Early Stopping")
                break

    wandb.finish()


if __name__ == "__main__":
    train()
