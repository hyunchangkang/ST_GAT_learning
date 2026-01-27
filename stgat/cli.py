# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb

from .config import ModelConfig, load_yaml_config, parse_versions, apply_yaml_to_cfg
from .data import MultiRunTextWindowDataset, collate_fn
from .graph import GraphBuilder
from .infer_export import infer_export
from .loss import UncertaintyLoss
from .model import DOGMSTGATUncertaintyNet
from .train_eval import train_one_epoch, eval_one_epoch
from .utils import set_seed, extract_zip_if_needed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["train", "infer"])
    parser.add_argument("--config", type=str, default="params_tw.yaml", help="Path to config yaml file.")

    parser.add_argument("--data_root", type=str, default="", help="Directory containing *_v#.txt files (or extracted subdir).")
    parser.add_argument("--data_zip", type=str, default=None, help="Optional: zip path to extract.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for .npz")
    parser.add_argument("--mode", type=str, default="train", choices=["debug", "train"],
                        help="train task only: debug reduces runtime (does not override YAML unless key absent)")

    # Training overrides (explicit CLI overrides only; YAML is default source of truth)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lidar_cap", type=int, default=None)
    parser.add_argument("--radar_cap", type=int, default=None)
    parser.add_argument("--sigma_bound", type=str, default=None, choices=["sigmoid", "clamp"],
                        help="Log-sigma bounding: sigmoid (recommended) or clamp (compat).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_windows_per_run", type=int, default=None, help="Limit windows per run for quick tests.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from (e.g., last_ckpt.pt).")

    # Inference (export)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for inference. If omitted, uses YAML model_path.")
    parser.add_argument("--infer_version", type=int, default=None, help="Run version v# to export. If omitted, uses YAML inference_version.")
    parser.add_argument("--infer_out", type=str, default=None, help="Output NPZ path. If omitted, uses YAML inference_save_dir + version.")
    parser.add_argument("--infer_full_points", action="store_true", help="Export sigmas for all points (no downsampling).")
    parser.add_argument("--infer_max_frames", type=int, default=None, help="Optional: export only first N frames after warmup.")

    args = parser.parse_args()

    # -------------------------
    # Load YAML (source of truth)
    # -------------------------
    yaml_cfg: Dict[str, Any] = {}
    if Path(args.config).exists():
        yaml_cfg = load_yaml_config(args.config)
        print(f"Loading config from {args.config}...")
    else:
        print(f"Config file {args.config} not found. Using CLI/defaults.")

    # data_root: CLI has priority, otherwise YAML
    if (not args.data_root) and ("data_root" in yaml_cfg):
        args.data_root = str(yaml_cfg["data_root"])
        print(f" -> Set data_root from yaml: {args.data_root}")

    if not args.data_root:
        raise ValueError("Set --data_root (or provide data_root in YAML).")

    # Seed
    seed = int(yaml_cfg.get("seed", 42))
    set_seed(seed)

    # Build cfg from defaults, then apply YAML, then apply explicit CLI overrides
    cfg = ModelConfig()
    apply_yaml_to_cfg(cfg, yaml_cfg)

    # CLI explicit overrides (only when provided)
    if args.epochs is not None:
        cfg.EPOCHS = int(args.epochs)
    if args.batch is not None:
        cfg.BATCH_SIZE = int(args.batch)
    if args.lidar_cap is not None:
        cfg.LIDAR_CAP_PER_FRAME = int(args.lidar_cap)
    if args.radar_cap is not None:
        cfg.RADAR_CAP_PER_FRAME = int(args.radar_cap)
    if args.sigma_bound is not None:
        cfg.SIGMA_BOUND_MODE = str(args.sigma_bound)

    # debug mode: only override runtime limits if YAML/CLI did not already specify
    if args.mode == "debug":
        if (args.epochs is None) and ("epochs" not in yaml_cfg):
            cfg.EPOCHS = 10
        if args.max_windows_per_run is None:
            args.max_windows_per_run = 800

    # Device
    device_str = str(yaml_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)
    print(
        f"Device: {device} | sigma_bound={cfg.SIGMA_BOUND_MODE}"
    )

    # Initialize wandb
    wandb.init(project="gat-dogm-training", config=cfg.__dict__)

    # Data root (optional zip extraction)
    data_root = extract_zip_if_needed(args.data_zip, args.data_root)

    model = DOGMSTGATUncertaintyNet(cfg).to(device)
    graph_builder = GraphBuilder(cfg)
    loss_fn = UncertaintyLoss(cfg)

    # -------------------------
    # Inference task (export)
    # -------------------------
    if args.task == "infer":
        # Version selection
        infer_version = args.infer_version
        if infer_version is None:
            iv = yaml_cfg.get("inference_version", None)
            if isinstance(iv, str) and iv.lower().startswith("v"):
                infer_version = int(iv[1:])
            elif iv is not None:
                infer_version = int(iv)
            else:
                infer_version = 4

        # Ckpt path selection
        ckpt_path = args.ckpt or yaml_cfg.get("model_path", "best_ckpt.pt")

        ckpt = torch.load(ckpt_path, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

        # Output path selection
        infer_out = args.infer_out
        if infer_out is None:
            out_root = str(yaml_cfg.get("inference_save_dir", "output/inference_results"))
            infer_out = str(Path(out_root) / f"v{infer_version}" / "sigma_export.npz")

        Path(infer_out).parent.mkdir(parents=True, exist_ok=True)

        infer_export(
            cfg=cfg,
            model=model,
            graph_builder=graph_builder,
            data_root=data_root,
            version=int(infer_version),
            out_path=infer_out,
            cache_dir=args.cache_dir,
            device=device,
            full_points=args.infer_full_points,
            max_frames=args.infer_max_frames,
        )
        return

    # -------------------------
    # Training task
    # -------------------------
    yaml_train_versions = parse_versions(yaml_cfg.get("train_versions"))
    yaml_val_versions = parse_versions(yaml_cfg.get("val_versions"))
    yaml_target_versions = parse_versions(yaml_cfg.get("target_versions"))

    if yaml_train_versions is not None:
        train_versions = yaml_train_versions
    elif yaml_target_versions is not None:
        train_versions = yaml_target_versions
    else:
        train_versions = [1] if args.mode == "debug" else [1, 5, 8, 10, 11, 13]

    val_versions = yaml_val_versions  # if None -> no validation

    train_ds = MultiRunTextWindowDataset(
        cfg,
        data_root=data_root,
        versions=train_versions,
        cache_dir=args.cache_dir,
        max_windows_per_run=args.max_windows_per_run,
        seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if val_versions is not None:
        val_ds = MultiRunTextWindowDataset(
            cfg,
            data_root=data_root,
            versions=val_versions,
            cache_dir=args.cache_dir,
            max_windows_per_run=args.max_windows_per_run,
            seed=seed + 81,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    # Resume
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming training from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch-1}, global_step {global_step}. Starting epoch {start_epoch}.")

    # Save paths (YAML-driven when available)
    script_dir = Path(__file__).resolve().parent.parent
    save_dir = Path(str(yaml_cfg.get("save_dir", script_dir / "output")))
    if not save_dir.is_absolute():
        save_dir = (script_dir / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(str(yaml_cfg.get("model_path", save_dir / "best_ckpt.pt")))
    if not model_path.is_absolute():
        model_path = (script_dir / model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = model_path
    last_ckpt_path = save_dir / "last_ckpt.pt"

    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, cfg.EPOCHS):
        last_epoch = epoch
        global_step, train_metrics = train_one_epoch(
            cfg,
            model,
            graph_builder,
            loss_fn,
            train_loader,
            optimizer,
            device,
            epoch,
            global_step,
            accum_steps=max(args.accum, 1),
        )

        wandb.log(train_metrics, step=epoch)
        print(
            f"[Train E{epoch:02d} Epoch Avg] "
            f"loss={train_metrics['train_loss_epoch']:.4f} "
            f"L={train_metrics['train_loss_lidar_epoch']:.4f} "
            f"R1={train_metrics['train_loss_r1_epoch']:.4f} "
            f"R2={train_metrics['train_loss_r2_epoch']:.4f}"
        )

        if val_loader is not None:
            metrics = eval_one_epoch(cfg, model, graph_builder, loss_fn, val_loader, device)
            print(
                f"[Val   E{epoch:02d}] "
                f"loss={metrics['total']:.4f} "
                f"L={metrics['lidar']:.4f} "
                f"R1={metrics['r1']:.4f} "
                f"R2={metrics['r2']:.4f}"
            )
            wandb.log(
                {
                    "val_loss": metrics["total"],
                    "val_loss_lidar": metrics["lidar"],
                    "val_loss_r1": metrics["r1"],
                    "val_loss_r2": metrics["r2"],
                    "epoch": epoch,
                },
                step=epoch,
            )

            if metrics["total"] < best_val:
                best_val = metrics["total"]
                payload = {
                    "cfg": cfg.__dict__,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                torch.save(payload, str(best_ckpt_path))
                print(f"Saved {best_ckpt_path}")

    # save last ckpt
    payload = {
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": last_epoch,
    }
    torch.save(payload, str(last_ckpt_path))
    print(f"Saved {last_ckpt_path}")
