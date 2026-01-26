# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb

from .config import ModelConfig, load_yaml_config, parse_versions, apply_yaml_to_cfg
from .data import MultiRunTextWindowDataset, collate_fn
from .graph import GraphBuilder
from .infer import infer_export
from .loss import UncertaintyLoss
from .model import DOGMSTGATUncertaintyNet
from .train_eval import train_one_epoch, eval_one_epoch
from .utils import set_seed, extract_zip_if_needed


def _resolve_config_path() -> str:
    """Find a default YAML in the current working directory."""
    candidates = ["params.yaml", "params_tw.yaml", "params.yml", "params_tw.yml"]
    for c in candidates:
        if Path(c).exists():
            return c
    # fallback to params_tw.yaml name (keeps prior convention)
    return "params_tw.yaml"


def load_all(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], ModelConfig]:
    cfg_path = config_path or _resolve_config_path()
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    yaml_cfg = load_yaml_config(cfg_path)
    cfg = apply_yaml_to_cfg(ModelConfig(), yaml_cfg)
    return yaml_cfg, cfg


def train_from_yaml(config_path: Optional[str] = None) -> None:
    yaml_cfg, cfg = load_all(config_path)

    # Required
    data_root = str(yaml_cfg.get("data_root", "")).strip()
    if not data_root:
        raise ValueError("YAML must include data_root")

    # Optional
    data_zip = yaml_cfg.get("data_zip", None)
    cache_dir = yaml_cfg.get("cache_dir", None)
    max_windows_per_run = yaml_cfg.get("max_windows_per_run", None)
    num_workers = int(yaml_cfg.get("num_workers", 8))
    accum_steps = int(yaml_cfg.get("accum", 1))
    resume_from = yaml_cfg.get("resume_from", None)
    mode = str(yaml_cfg.get("mode", "train")).lower()

    # debug mode (same behavior as cli.py but YAML-driven)
    if mode == "debug":
        if "epochs" not in yaml_cfg:
            cfg.EPOCHS = 10
        if max_windows_per_run is None:
            max_windows_per_run = 800

    # Seed
    seed = int(yaml_cfg.get("seed", 42))
    set_seed(seed)

    # Device
    device_str = str(yaml_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)
    print(f"Device: {device} | sigma_bound={cfg.SIGMA_BOUND_MODE}")

    # Optional zip extraction
    data_root = extract_zip_if_needed(data_zip, data_root)

    model = DOGMSTGATUncertaintyNet(cfg).to(device)
    graph_builder = GraphBuilder(cfg)
    loss_fn = UncertaintyLoss(cfg)

    # Versions
    yaml_train_versions = parse_versions(yaml_cfg.get("train_versions"))
    yaml_val_versions = parse_versions(yaml_cfg.get("val_versions"))
    yaml_target_versions = parse_versions(yaml_cfg.get("target_versions"))

    if yaml_train_versions is not None:
        train_versions = yaml_train_versions
    elif yaml_target_versions is not None:
        train_versions = yaml_target_versions
    else:
        train_versions = [1, 5, 8, 10, 11, 13]

    val_versions = yaml_val_versions

    train_ds = MultiRunTextWindowDataset(
        cfg,
        data_root=data_root,
        versions=train_versions,
        cache_dir=cache_dir,
        max_windows_per_run=max_windows_per_run,
        seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
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
            cache_dir=cache_dir,
            max_windows_per_run=max_windows_per_run,
            seed=seed + 81,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # W&B (YAML-controlled)
    use_wandb = bool(yaml_cfg.get("use_wandb", True))
    if use_wandb:
        wandb.init(project=str(yaml_cfg.get("wandb_project", "gat-dogm-training")), config=cfg.__dict__)

    # Save paths
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

    # Resume
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if resume_from and Path(str(resume_from)).exists():
        resume_from = str(resume_from)
        print(f"Resuming training from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch-1}, global_step {global_step}. Starting epoch {start_epoch}.")

    for epoch in range(start_epoch, cfg.EPOCHS):
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
            accum_steps=max(accum_steps, 1),
        )

        if use_wandb:
            wandb.log({"epoch": epoch, **train_metrics}, step=epoch)


        print(
            f"[Train E{epoch:03d}] "
            f"loss={train_metrics['train_loss_epoch']:.4f} "
            f"L={train_metrics['train_loss_lidar_epoch']:.4f} "
            f"R1={train_metrics['train_loss_r1_epoch']:.4f} "
            f"R2={train_metrics['train_loss_r2_epoch']:.4f}"
        )

        # Validation
        if val_loader is not None:
            metrics = eval_one_epoch(cfg, model, graph_builder, loss_fn, val_loader, device)
            val_total = float(metrics["total"])
            print(
                f"[Val   E{epoch:03d}] loss={val_total:.4f} "
                f"L={metrics['lidar']:.4f} R1={metrics['r1']:.4f} R2={metrics['r2']:.4f}"
            )

            if use_wandb:
                wandb.log(
                {
                    "epoch": epoch,
                    "val_loss": val_total,
                    "val_lidar": metrics["lidar"],
                    "val_r1": metrics["r1"],
                    "val_r2": metrics["r2"],
                },
                step=epoch,
            )
            if val_total < best_val:
                best_val = val_total
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, str(best_ckpt_path))
                print(f"[OK] Saved best checkpoint: {best_ckpt_path} (val_loss={best_val:.4f})")

        # Always save last
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step, "cfg": cfg.__dict__},
            str(last_ckpt_path),
        )

    print(f"[DONE] Training finished. Best ckpt: {best_ckpt_path} | Last ckpt: {last_ckpt_path}")


def infer_from_yaml(config_path: Optional[str] = None) -> None:
    yaml_cfg, cfg = load_all(config_path)

    data_root = str(yaml_cfg.get("data_root", "")).strip()
    if not data_root:
        raise ValueError("YAML must include data_root")

    data_zip = yaml_cfg.get("data_zip", None)
    cache_dir = yaml_cfg.get("cache_dir", None)

    seed = int(yaml_cfg.get("seed", 42))
    set_seed(seed)

    device_str = str(yaml_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)
    print(f"Device: {device} | sigma_bound={cfg.SIGMA_BOUND_MODE}")

    data_root = extract_zip_if_needed(data_zip, data_root)

    # Which run version to export
    iv = yaml_cfg.get("inference_version", None)
    if isinstance(iv, str) and iv.lower().startswith("v"):
        infer_version = int(iv[1:])
    elif iv is not None:
        infer_version = int(iv)
    else:
        raise ValueError("YAML must include inference_version")

    # Which checkpoint to load
    ckpt_path = str(yaml_cfg.get("model_path", "output/best_ckpt.pt"))
    if not Path(ckpt_path).exists():
        # if model_path is relative, try relative to project root
        script_dir = Path(__file__).resolve().parent.parent
        alt = (script_dir / ckpt_path).resolve()
        if alt.exists():
            ckpt_path = str(alt)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Output NPZ path
    out_root = str(yaml_cfg.get("inference_save_dir", "output/inference_results"))
    out_path = str(Path(out_root) / f"v{infer_version}" / "sigma_export.npz")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Export options
    full_points = bool(yaml_cfg.get("infer_full_points", False))
    max_frames = yaml_cfg.get("infer_max_frames", None)
    max_frames = int(max_frames) if max_frames is not None else None

    model = DOGMSTGATUncertaintyNet(cfg).to(device)
    graph_builder = GraphBuilder(cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    infer_export(
        cfg=cfg,
        model=model,
        graph_builder=graph_builder,
        data_root=data_root,
        version=int(infer_version),
        out_path=out_path,
        cache_dir=cache_dir,
        device=device,
        full_points=full_points,
        max_frames=max_frames,
    )
    print(f"[DONE] Saved NPZ: {out_path}")
