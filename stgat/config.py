# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


# NOTE:
# - Keep ModelConfig field names stable to preserve checkpoint compatibility.
# - YAML is the source of truth when keys exist (CLI can still override explicitly).


@dataclass
class ModelConfig:
    # Unified raw vector spec (11-D)
    INPUT_DIM: int = 11
    IDX_POS: Tuple[int, int] = (0, 1)
    IDX_DT: int = 2
    IDX_EGO: Tuple[int, int] = (3, 4)
    IDX_INTENSITY: int = 5
    IDX_VR: int = 6
    IDX_SNR: int = 7
    IDX_SID: Tuple[int, int, int] = (8, 9, 10)


    # Feature normalization (divide inputs/targets during training)
    # - Positions (x,y): /POS_NORM
    # - LiDAR intensity : /INTENSITY_NORM
    # - Radar radial velocity v_r: /VR_NORM
    # - Radar SNR: /SNR_NORM
    POS_NORM: float = 10.0
    VR_NORM: float = 5.0
    SNR_NORM: float = 50.0
    INTENSITY_NORM: float = 4000.0

    # Window / frames
    WINDOW: int = 4
    DT_DEFAULT: float = 0.1

    # Downsample caps (TRAINING); inference can disable
    LIDAR_CAP_PER_FRAME: int = 720
    RADAR_CAP_PER_FRAME: int = 32

    # Graph construction
    K_LIDAR_SPATIAL: int = 32
    K_RADAR_SPATIAL: int = 6
    K_TEMPORAL: int = 12
    TEMPORAL_ADJ_ONLY: bool = True


    # Temporal distance gate (post-filter after kNN)
    TEMPORAL_RADIUS_LIDAR: float = 0.5  ## 0.15 -> 0.5
    TEMPORAL_RADIUS_RADAR: float = 0.3  ## 0.25 ->0.5
    # Cross edges
    CROSS_RADIUS: float = 4.0       # 0.5 ->4.0
    K_CROSS_L2R: int = 8            # 4 -> 8 
    K_CROSS_R2L: int = 4            # 8 -> 4
    CROSS_DROPOUT: float = 0.35

    # Model
    HIDDEN_DIM: int = 64
    NUM_HEADS: int = 4
    DROPOUT: float = 0.1
    EDGE_DIM: int = 4

    # Sigma constraints
    # - We bound *log_sigma* per sensor type.
    # - User-facing minima are specified in sigma (std) units.
    MIN_SIGMA_LIDAR: float = 0.05   # meters
    MIN_SIGMA_RADAR: float = 0.3    # m/s (radial velocity)
    MAX_LOG_SIGMA: float = 2.0
    SIGMA_BOUND_MODE: str = 'sigmoid'  # 'sigmoid' (recommended) or 'clamp' (compat)

    # Loss
    REG_LAMBDA: float = 1e-3
    RADAR_LOSS_WEIGHT: float = 5.0
    MU_LOSS_WEIGHT: float = 1.0
    SIGMA_NLL_WEIGHT: float = 1.0
    ASSOC_TOPK: int = 5
    ASSOC_TAU: float = 0.5
    ASSOC_GATE_LIDAR: float = 0.15
    ASSOC_GATE_RADAR: float = 0.3

    # Cross residual schedule
    CROSS_ALPHA_MAX: float = 0.15
    CROSS_WARMUP_STEPS: int = 3000
    CROSS_RAMP_STEPS: int = 10000

    # Training defaults
    BATCH_SIZE: int = 2
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 35
    GRAD_CLIP: float = 1.0
    AMP: bool = True



def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_versions(vs: Any) -> Optional[List[int]]:
    if vs is None:
        return None
    out: List[int] = []
    for x in vs:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append(int(x))
        elif isinstance(x, str):
            s = x.strip()
            if s.lower().startswith("v"):
                s = s[1:].strip()
            if s == "":
                continue
            out.append(int(s))
        else:
            raise TypeError(f"Unsupported version type in YAML: {type(x)} ({x})")
    return out if len(out) > 0 else None


def apply_yaml_to_cfg(cfg: ModelConfig, yaml_cfg: Dict[str, Any]) -> ModelConfig:
    """Apply YAML keys to cfg in-place.

    This function only maps keys that are explicitly present in the YAML file.
    It does not introduce new hyperparameters beyond what YAML specifies.
    """

    # ------------------------
    # 2) Model hyperparameters
    # ------------------------
    if "window_size" in yaml_cfg:
        cfg.WINDOW = int(yaml_cfg["window_size"])
    if "hidden_dim" in yaml_cfg:
        cfg.HIDDEN_DIM = int(yaml_cfg["hidden_dim"])
    if "num_heads" in yaml_cfg:
        cfg.NUM_HEADS = int(yaml_cfg["num_heads"])
    if "dropout" in yaml_cfg:
        cfg.DROPOUT = float(yaml_cfg["dropout"])

    # ------------------------
    # 3) Graph construction
    # ------------------------
    if "max_num_neighbors_lidar" in yaml_cfg:
        cfg.K_LIDAR_SPATIAL = int(yaml_cfg["max_num_neighbors_lidar"])
    if "max_num_neighbors_radar" in yaml_cfg:
        cfg.K_RADAR_SPATIAL = int(yaml_cfg["max_num_neighbors_radar"])
    if "k_temporal" in yaml_cfg:
        cfg.K_TEMPORAL = int(yaml_cfg["k_temporal"])
    if "temporal_adj_only" in yaml_cfg:
        cfg.TEMPORAL_ADJ_ONLY = bool(yaml_cfg["temporal_adj_only"])
    if "temporal_radius_ll" in yaml_cfg:
        cfg.TEMPORAL_RADIUS_LIDAR = float(yaml_cfg["temporal_radius_ll"])
    if "temporal_radius_rr" in yaml_cfg:
        cfg.TEMPORAL_RADIUS_RADAR = float(yaml_cfg["temporal_radius_rr"])

    if "radius_cross" in yaml_cfg:
        cfg.CROSS_RADIUS = float(yaml_cfg["radius_cross"])
    if "k_cross_l2r" in yaml_cfg:
        cfg.K_CROSS_L2R = int(yaml_cfg["k_cross_l2r"])
    if "k_cross_r2l" in yaml_cfg:
        cfg.K_CROSS_R2L = int(yaml_cfg["k_cross_r2l"])
    if "cross_dropout" in yaml_cfg:
        cfg.CROSS_DROPOUT = float(yaml_cfg["cross_dropout"])

    # ------------------------
    # 4) Sigma constraints
    # ------------------------
    if "min_sigma_lidar_m" in yaml_cfg:
        cfg.MIN_SIGMA_LIDAR = float(yaml_cfg["min_sigma_lidar_m"])
    if "min_sigma_radar_v" in yaml_cfg:
        cfg.MIN_SIGMA_RADAR = float(yaml_cfg["min_sigma_radar_v"])
    if "max_log_sigma" in yaml_cfg:
        cfg.MAX_LOG_SIGMA = float(yaml_cfg["max_log_sigma"])
    if "sigma_bound_mode" in yaml_cfg:
        cfg.SIGMA_BOUND_MODE = str(yaml_cfg["sigma_bound_mode"])

    # ------------------------
    # 5) Training
    # ------------------------
    if "batch_size" in yaml_cfg:
        cfg.BATCH_SIZE = int(yaml_cfg["batch_size"])
    if "learning_rate" in yaml_cfg:
        cfg.LR = float(yaml_cfg["learning_rate"])
    if "weight_decay" in yaml_cfg:
        cfg.WEIGHT_DECAY = float(yaml_cfg["weight_decay"])
    if "epochs" in yaml_cfg:
        cfg.EPOCHS = int(yaml_cfg["epochs"])
    if "grad_clip" in yaml_cfg:
        cfg.GRAD_CLIP = float(yaml_cfg["grad_clip"])
    if "amp" in yaml_cfg:
        cfg.AMP = bool(yaml_cfg["amp"])

    return cfg
