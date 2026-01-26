# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import ModelConfig
from .utils import read_txt_np as _read_txt_np, assign_to_odom_frames as _assign_to_odom_frames, sort_by_frame as _sort_by_frame


class MultiRunTextWindowDataset(torch.utils.data.Dataset):
    """
    Training dataset yielding:
      sample_t  : nodes for frames [t-3..t]   (WINDOW frames, last is current)

    Supervision is applied on the last window frame (t) against itself (current-frame training).

    Caches per v are built under cache_dir as cache_v{v}.npz.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        data_root: str,
        versions: List[int],
        cache_dir: Optional[str] = None,
        max_windows_per_run: Optional[int] = None,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.data_root = Path(data_root)
        self.versions = list(versions)
        self.seed = seed

        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_root / "_cache_npz")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.runs: List[Dict[str, Any]] = []
        self.index: List[Tuple[int, int]] = []  # (run_i, t_end)

        for vi, v in enumerate(self.versions):
            run = self._load_or_build_run_cache(v)
            self.runs.append(run)

            F = int(run["F"])
            # valid t_end: need [t_end-3..t_end] within [0..F-1] AND t_end+1 exists
            t_ends = list(range(cfg.WINDOW - 1, F))
            if max_windows_per_run is not None:
                rng = np.random.RandomState(seed + v)
                if len(t_ends) > max_windows_per_run:
                    t_ends = rng.choice(t_ends, size=max_windows_per_run, replace=False).tolist()
                    t_ends.sort()
            for t_end in t_ends:
                self.index.append((vi, t_end))

        if len(self.index) == 0:
            raise RuntimeError("No valid windows found. Check your data_root and versions.")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _pad_offsets(off: np.ndarray, F: int) -> np.ndarray:
        if off.shape[0] == F + 1:
            return off
        if off.shape[0] < F + 1:
            last = off[-1] if off.size > 0 else 0
            pad = np.full((F + 1 - off.shape[0],), last, dtype=np.int64)
            return np.concatenate([off, pad], axis=0)
        return off[:F+1]

    def _load_or_build_run_cache(self, v: int) -> Dict[str, Any]:
        cache_path = self.cache_dir / f"cache_v{v}.npz"
        if cache_path.exists():
            z = np.load(str(cache_path), allow_pickle=False)
            return {
                "v": v,
                "F": int(z["F"]),
                "t_odom": z["t_odom"].astype(np.float32),
                "pose": z["pose"].astype(np.float32),
                "twist": z["twist"].astype(np.float32),
                "lidar": z["lidar"].astype(np.float32),
                "lidar_off": z["lidar_off"].astype(np.int64),
                "r1": z["r1"].astype(np.float32),
                "r1_off": z["r1_off"].astype(np.int64),
                "r2": z["r2"].astype(np.float32),
                "r2_off": z["r2_off"].astype(np.int64),
            }

        lidar_path = self.data_root / f"LiDARMap_BaseScan_v{v}.txt"
        r1_path = self.data_root / f"Radar1Map_BaseScan_v{v}.txt"
        r2_path = self.data_root / f"Radar2Map_BaseScan_v{v}.txt"
        odom_path = self.data_root / f"odom_filtered_v{v}.txt"

        if not (lidar_path.exists() and r1_path.exists() and r2_path.exists() and odom_path.exists()):
            raise FileNotFoundError(f"Missing files for v{v} under {self.data_root}")

        odom = _read_txt_np(odom_path, expected_cols=6, dtype=np.float32)
        t_odom = odom[:, 0].astype(np.float32)
        pose = odom[:, 1:4].astype(np.float32)   # x,y,yaw
        twist = odom[:, 4:6].astype(np.float32)  # v,w
        F_ = t_odom.shape[0]

        # LiDAR: [t,x,y,intensity] -> store [x,y,intensity]
        lidar_raw = _read_txt_np(lidar_path, expected_cols=4, dtype=np.float32)
        fidx = _assign_to_odom_frames(lidar_raw[:, 0], t_odom)
        lidar_data = np.stack([lidar_raw[:, 1], lidar_raw[:, 2], lidar_raw[:, 3]], axis=1).astype(np.float32)
        lidar_sorted, lidar_off = _sort_by_frame(fidx, lidar_data)
        lidar_off = self._pad_offsets(lidar_off, F_)

        # Radar1: [t,x,y,vr,snr] -> store [x,y,vr,snr]
        r1_raw = _read_txt_np(r1_path, expected_cols=5, dtype=np.float32)
        fidx = _assign_to_odom_frames(r1_raw[:, 0], t_odom)
        r1_data = np.stack([r1_raw[:, 1], r1_raw[:, 2], r1_raw[:, 3], r1_raw[:, 4]], axis=1).astype(np.float32)
        r1_sorted, r1_off = _sort_by_frame(fidx, r1_data)
        r1_off = self._pad_offsets(r1_off, F_)

        # Radar2
        r2_raw = _read_txt_np(r2_path, expected_cols=5, dtype=np.float32)
        fidx = _assign_to_odom_frames(r2_raw[:, 0], t_odom)
        r2_data = np.stack([r2_raw[:, 1], r2_raw[:, 2], r2_raw[:, 3], r2_raw[:, 4]], axis=1).astype(np.float32)
        r2_sorted, r2_off = _sort_by_frame(fidx, r2_data)
        r2_off = self._pad_offsets(r2_off, F_)

        np.savez_compressed(
            str(cache_path),
            F=np.int64(F_),
            t_odom=t_odom,
            pose=pose,
            twist=twist,
            lidar=lidar_sorted,
            lidar_off=lidar_off,
            r1=r1_sorted,
            r1_off=r1_off,
            r2=r2_sorted,
            r2_off=r2_off,
        )

        return {
            "v": v, "F": int(F_), "t_odom": t_odom, "pose": pose, "twist": twist,
            "lidar": lidar_sorted, "lidar_off": lidar_off,
            "r1": r1_sorted, "r1_off": r1_off,
            "r2": r2_sorted, "r2_off": r2_off,
        }

    def _slice_frame(self, arr: np.ndarray, off: np.ndarray, f: int) -> np.ndarray:
        s = int(off[f]); e = int(off[f+1])
        if e <= s:
            return arr[0:0]
        return arr[s:e]

    def _downsample(self, pts: np.ndarray, cap: int, rng: np.random.RandomState) -> np.ndarray:
        if cap <= 0 or pts.shape[0] <= cap:
            return pts
        idx = rng.choice(pts.shape[0], size=cap, replace=False)
        return pts[idx]

    def __getitem__(self, idx: int):
        cfg = self.cfg
        run_i, t_end = self.index[idx]
        run = self.runs[run_i]

        F = int(run["F"])
        t_odom = run["t_odom"]
        pose = run["pose"]
        twist = run["twist"]

        frames = [t_end - 3, t_end - 2, t_end - 1, t_end]

        rng = np.random.RandomState(self.seed + idx * 1337 + int(run["v"]) * 17)
        x_list: List[torch.Tensor] = []
        fid_list: List[torch.Tensor] = []
        sid_list: List[torch.Tensor] = []

        t_cur = float(t_odom[t_end])
        pose_win = np.stack([pose[f] for f in frames], axis=0).astype(np.float32)

        for local_f, f in enumerate(frames):
            dt = float(t_cur - float(t_odom[f]))
            v, w = twist[f].astype(np.float32)

            # LiDAR
            L = self._slice_frame(run["lidar"], run["lidar_off"], f)
            L = self._downsample(L, cfg.LIDAR_CAP_PER_FRAME, rng)
            if L.shape[0] > 0:
                x = np.zeros((L.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = L[:, 0]; x[:, 1] = L[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_INTENSITY] = L[:, 2]
                x[:, cfg.IDX_SID[0]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((L.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((L.shape[0],), 0, dtype=torch.long))

            # Radar1
            R1 = self._slice_frame(run["r1"], run["r1_off"], f)
            R1 = self._downsample(R1, cfg.RADAR_CAP_PER_FRAME, rng)
            if R1.shape[0] > 0:
                x = np.zeros((R1.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R1[:, 0]; x[:, 1] = R1[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R1[:, 2]; x[:, cfg.IDX_SNR] = R1[:, 3]
                x[:, cfg.IDX_SID[1]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R1.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R1.shape[0],), 1, dtype=torch.long))

            # Radar2
            R2 = self._slice_frame(run["r2"], run["r2_off"], f)
            R2 = self._downsample(R2, cfg.RADAR_CAP_PER_FRAME, rng)
            if R2.shape[0] > 0:
                x = np.zeros((R2.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R2[:, 0]; x[:, 1] = R2[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R2[:, 2]; x[:, cfg.IDX_SNR] = R2[:, 3]
                x[:, cfg.IDX_SID[2]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R2.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R2.shape[0],), 2, dtype=torch.long))

        if len(x_list) == 0:
            return self.__getitem__((idx + 1) % len(self))

        x_t = torch.cat(x_list, dim=0).float()
        frame_id_t = torch.cat(fid_list, dim=0)
        sensor_id_t = torch.cat(sid_list, dim=0)

        sample_t = {
            "x": x_t,
            "frame_id": frame_id_t,
            "sensor_id": sensor_id_t,
            "pose_by_frame": torch.from_numpy(pose_win).float(),  # (WINDOW,3)
        }
        return sample_t


def collate_fn(batch):
    sample_ts = batch

    x_list, f_list, sid_list, bid_list = [], [], [], []
    pose_list = []

    for b, st in enumerate(sample_ts):
        x = st["x"]
        f = st["frame_id"]
        sid = st["sensor_id"]
        x_list.append(x)
        f_list.append(f)
        sid_list.append(sid)
        bid_list.append(torch.full((x.size(0),), b, dtype=torch.long))
        pose_list.append(st["pose_by_frame"])

    x = torch.cat(x_list, dim=0)
    frame_id = torch.cat(f_list, dim=0)
    sensor_id = torch.cat(sid_list, dim=0)
    batch_id = torch.cat(bid_list, dim=0)

    pose_by_frame = torch.stack(pose_list, dim=0)  # (B, WINDOW, 3)

    return (x, frame_id, batch_id, sensor_id, pose_by_frame)




