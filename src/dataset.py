import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from torch_geometric.data import HeteroData

class SensorFusionDataset(Dataset):
    def __init__(self, data_root, versions, window_size=4):
        self.datasets = [SingleSequenceDataset(data_root, v, window_size) for v in versions]
        self.lengths = [len(d) for d in self.datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        return self.datasets[ds_idx][idx - self.cum_lengths[ds_idx]]

class SingleSequenceDataset(Dataset):
    def __init__(self, data_root, ver, window_size):
        self.window_size = window_size
        # Load Raw Data
        self.lidar_df = pd.read_csv(os.path.join(data_root, "Basescan", f"LiDARMap_BaseScan_{ver}.txt"), sep='\s+', header=None, names=['t', 'x', 'y', 'I'])
        self.radar1_df = pd.read_csv(os.path.join(data_root, "Basescan", f"Radar1Map_BaseScan_{ver}.txt"), sep='\s+', header=None, names=['t', 'x', 'y', 'vr', 'SNR'])
        self.radar2_df = pd.read_csv(os.path.join(data_root, "Basescan", f"Radar2Map_BaseScan_{ver}.txt"), sep='\s+', header=None, names=['t', 'x', 'y', 'vr', 'SNR'])
        self.odom_df = pd.read_csv(os.path.join(data_root, f"odom_filtered_{ver}.txt"), sep='\s+', header=None, names=['t', 'x', 'y', 'yaw', 'v', 'w'])
        
        # Timestamp binning (0.1s interval)
        self.ts = sorted(self.lidar_df['t'].round(1).unique())
        
        # [수정] t+1 시점 데이터를 GT로 사용하므로, 마지막 프레임은 학습 불가능 (t+1이 없음)
        # 따라서 범위를 하나 줄입니다.
        self.indices = range(window_size - 1, len(self.ts) - 1)

        # Empirical scaling
        self.scale_pose = 10.0
        self.scale_lidar_i = 4000.0 
        self.scale_radar_v = 5.0    
        self.scale_radar_snr = 50.0

    def get_T(self, t):
        """ Get Global Transformation matrix at rounded timestamp t """
        row = self.odom_df[self.odom_df['t'].round(1) == round(t, 1)]
        if row.empty: return np.eye(3)
        v = row.iloc[0]
        c, s = np.cos(v['yaw']), np.sin(v['yaw'])
        return np.array([[c, -s, v['x']], [s, c, v['y']], [0, 0, 1]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Current window ends at t
        current_idx = self.indices[idx]
        base_t = self.ts[current_idx]      # Current frame (t)
        next_t = self.ts[current_idx + 1]  # Future frame (t+1)
        
        window_ts = self.ts[current_idx - self.window_size + 1 : current_idx + 1]
        
        T_base = self.get_T(base_t)
        T_inv_base = np.linalg.inv(T_base)
        
        data = HeteroData()

        # =========================================================
        # 1. Input Data Construction (t-3 ~ t)
        # =========================================================
        for key, df, dim in [('lidar', self.lidar_df, 4), ('radar1', self.radar1_df, 5), ('radar2', self.radar2_df, 5)]:
            pts_list = []
            for t_bin in window_ts:
                curr = df[df['t'].round(1) == round(t_bin, 1)].values
                if len(curr) > 0:
                    # Align all past frames to current base_t coordinate
                    T_rel = T_inv_base @ self.get_T(t_bin)
                    xy_h = np.hstack([curr[:, 1:3], np.ones((len(curr), 1))])
                    trans_xy = (T_rel @ xy_h.T).T[:, :2] / self.scale_pose 
                    
                    dt_val = round(base_t - t_bin, 1) # Relative time
                    features = curr[:, 3:].copy()
                    
                    if key == 'lidar':
                        features[:, 0] = np.clip(features[:, 0] / self.scale_lidar_i, 0, 1)
                    else:
                        features[:, 0] = np.clip(features[:, 0] / self.scale_radar_v, -1, 1)
                        features[:, 1] = np.clip(features[:, 1] / self.scale_radar_snr, 0, 1)
                    
                    # Node features: [trans_x, trans_y, normalized_features..., dt_bin]
                    pts_list.append(np.hstack([trans_xy, features, np.full((len(curr), 1), dt_val)]))
            
            if pts_list:
                res = np.vstack(pts_list)
                data[key].x = torch.tensor(res, dtype=torch.float)
                data[key].pos = torch.tensor(res[:, :2], dtype=torch.float)
            else:
                data[key].x, data[key].pos = torch.empty((0, dim)), torch.empty((0, 2))

        # =========================================================
        # 2. GT Data Construction (t+1 transformed to t frame)
        # [중요] GT 포인트들도 PyG의 'Node'로 정의하여 Batch 처리를 지원하게 함
        # =========================================================
        
        T_next = self.get_T(next_t)
        # Transform from (t+1) global to (t) local
        # P_t = T_base_inv * T_next * P_next
        T_next_to_base = T_inv_base @ T_next

        # (1) LiDAR GT (for Geometric Consistency)
        l_next = self.lidar_df[self.lidar_df['t'].round(1) == round(next_t, 1)].values
        if len(l_next) > 0:
            xy_h = np.hstack([l_next[:, 1:3], np.ones((len(l_next), 1))])
            trans_xy = (T_next_to_base @ xy_h.T).T[:, :2] / self.scale_pose
            
            data['gt_lidar'].pos = torch.tensor(trans_xy, dtype=torch.float)
            # Dummy feature x required for PyG batching mechanism
            data['gt_lidar'].x = torch.zeros((len(trans_xy), 1), dtype=torch.float)
        else:
            data['gt_lidar'].pos = torch.empty((0, 2))
            data['gt_lidar'].x = torch.empty((0, 1))

        # (2) Radar GT (Radar1 + Radar2 Merged for Ghost Filtering)
        r_next_list = []
        for rdf in [self.radar1_df, self.radar2_df]:
            r_val = rdf[rdf['t'].round(1) == round(next_t, 1)].values
            if len(r_val) > 0: r_next_list.append(r_val[:, 1:3])
        
        if r_next_list:
            r_next_all = np.vstack(r_next_list)
            xy_h = np.hstack([r_next_all, np.ones((len(r_next_all), 1))])
            trans_xy = (T_next_to_base @ xy_h.T).T[:, :2] / self.scale_pose
            
            data['gt_radar'].pos = torch.tensor(trans_xy, dtype=torch.float)
            data['gt_radar'].x = torch.zeros((len(trans_xy), 1), dtype=torch.float)
        else:
            data['gt_radar'].pos = torch.empty((0, 2))
            data['gt_radar'].x = torch.empty((0, 1))
            
        # Delta Time (Scalar)
        data['dt_sec'] = torch.tensor([round(next_t - base_t, 1)], dtype=torch.float)

        return data

def get_selected_dataset(root, vers, win):
    return SensorFusionDataset(root, vers, win)