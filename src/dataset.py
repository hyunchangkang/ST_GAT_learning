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
        
        # Timestamp binning (0.1s interval) to handle jitter
        self.ts = sorted(self.lidar_df['t'].round(1).unique())
        
        # [수정] Valid indices
        # 기존: range(window_size - 1, len(self.ts) - 1) -> t+1을 위해 마지막 하나를 남겨둠
        # 변경: range(window_size - 1, len(self.ts))     -> t 시점 복원이므로 마지막 프레임까지 다 씀
        self.indices = range(window_size - 1, len(self.ts))

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
        base_t = self.ts[current_idx]      # Current frame for coordinate base (t)
        
        # [수정] target_t (t+1) 삭제됨
        
        window_ts = self.ts[current_idx - self.window_size + 1 : current_idx + 1]
        T_inv_base = np.linalg.inv(self.get_T(base_t))
        
        data = HeteroData()

        # Input data construction (t-3 ~ t)
        for key, df, dim in [('lidar', self.lidar_df, 4), ('radar1', self.radar1_df, 5), ('radar2', self.radar2_df, 5)]:
            pts_list = []
            for t_bin in window_ts:
                curr = df[df['t'].round(1) == round(t_bin, 1)].values
                if len(curr) > 0:
                    # Align all past frames to current base_t coordinate
                    T_rel = T_inv_base @ self.get_T(t_bin)
                    xy_h = np.hstack([curr[:, 1:3], np.ones((len(curr), 1))])
                    trans_xy = (T_rel @ xy_h.T).T[:, :2] / self.scale_pose 
                    
                    dt_val = round(base_t - t_bin, 1) # Relative time (0.0, 0.1, 0.2, 0.3)
                    features = curr[:, 3:].copy()
                    
                    # Feature scaling (No masking applied to current frame dt=0)
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
                # dt info is stored in the last column of x for edge filtering in utils.py
            else:
                data[key].x, data[key].pos = torch.empty((0, dim)), torch.empty((0, 2))

        # [수정] Ground Truth construction (t+1) 로직 전체 삭제
        # 이제 모델은 t 시점의 데이터를 마스킹하고 복원하는 것이 목표이므로, 
        # t+1 데이터는 필요하지 않음.
        
        return data

def get_selected_dataset(root, vers, win):
    return SensorFusionDataset(root, vers, win)