import torch
import torch.nn as nn
import math

class SelfSupervisedLoss(nn.Module):

    def __init__(self, device, min_sigma_lidar_m=0.02, min_sigma_radar_v=0.1, dist_threshold=1.0):
        """
        [Physical Constraints Setup]
        Accepts physical units (meters, m/s) and converts them to normalized units 
        used by the network training process.
        
        Args:
            device: torch device
            min_sigma_lidar_m (float): Minimum physical position uncertainty for LiDAR (Default: 0.1m = 10cm)
            min_sigma_radar_v (float): Minimum physical velocity uncertainty for Radar (Default: 0.15m/s)
            dist_threshold (float): Distance threshold for matching (m)
        """
        super().__init__()
        self.device = device
        self.dist_threshold = dist_threshold
        
        # --- [Hardcoded Scale Factors from dataset.py] ---
        # Ensure these match the scaling factors in SingleSequenceDataset!
        SCALE_POSE = 10.0      # LiDAR Position Scaling
        SCALE_RADAR_V = 5.0    # Radar Velocity Scaling

        # --- [Conversion: Physical -> Normalized] ---
        # The network predicts normalized values, so we clamp using normalized thresholds.
        self.min_sig_l = min_sigma_lidar_m / SCALE_POSE  # e.g., 0.1m / 10.0 = 0.01
        self.min_sig_r = min_sigma_radar_v / SCALE_RADAR_V # e.g., 0.15m/s / 5.0 = 0.03

    def forward(self, mu_l, log_l, gt_l, mu_r1, log_r1, gt_r1, pos_r1, mu_r2, log_r2, gt_r2, pos_r2):
        # 1. Individual sensor loss calculations (Same as original)
        l_l, n_l = self._lidar_loss(mu_l, log_l, gt_l)
        l_r1, n_r1 = self._radar_loss(mu_r1, log_r1, gt_r1, pos_r1)
        l_r2, n_r2 = self._radar_loss(mu_r2, log_r2, gt_r2, pos_r2)

        # 2. Original Weights: 0.5, 0.25, 0.25
        w_l, w_r1, w_r2 = 0.5, 0.25, 0.25
        
        total_loss = 0.0
        has_loss = False
        
        if n_l > 0:
            total_loss += w_l * l_l
            has_loss = True
        if n_r1 > 0:
            total_loss += w_r1 * l_r1
            has_loss = True
        if n_r2 > 0:
            total_loss += w_r2 * l_r2
            has_loss = True

        if not has_loss:
            return torch.tensor(0.0, device=self.device, requires_grad=True), l_l, l_r1, l_r2
            
        return total_loss, l_l, l_r1, l_r2

    def _lidar_loss(self, mu, log_v, gt):
        """
        Original LiDAR spatial distance loss with new sigma clamping.
        """
        if mu.size(0) == 0 or gt.size(0) == 0: 
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        
        # [Modified] Clamp sigma based on physical value
        sigma = torch.exp(0.5 * log_v)
        sigma = torch.clamp(sigma, min=self.min_sig_l)
        valid_log_v = 2 * torch.log(sigma)
        
        d = torch.cdist(mu, gt)
        min_d, _ = torch.min(d, dim=1)
        
        # Original distance filtering
        mask = min_d < self.dist_threshold
        if not mask.any(): 
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        
        v_log_v = valid_log_v[mask]
        # Predictive NLL formula
        loss = (0.5 * torch.exp(-v_log_v) * (min_d[mask]**2) + 0.5 * v_log_v).mean()
        return loss, mask.sum().item()

    def _radar_loss(self, mu, log_v, gt, pos):
        """
        Original Radar velocity error loss with new sigma clamping.
        """
        if mu.size(0) == 0 or gt.size(0) == 0: 
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        
        # [Modified] Clamp sigma based on physical value
        sigma = torch.exp(0.5 * log_v)
        sigma = torch.clamp(sigma, min=self.min_sig_r)
        valid_log_v = 2 * torch.log(sigma)
        
        # Original anchor-based matching
        d = torch.cdist(pos, gt[:, :2])
        min_d, idx = torch.min(d, dim=1)
        
        mask = min_d < self.dist_threshold
        if not mask.any(): 
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        
        matched_gt_v = gt[idx[mask], 2].unsqueeze(1)
        err = (mu[mask] - matched_gt_v)**2
        v_log_v = valid_log_v[mask]
        loss = (0.5 * torch.exp(-v_log_v) * err + 0.5 * v_log_v).mean()
        return loss, mask.sum().item()