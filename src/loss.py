import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSupervisedLoss(nn.Module):
    def __init__(self, device, min_sigma_lidar_m=0.03, min_sigma_radar_v=0.1):
        super().__init__()
        self.device = device
        
        # Scaling Factors (dataset.py와 일치)
        self.SCALE_POSE = 10.0      
        self.SCALE_RADAR_V = 5.0    

        # Min Sigma Thresholds (Normalized)
        self.min_sig_l = min_sigma_lidar_m / self.SCALE_POSE
        self.min_sig_r = min_sigma_radar_v / self.SCALE_RADAR_V
        
        # [수정 사항 2] Penalty 삭제
        # model.py에서 이미 Max Sigma 제한을 걸었으므로, 불필요한 규제 제거
        self.penalty_weight = 0.0

    def forward(self, mu_l, sig_l, gt_l, mask_idx_l,
                mu_r1, sig_r1, gt_r1, mask_idx_r1,
                mu_r2, sig_r2, gt_r2, mask_idx_r2):
        
        # 각 센서별로 NLL만 계산하여 반환받음
        l_l = self._lidar_loss(mu_l, sig_l, gt_l, mask_idx_l)
        l_r1 = self._radar_loss(mu_r1, sig_r1, gt_r1, mask_idx_r1)
        l_r2 = self._radar_loss(mu_r2, sig_r2, gt_r2, mask_idx_r2)

        total_loss = l_l + l_r1 + l_r2
        
        # 로깅을 위해 분해된 값 반환
        return total_loss, l_l, l_r1, l_r2

    def _lidar_loss(self, mu, sigma, gt, mask_idx):
        # 마스킹 된 데이터가 없으면 0 반환
        if mask_idx is None or len(mask_idx) == 0:
            return torch.tensor(0.0, device=self.device)

        pred = mu[mask_idx]
        sig = sigma[mask_idx]
        target = gt

        # 수치 안정성을 위한 최소값 제한
        sig = torch.clamp(sig, min=self.min_sig_l)

        # 1. NLL Loss (메인: 불확실성 추론)
        err_sq = (pred - target)**2
        var = sig**2
        log_var = torch.log(var)
        nll_loss = (0.5 * err_sq / var) + (0.5 * log_var)

        return nll_loss.mean()

    def _radar_loss(self, mu, sigma, gt, mask_idx):
        if mask_idx is None or len(mask_idx) == 0:
            return torch.tensor(0.0, device=self.device)

        pred = mu[mask_idx]
        sig = sigma[mask_idx]
        target = gt

        sig = torch.clamp(sig, min=self.min_sig_r)

        # 1. NLL Loss
        err_sq = (pred - target)**2
        var = sig**2
        log_var = torch.log(var)
        nll_loss = (0.5 * err_sq / var) + (0.5 * log_var)

        return nll_loss.mean()
