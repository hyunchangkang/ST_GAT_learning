import torch
import torch.nn as nn
import math
from torch_scatter import scatter_add, scatter_mean

class SpatiotemporalUncertaintyLoss(nn.Module):
    def __init__(self, device, config, penalty_weight=2.0):
        super().__init__()
        self.device = device
        
        # Scaling Factors
        self.SCALE_POSE = 10.0
        self.SCALE_RADAR_V = 5.0
        
        # Weights (Lambda)
        self.w_lidar = float(config.get('lambda_lidar_loss', 1.0))
        self.w_radar = float(config.get('lambda_radar_loss', 1.0))
        self.w_l_intensity = float(config.get('lambda_lidar_intensity', 1.0))
        self.w_r_temp = float(config.get('lambda_radar_temporal_loss', 1.0))
        self.w_r_spat = float(config.get('lambda_radar_spatial_loss', 0.1)) 

        # Clamping
        min_l = float(config.get('min_sigma_lidar_m', 0.03)) 
        max_l = float(config.get('max_sigma_lidar_m', 0.2))
        self.L_MIN = 2 * math.log(min_l / self.SCALE_POSE + 1e-9)
        self.L_MAX = 2 * math.log(max_l / self.SCALE_POSE + 1e-9)

        min_r = float(config.get('min_sigma_radar_v', 0.10)) 
        max_r = float(config.get('max_sigma_radar_v', 3.0)) 
        self.R_MIN = 2 * math.log(min_r / self.SCALE_RADAR_V + 1e-9)
        self.R_MAX = 2 * math.log(max_r / self.SCALE_RADAR_V + 1e-9)

        self.GHOST_PENALTY_VAL = 2.0 

        print(f"[Loss] Init Complete. LiDAR: Local Trend Consistency Mode.")

    def forward(self, outputs, batch, dt, edge_index_dict):
        total_loss = 0
        log_metrics = {}
        safe_dt = dt if dt > 0.01 else 0.1
        
        # =========================================================
        # 1. LiDAR Loss: Local Trend Consistency at time t
        # =========================================================
        if 'lidar' in batch.node_types and batch['lidar'].x.size(0) > 0:
            # Model output is only log_var_pos (Sigma)
            log_var_pos = outputs['lidar'] 
            curr_pos = batch['lidar'].pos 
            curr_int = batch['lidar'].x[:, 2:3] # Raw Intensity
            
            log_var_pos = torch.clamp(log_var_pos, min=self.L_MIN, max=self.L_MAX)
            spatial_edge_key = ('lidar', 'spatial', 'lidar')
            
            if spatial_edge_key in edge_index_dict:
                edge_index = edge_index_dict[spatial_edge_key]
                if edge_index.numel() > 0:
                    src, dst = edge_index[0], edge_index[1]
                    num_nodes = curr_pos.size(0)
                    
                    # Compute neighbor means (Trends)
                    mean_pos_nb = scatter_mean(curr_pos[src], dst, dim=0, dim_size=num_nodes)
                    mean_int_nb = scatter_mean(curr_int[src], dst, dim=0, dim_size=num_nodes)
                    
                    # Residuals: Deviation from local geometric & intensity trend
                    spatial_res_sq = torch.sum((curr_pos - mean_pos_nb)**2, dim=1, keepdim=True)
                    intensity_res_sq = (curr_int - mean_int_nb)**2
                    combined_res_sq = spatial_res_sq + (self.w_l_intensity * intensity_res_sq)
                    
                    # Self-supervised NLL Loss
                    precision = torch.exp(-log_var_pos)
                    l_nll = 0.5 * precision * combined_res_sq + 0.5 * log_var_pos
                    
                    weighted_l_loss = torch.mean(l_nll) * self.w_lidar
                    total_loss += weighted_l_loss
                    log_metrics['loss_lidar'] = weighted_l_loss.item()
                    log_metrics['lidar_sigma_mean'] = torch.exp(0.5 * log_var_pos).mean().item()

        # =========================================================
        # 2. Radar Loss
        # =========================================================
        pos_L = batch['lidar'].pos
        
        for r_key in ['radar1', 'radar2']:
            # Radar 데이터가 아예 없는 경우 스킵
            if r_key not in batch.node_types or batch[r_key].x.size(0) == 0: 
                continue
            
            log_var_vel = outputs[r_key] 
            curr_pos = batch[r_key].pos
            curr_batch = batch[r_key].batch
            
            log_var_vel = torch.clamp(log_var_vel, min=self.R_MIN, max=self.R_MAX)
            
            # [A] Temporal
            temp_edge_key = (r_key, 'temporal', r_key)
            direction_vec = torch.zeros_like(curr_pos) 
            
            if temp_edge_key in edge_index_dict:
                edge_index = edge_index_dict[temp_edge_key]
                src_idx, dst_idx = edge_index[0], edge_index[1]
                move_vec = curr_pos[dst_idx] - curr_pos[src_idx]
                dist = torch.norm(move_vec, dim=1, keepdim=True) + 1e-9
                unit_vec = move_vec / dist
                direction_vec = scatter_mean(unit_vec, dst_idx, dim=0, dim_size=curr_pos.size(0))
                dir_norm = torch.norm(direction_vec, dim=1, keepdim=True) + 1e-9
                direction_vec = direction_vec / dir_norm

            raw_doppler = batch[r_key].x[:, 2:3]
            raw_speed = torch.abs(raw_doppler)
            vel_physics = raw_speed * direction_vec
            pos_physics = curr_pos + vel_physics * safe_dt
            
            physics_err_sq = torch.zeros_like(log_var_vel)
            
            # [수정] GT Radar 체크도 'in' 연산자 사용
            if 'gt_radar' in batch.node_types and batch['gt_radar'].pos.size(0) > 0:
                batch_size = int(curr_batch.max().item()) + 1
                for b in range(batch_size):
                    mask_r = (curr_batch == b)
                    mask_g = (batch['gt_radar'].batch == b)
                    p_phy_b = pos_physics[mask_r]
                    g_b = batch['gt_radar'].pos[mask_g]
                    if p_phy_b.size(0) == 0 or g_b.size(0) == 0: continue
                    d_mat = torch.cdist(p_phy_b, g_b)
                    min_d, _ = torch.min(d_mat, dim=1, keepdim=True)
                    physics_err_sq[mask_r] = min_d ** 2

            # [B] Spatial
            spatial_err_sq = torch.ones_like(log_var_vel) * self.GHOST_PENALTY_VAL
            edge_key_l = (r_key, 'to', 'lidar')
            
            if edge_key_l in edge_index_dict:
                edge_index = edge_index_dict[edge_key_l]
                src_r, dst_l = edge_index[0], edge_index[1]
                dist_sq_edges = torch.sum((curr_pos[src_r] - pos_L[dst_l])**2, dim=1)
                num_nodes = curr_pos.size(0)
                sum_dist = scatter_add(dist_sq_edges, src_r, dim=0, dim_size=num_nodes)
                count_n = scatter_add(torch.ones_like(dist_sq_edges), src_r, dim=0, dim_size=num_nodes)
                has_neighbor = (count_n > 0)
                if has_neighbor.any():
                    val = sum_dist[has_neighbor] / (count_n[has_neighbor] ** 2)
                    spatial_err_sq[has_neighbor] = val.unsqueeze(1)

            # [C] Integration
            sigma_v_sq = torch.exp(log_var_vel)
            denominator = 2 * sigma_v_sq * (safe_dt ** 2) + 1e-9
            
            raw_temporal = torch.mean(physics_err_sq / denominator)
            raw_spatial = torch.mean(spatial_err_sq / denominator)
            raw_reg = torch.mean(0.5 * log_var_vel)
            
            weighted_temporal = raw_temporal * self.w_r_temp
            weighted_spatial = raw_spatial * self.w_r_spat
            
            r_loss_inner = weighted_temporal + weighted_spatial + raw_reg
            final_r_loss = r_loss_inner * self.w_radar
            
            total_loss += final_r_loss
            
            log_metrics[f'{r_key}_loss_temporal'] = (weighted_temporal * self.w_radar).item()
            log_metrics[f'{r_key}_loss_spatial'] = (weighted_spatial * self.w_radar).item()
            log_metrics[f'{r_key}_loss_total'] = final_r_loss.item()
            log_metrics[f'{r_key}_sigma_mean'] = torch.exp(0.5 * log_var_vel).mean().item()

        return total_loss, log_metrics