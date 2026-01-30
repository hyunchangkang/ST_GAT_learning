import torch
import torch.nn as nn
import math
from torch_scatter import scatter_add, scatter_mean

class SpatiotemporalUncertaintyLoss(nn.Module):
    def __init__(self, device, config, penalty_weight=2.0):
        super().__init__()
        self.device = device
        
        # Scaling Factors (Matching dataset.py)
        self.SCALE_POSE = 10.0
        self.SCALE_RADAR_V = 5.0
        
        # [Config] Master Weights
        self.w_lidar = float(config.get('lambda_lidar_loss', 1.0))
        self.w_radar = float(config.get('lambda_radar_loss', 1.0))
        
        # [Config] LiDAR Decomposed Weights
        self.w_l_spatial = float(config.get('lambda_lidar_spatial_loss', 1.0))
        self.w_l_intensity = float(config.get('lambda_lidar_intensity_loss', 1.0))
        
        # [Config] Radar Component Weights
        self.w_r_temp = float(config.get('lambda_radar_temporal_loss', 1.0))
        self.w_r_spat = float(config.get('lambda_radar_spatial_loss', 1.0)) 
        
        # [Config] Search Parameters
        self.K_NEIGHBORS = int(config.get('spatial_k_neighbors', 5))
        self.radius_cross = float(config.get('radius_cross', 0.6)) / self.SCALE_POSE 

        # [Clamping] Pre-compute log limits to avoid repeated math
        min_l = float(config.get('min_sigma_lidar_m', 0.03)) 
        max_l = float(config.get('max_sigma_lidar_m', 0.5))
        self.L_MIN = 2 * math.log(min_l / self.SCALE_POSE + 1e-9)
        self.L_MAX = 2 * math.log(max_l / self.SCALE_POSE + 1e-9)

        min_r = float(config.get('min_sigma_radar_v', 0.10)) 
        max_r = float(config.get('max_sigma_radar_v', 5.0))
        self.R_MIN = 2 * math.log(min_r / self.SCALE_RADAR_V + 1e-9)
        self.R_MAX = 2 * math.log(max_r / self.SCALE_RADAR_V + 1e-9)

        self.GHOST_PENALTY_VAL = (self.radius_cross ** 2)

        print(f"[Loss] Final Decomposed Logic Initialized.")

    def forward(self, outputs, batch, edge_index_dict):
        total_loss = 0
        log_metrics = {}
        
        # 1. LiDAR Loss: Spatio-Intensity Decomposition
        if 'lidar' in batch.node_types and batch['lidar'].x.size(0) > 0:
            log_var_pos = torch.clamp(outputs['lidar'], min=self.L_MIN, max=self.L_MAX)
            curr_pos, curr_int = batch['lidar'].pos, batch['lidar'].x[:, 2:3]
            
            spatial_edge_key = ('lidar', 'spatial', 'lidar')
            if spatial_edge_key in edge_index_dict:
                edge_index = edge_index_dict[spatial_edge_key]
                if edge_index.numel() > 0:
                    src, dst = edge_index[0], edge_index[1]
                    num_nodes = curr_pos.size(0)
                    
                    # Trend Computation
                    mean_pos = scatter_mean(curr_pos[src], dst, dim=0, dim_size=num_nodes)
                    mean_int = scatter_mean(curr_int[src], dst, dim=0, dim_size=num_nodes)
                    
                    # Residual Calculation
                    res_pos = torch.sum((curr_pos - mean_pos)**2, dim=1, keepdim=True)
                    res_int = (curr_int - mean_int)**2
                    
                    precision = torch.exp(-log_var_pos)
                    
                    # Individual Term Calculation
                    # We keep terms separated for logging before weighting
                    l_spat_term = torch.mean(0.5 * precision * res_pos)
                    l_int_term = torch.mean(0.5 * precision * res_int)
                    l_reg_term = torch.mean(0.5 * log_var_pos)
                    
                    # Final LiDAR Integration
                    final_l_loss = ( (self.w_l_spatial * l_spat_term) + 
                                     (self.w_l_intensity * l_int_term) + 
                                     l_reg_term ) * self.w_lidar
                    
                    total_loss += final_l_loss
                    
                    # Metrics for WandB
                    log_metrics['loss_lidar'] = final_l_loss.item()
                    log_metrics['lidar_spatial_loss'] = (l_spat_term * self.w_l_spatial * self.w_lidar).item()
                    log_metrics['lidar_intensity_loss'] = (l_int_term * self.w_l_intensity * self.w_lidar).item()
                    log_metrics['lidar_sigma_mean'] = torch.exp(0.5 * log_var_pos).mean().item()

        # 2. Radar Loss: Physics-based Temporal Consistency
        for r_key in ['radar1', 'radar2']:
            if r_key not in batch.node_types or batch[r_key].x.size(0) == 0: continue
            
            log_var_vel = torch.clamp(outputs[r_key], min=self.R_MIN, max=self.R_MAX)
            curr_pos, curr_batch = batch[r_key].pos, batch[r_key].batch
            node_dt = batch['dt_sec'][curr_batch].unsqueeze(1).clamp(min=0.01)
            
            # [A] Temporal Prediction (t -> t+1)
            physics_err_sq = torch.zeros_like(log_var_vel)
            temp_edge_key = (r_key, 'temporal', r_key)
            if temp_edge_key in edge_index_dict:
                e_idx = edge_index_dict[temp_edge_key]
                src, dst = e_idx[0], e_idx[1]
                # Direction unit vector from t to t+1
                move_vec = curr_pos[dst] - curr_pos[src]
                unit_vec = move_vec / (torch.norm(move_vec, dim=1, keepdim=True) + 1e-9)
                # Prediction using Doppler speed at t
                speed_t = torch.abs(batch[r_key].x[src, 2:3])
                pred_pos_next = curr_pos[src] + (speed_t * unit_vec * node_dt[src])
                
                if 'gt_radar' in batch.node_types and batch['gt_radar'].pos.size(0) > 0:
                    d_mat = torch.cdist(pred_pos_next, batch['gt_radar'].pos)
                    min_d, _ = torch.min(d_mat, dim=1, keepdim=True)
                    physics_err_sq[src] = min_d ** 2

            # [B] Spatial consistency (Ghost filtering)
            spatial_err_sq = torch.ones_like(log_var_vel) * self.GHOST_PENALTY_VAL
            edge_key_l = (r_key, 'to', 'lidar')
            if edge_key_l in edge_index_dict and batch['lidar'].pos.size(0) > 0:
                e_idx = edge_index_dict[edge_key_l]
                src_r, dst_l = e_idx[0], e_idx[1]
                dist_sq = torch.sum((curr_pos[src_r] - batch['lidar'].pos[dst_l])**2, dim=1)
                sum_d = scatter_add(dist_sq, src_r, dim=0, dim_size=curr_pos.size(0))
                cnt_d = scatter_add(torch.ones_like(dist_sq), src_r, dim=0, dim_size=curr_pos.size(0))
                has_n = (cnt_d > 0)
                spatial_err_sq[has_n] = (sum_d[has_n] / (cnt_d[has_n]**2)).unsqueeze(1)

            # [C] Integration
            denominator = 2 * torch.exp(log_var_vel) * (node_dt ** 2) + 1e-9
            r_temp_loss = torch.mean(physics_err_sq / denominator)
            r_spat_loss = torch.mean(spatial_err_sq / denominator)
            r_reg_loss = torch.mean(0.5 * log_var_vel)
            
            final_r_loss = ( (self.w_r_temp * r_temp_loss) + 
                             (self.w_r_spat * r_spat_loss) + 
                             r_reg_loss ) * self.w_radar
            
            total_loss += final_r_loss
            log_metrics[f'{r_key}_loss_total'] = final_r_loss.item()
            log_metrics[f'{r_key}_loss_temporal'] = (r_temp_loss * self.w_r_temp * self.w_radar).item()
            log_metrics[f'{r_key}_loss_spatial'] = (r_spat_loss * self.w_r_spat * self.w_radar).item()
            log_metrics[f'{r_key}_sigma_mean'] = torch.exp(0.5 * log_var_vel).mean().item()

        return total_loss, log_metrics