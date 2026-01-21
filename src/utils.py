import torch
from torch_cluster import radius, radius_graph

def build_graph(data, r_ll, r_rr, r_cross, r_t_ll, r_t_rr, device):
    """
    Builds a Spatio-Temporal Graph with Dynamic Boundary Guarding.
    Accommodates variable point counts across frames by clamping indices.
    """
    edge_dict = {}
    
    # 1. Intra-sensor Edges (Spatial & Temporal)
    for nt in ['lidar', 'radar1', 'radar2']:
        if data[nt].x.size(0) == 0: continue
        
        pos = data[nt].pos
        # dt_bin: [0.0, 0.1, 0.2, 0.3] (Last column of node features)
        dt = data[nt].x[:, -1] 
        
        s_radius = r_ll if nt == 'lidar' else r_rr
        t_radius = r_t_ll if nt == 'lidar' else r_t_rr
        
        # --- [A] Spatial Edges (Same Frame: dt_i == dt_j) ---
        spatial_indices = []
        for t_val in [0.0, 0.1, 0.2, 0.3]:
            # Use a tolerance margin for floating point comparison
            mask = (dt - t_val).abs() < 0.05
            if mask.sum() > 1:
                idx = torch.where(mask)[0]
                # Ensure memory contiguity before GPU kernel
                p = pos[mask].contiguous()
                
                if torch.isfinite(p).all():
                    e = radius_graph(p, r=s_radius, loop=False)
                    if e.numel() > 0:
                        # [Dynamic Guard] Clamp to current frame's point count
                        max_idx = idx.size(0) - 1
                        e = e.clamp(min=0, max=max_idx)
                        spatial_indices.append(torch.stack([idx[e[0]], idx[e[1]]], dim=0))
        
        if spatial_indices:
            edge_dict[(nt, 'spatial', nt)] = torch.cat(spatial_indices, dim=1).to(device)

        # --- [B] Temporal Edges (Adjacent Frames: |dt_i - dt_j| == 0.1) ---
        temporal_indices = []
        for t_now, t_prev in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]:
            mask_now = (dt - t_now).abs() < 0.05
            mask_prev = (dt - t_prev).abs() < 0.05
            
            if mask_now.any() and mask_prev.any():
                idx_now = torch.where(mask_now)[0]
                idx_prev = torch.where(mask_prev)[0]
                
                p_now = pos[mask_now].contiguous()
                p_prev = pos[mask_prev].contiguous()
                
                if torch.isfinite(p_now).all() and torch.isfinite(p_prev).all():
                    # Local boundaries for clamping
                    max_now = idx_now.size(0) - 1
                    max_prev = idx_prev.size(0) - 1
                    
                    # Forward: prev -> now
                    e_fwd = radius(p_now, p_prev, r=t_radius)
                    if e_fwd.numel() > 0:
                        e_fwd[0] = e_fwd[0].clamp(min=0, max=max_now)
                        e_fwd[1] = e_fwd[1].clamp(min=0, max=max_prev)
                        temporal_indices.append(torch.stack([idx_prev[e_fwd[1]], idx_now[e_fwd[0]]]))
                    
                    # Backward: now -> prev
                    e_bwd = radius(p_prev, p_now, r=t_radius)
                    if e_bwd.numel() > 0:
                        e_bwd[0] = e_bwd[0].clamp(min=0, max=max_prev)
                        e_bwd[1] = e_bwd[1].clamp(min=0, max=max_now)
                        temporal_indices.append(torch.stack([idx_now[e_bwd[1]], idx_prev[e_bwd[0]]]))
                
        if temporal_indices:
            edge_dict[(nt, 'temporal', nt)] = torch.cat(temporal_indices, dim=1).to(device)

    # 2. Inter-sensor Fusion Edges (Same Frame Only)
    for r_nt in ['radar1', 'radar2']:
        if data['lidar'].x.size(0) > 0 and data[r_nt].x.size(0) > 0:
            f_r2l, f_l2r = [], []
            dt_l, dt_r = data['lidar'].x[:, -1], data[r_nt].x[:, -1]
            
            for t_val in [0.0, 0.1, 0.2, 0.3]:
                m_l, m_r = (dt_l - t_val).abs() < 0.05, (dt_r - t_val).abs() < 0.05
                if m_l.any() and m_r.any():
                    idx_l, idx_r = torch.where(m_l)[0], torch.where(m_r)[0]
                    p_l, p_r = data['lidar'].pos[m_l].contiguous(), data[r_nt].pos[m_r].contiguous()
                    
                    if torch.isfinite(p_l).all() and torch.isfinite(p_r).all():
                        max_l, max_r = idx_l.size(0) - 1, idx_r.size(0) - 1
                        
                        # Radar to LiDAR
                        e_r2l = radius(p_l, p_r, r=r_cross)
                        if e_r2l.numel() > 0:
                            e_r2l[0] = e_r2l[0].clamp(min=0, max=max_l)
                            e_r2l[1] = e_r2l[1].clamp(min=0, max=max_r)
                            f_r2l.append(torch.stack([idx_r[e_r2l[1]], idx_l[e_r2l[0]]]))
                        
                        # LiDAR to Radar
                        e_l2r = radius(p_r, p_l, r=r_cross)
                        if e_l2r.numel() > 0:
                            e_l2r[0] = e_l2r[0].clamp(min=0, max=max_r)
                            e_l2r[1] = e_l2r[1].clamp(min=0, max=max_l)
                            f_l2r.append(torch.stack([idx_l[e_l2r[1]], idx_r[e_l2r[0]]]))
            
            if f_r2l: edge_dict[(r_nt, 'to', 'lidar')] = torch.cat(f_r2l, dim=1).to(device)
            if f_l2r: edge_dict[('lidar', 'to', r_nt)] = torch.cat(f_l2r, dim=1).to(device)

    return edge_dict