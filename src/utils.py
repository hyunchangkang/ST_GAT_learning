import torch
from torch_cluster import radius, radius_graph

def compute_edge_attr(src_idx, dst_idx, pos, dt, radius_val, device):
    """
    Compute edge attributes for intra-type edges.
   
    """
    rel_pos = pos[src_idx] - pos[dst_idx]
    dist = torch.norm(rel_pos, dim=1, keepdim=True)
    dt_diff = (dt[src_idx] - dt[dst_idx]).unsqueeze(1)

    safe_r = radius_val if radius_val > 0 else 1.0
    edge_attr = torch.cat([rel_pos / safe_r, dist / safe_r, dt_diff], dim=1).to(device)
    return edge_attr

def build_graph(data, r_ll, r_rr, r_cross, r_t_ll, r_t_rr, max_Lnum, max_Rnum, device):
    """
    Build hetero graph edges dynamically based on unique timestamps.
   
    """
    edge_index_dict = {}
    edge_attr_dict = {}

    # 1) Intra-sensor edges (Spatial & Temporal)
    for nt in ["lidar", "radar1", "radar2"]:
        if data[nt].x.size(0) == 0: continue

        pos, dt_col, batch = data[nt].pos, data[nt].x[:, -1], data[nt].batch
        unique_ts = torch.unique(dt_col) # [Modified] Dynamic time extraction

        s_radius = r_ll if nt == "lidar" else r_rr
        t_radius = r_t_ll if nt == "lidar" else r_t_rr
        k_limit = max_Lnum if nt == "lidar" else max_Rnum

        # --- A) Spatial edges ---
        spatial_indices, spatial_attrs = [], []
        for t_val in unique_ts:
            mask = (dt_col - t_val).abs() < 0.05
            if mask.sum() > 1:
                idx = torch.where(mask)[0]
                e = radius_graph(pos[mask], r=s_radius, batch=batch[mask], loop=False, max_num_neighbors=k_limit)
                if e.numel() > 0:
                    src, dst = idx[e[0]], idx[e[1]]
                    spatial_indices.append(torch.stack([src, dst], dim=0))
                    spatial_attrs.append(compute_edge_attr(src, dst, pos, dt_col, s_radius, device))

        if spatial_indices:
            edge_index_dict[(nt, "spatial", nt)] = torch.cat(spatial_indices, dim=1).to(device)
            edge_attr_dict[(nt, "spatial", nt)] = torch.cat(spatial_attrs, dim=0).to(device)

        # --- B) Temporal edges ---
        temporal_indices, temporal_attrs = [], []
        sorted_ts = torch.sort(unique_ts, descending=True)[0] # Sort: [0.3, 0.2, 0.1, 0.0]
        for i in range(len(sorted_ts) - 1):
            t_prev, t_now = sorted_ts[i], sorted_ts[i+1] # [Modified] Dynamic pair
            m_now, m_prev = (dt_col - t_now).abs() < 0.05, (dt_col - t_prev).abs() < 0.05
            if m_now.any() and m_prev.any():
                idx_now, idx_prev = torch.where(m_now)[0], torch.where(m_prev)[0]
                # Forward: prev -> now (src: prev, dst: now)
                e_fwd = radius(x=pos[m_now], y=pos[m_prev], r=t_radius, batch_x=batch[m_now], batch_y=batch[m_prev], max_num_neighbors=k_limit)
                if e_fwd.numel() > 0:
                    src, dst = idx_prev[e_fwd[0]], idx_now[e_fwd[1]]
                    temporal_indices.append(torch.stack([src, dst], dim=0))
                    temporal_attrs.append(compute_edge_attr(src, dst, pos, dt_col, t_radius, device))
                # Backward: now -> prev (src: now, dst: prev)
                e_bwd = radius(x=pos[m_prev], y=pos[m_now], r=t_radius, batch_x=batch[m_prev], batch_y=batch[m_now], max_num_neighbors=k_limit)
                if e_bwd.numel() > 0:
                    src, dst = idx_now[e_bwd[0]], idx_prev[e_bwd[1]]
                    temporal_indices.append(torch.stack([src, dst], dim=0))
                    temporal_attrs.append(compute_edge_attr(src, dst, pos, dt_col, t_radius, device))

        if temporal_indices:
            edge_index_dict[(nt, "temporal", nt)] = torch.cat(temporal_indices, dim=1).to(device)
            edge_attr_dict[(nt, "temporal", nt)] = torch.cat(temporal_attrs, dim=0).to(device)

    # 2) Inter-sensor edges (Radar <-> LiDAR)
    for r_nt in ["radar1", "radar2"]:
        if data["lidar"].x.size(0) > 0 and data[r_nt].x.size(0) > 0:
            f_r2l_idx, f_r2l_attr, f_l2r_idx, f_l2r_attr = [], [], [], []
            dt_l, dt_r = data["lidar"].x[:, -1], data[r_nt].x[:, -1]
            pos_l, pos_r = data["lidar"].pos, data[r_nt].pos
            batch_l, batch_r = data["lidar"].batch, data[r_nt].batch
            combined_ts = torch.unique(torch.cat([dt_l, dt_r]))

            for t_val in combined_ts:
                m_l, m_r = (dt_l - t_val).abs() < 0.05, (dt_r - t_val).abs() < 0.05
                if m_l.any() and m_r.any():
                    idx_l, idx_r = torch.where(m_l)[0], torch.where(m_r)[0]
                    # [CUDA Fix] Ensure index mapping consistency
                    # Radar -> LiDAR (src: Radar, dst: LiDAR)
                    e_r2l = radius(x=pos_l[m_l], y=pos_r[m_r], r=r_cross, batch_x=batch_l[m_l], batch_y=batch_r[m_r], max_num_neighbors=max_Rnum)
                    if e_r2l.numel() > 0:
                        src, dst = idx_r[e_r2l[0]], idx_l[e_r2l[1]]
                        f_r2l_idx.append(torch.stack([src, dst], dim=0))
                        rel_pos = pos_r[src] - pos_l[dst]
                        dist = torch.norm(rel_pos, dim=1, keepdim=True)
                        dt_diff = (dt_r[src] - dt_l[dst]).unsqueeze(1)
                        f_r2l_attr.append(torch.cat([rel_pos/r_cross, dist/r_cross, dt_diff], dim=1).to(device))

                    # LiDAR -> Radar (src: LiDAR, dst: Radar)
                    e_l2r = radius(x=pos_r[m_r], y=pos_l[m_l], r=r_cross, batch_x=batch_r[m_r], batch_y=batch_l[m_l], max_num_neighbors=max_Rnum)
                    if e_l2r.numel() > 0:
                        src, dst = idx_l[e_l2r[0]], idx_r[e_l2r[1]]
                        f_l2r_idx.append(torch.stack([src, dst], dim=0))
                        rel_pos = pos_l[src] - pos_r[dst]
                        dist = torch.norm(rel_pos, dim=1, keepdim=True)
                        dt_diff = (dt_l[src] - dt_r[dst]).unsqueeze(1)
                        f_l2r_attr.append(torch.cat([rel_pos/r_cross, dist/r_cross, dt_diff], dim=1).to(device))

            if f_r2l_idx:
                edge_index_dict[(r_nt, "to", "lidar")] = torch.cat(f_r2l_idx, dim=1).to(device)
                edge_attr_dict[(r_nt, "to", "lidar")] = torch.cat(f_r2l_attr, dim=0).to(device)
            if f_l2r_idx:
                edge_index_dict[("lidar", "to", r_nt)] = torch.cat(f_l2r_idx, dim=1).to(device)
                edge_attr_dict[("lidar", "to", r_nt)] = torch.cat(f_l2r_attr, dim=0).to(device)

    return edge_index_dict, edge_attr_dict

def apply_masking(batch, mask_ratio=0.3, lidar_pos_noise_std=0.006):
    """
    Self-supervised masking (Original function restored)
   
    """
    mask_dict, gt_dict = {}, {}
    if "lidar" in batch.node_types:
        x, current_mask = batch["lidar"].x, (batch["lidar"].x[:, -1].abs() < 0.05)
        current_indices = torch.where(current_mask)[0]
        num_mask = int(len(current_indices) * mask_ratio)
        if num_mask > 0:
            perm = torch.randperm(len(current_indices), device=current_indices.device)
            selected_idx = current_indices[perm[:num_mask]]
            gt_dict["lidar"] = x[selected_idx, :2].clone()
            batch["lidar"].x[selected_idx, 0:2] = 0.0
            batch["lidar"].pos[selected_idx] += torch.randn_like(batch["lidar"].pos[selected_idx]) * lidar_pos_noise_std
            mask_dict["lidar"] = selected_idx

    for r_type in ["radar1", "radar2"]:
        if r_type in batch.node_types:
            x, current_mask = batch[r_type].x, (batch[r_type].x[:, -1].abs() < 0.05)
            current_indices = torch.where(current_mask)[0]
            num_mask = int(len(current_indices) * mask_ratio)
            if num_mask > 0:
                perm = torch.randperm(len(current_indices), device=current_indices.device)
                selected_idx = current_indices[perm[:num_mask]]
                gt_dict[r_type] = x[selected_idx, 2:3].clone()
                batch[r_type].x[selected_idx, 2] = 0.0
                mask_dict[r_type] = selected_idx
    return batch, mask_dict, gt_dict