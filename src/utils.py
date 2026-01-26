import torch
from torch_cluster import radius, radius_graph


def compute_edge_attr(src_idx, dst_idx, pos, dt, radius_val, device):
    """
    Compute edge attributes for intra-type edges.

    Args:
        src_idx, dst_idx: source/destination node indices (global indices in this node-type)
        pos: node positions (N, 2)
        dt: node time stamp (N,) (e.g., 0.0, 0.1, 0.2, 0.3)
        radius_val: normalization radius (float)
        device: torch device

    Returns:
        edge_attr: (E, 4) -> [dx_norm, dy_norm, dist_norm, dt_diff]
    """
    rel_pos = pos[src_idx] - pos[dst_idx]  # (E, 2)
    dist = torch.norm(rel_pos, dim=1, keepdim=True)  # (E, 1)
    dt_diff = (dt[src_idx] - dt[dst_idx]).unsqueeze(1)  # (E, 1)

    safe_r = radius_val if radius_val > 0 else 1.0

    edge_attr = torch.cat(
        [rel_pos / safe_r, dist / safe_r, dt_diff],
        dim=1,
    ).to(device)

    return edge_attr


def build_graph(data, r_ll, r_rr, r_cross, r_t_ll, r_t_rr, max_Lnum, max_Rnum, device):
    """
    Build hetero graph edges for spatial/temporal intra-sensor edges and inter-sensor edges.

    Notes:
        - Uses float-safe dt matching: |dt - t_val| < 0.05
        - Temporal edges use sensor-specific neighbor limits (k_limit).
    """
    edge_index_dict = {}
    edge_attr_dict = {}

    # 1) Intra-sensor edges (Spatial & Temporal)
    for nt in ["lidar", "radar1", "radar2"]:
        if data[nt].x.size(0) == 0:
            continue

        pos = data[nt].pos
        dt = data[nt].x[:, -1]
        batch = data[nt].batch

        # Sensor-specific parameters
        s_radius = r_ll if nt == "lidar" else r_rr
        t_radius = r_t_ll if nt == "lidar" else r_t_rr
        k_limit = max_Lnum if nt == "lidar" else max_Rnum

        # --- A) Spatial edges ---
        spatial_indices = []
        spatial_attrs = []

        for t_val in [0.0, 0.1, 0.2, 0.3]:
            mask = (dt - t_val).abs() < 0.05
            if mask.sum() > 1:
                idx = torch.where(mask)[0]
                p_masked = pos[mask]
                b_masked = batch[mask]

                e = radius_graph(
                    p_masked,
                    r=s_radius,
                    batch=b_masked,
                    loop=False,
                    max_num_neighbors=k_limit,
                )

                if e.numel() > 0:
                    src_global = idx[e[0]]
                    dst_global = idx[e[1]]
                    spatial_indices.append(torch.stack([src_global, dst_global], dim=0))
                    attr = compute_edge_attr(src_global, dst_global, pos, dt, s_radius, device)
                    spatial_attrs.append(attr)

        if spatial_indices:
            edge_index_dict[(nt, "spatial", nt)] = torch.cat(spatial_indices, dim=1).to(device)
            edge_attr_dict[(nt, "spatial", nt)] = torch.cat(spatial_attrs, dim=0).to(device)

        # --- B) Temporal edges ---
        temporal_indices = []
        temporal_attrs = []

        for t_now, t_prev in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]:
            mask_now = (dt - t_now).abs() < 0.05
            mask_prev = (dt - t_prev).abs() < 0.05

            if mask_now.any() and mask_prev.any():
                idx_now = torch.where(mask_now)[0]
                idx_prev = torch.where(mask_prev)[0]
                p_now, b_now = pos[mask_now], batch[mask_now]
                p_prev, b_prev = pos[mask_prev], batch[mask_prev]

                # Forward: prev -> now
                e_fwd = radius(
                    x=p_now,
                    y=p_prev,
                    r=t_radius,
                    batch_x=b_now,
                    batch_y=b_prev,
                    max_num_neighbors=k_limit,
                )
                if e_fwd.numel() > 0:
                    src = idx_prev[e_fwd[0]]
                    dst = idx_now[e_fwd[1]]
                    temporal_indices.append(torch.stack([src, dst], dim=0))
                    attr = compute_edge_attr(src, dst, pos, dt, t_radius, device)
                    temporal_attrs.append(attr)

                # Backward: now -> prev
                e_bwd = radius(
                    x=p_prev,
                    y=p_now,
                    r=t_radius,
                    batch_x=b_prev,
                    batch_y=b_now,
                    max_num_neighbors=k_limit,
                )
                if e_bwd.numel() > 0:
                    src = idx_now[e_bwd[0]]
                    dst = idx_prev[e_bwd[1]]
                    temporal_indices.append(torch.stack([src, dst], dim=0))
                    attr = compute_edge_attr(src, dst, pos, dt, t_radius, device)
                    temporal_attrs.append(attr)

        if temporal_indices:
            edge_index_dict[(nt, "temporal", nt)] = torch.cat(temporal_indices, dim=1).to(device)
            edge_attr_dict[(nt, "temporal", nt)] = torch.cat(temporal_attrs, dim=0).to(device)

    # 2) Inter-sensor edges (Radar <-> LiDAR)
    for r_nt in ["radar1", "radar2"]:
        if data["lidar"].x.size(0) > 0 and data[r_nt].x.size(0) > 0:
            f_r2l_idx, f_r2l_attr = [], []
            f_l2r_idx, f_l2r_attr = [], []

            dt_l = data["lidar"].x[:, -1]
            dt_r = data[r_nt].x[:, -1]
            batch_l = data["lidar"].batch
            batch_r = data[r_nt].batch
            pos_l = data["lidar"].pos
            pos_r = data[r_nt].pos

            for t_val in [0.0, 0.1, 0.2, 0.3]:
                m_l = (dt_l - t_val).abs() < 0.05
                m_r = (dt_r - t_val).abs() < 0.05

                if m_l.any() and m_r.any():
                    idx_l = torch.where(m_l)[0]
                    idx_r = torch.where(m_r)[0]
                    p_l, b_l = pos_l[m_l], batch_l[m_l]
                    p_r, b_r = pos_r[m_r], batch_r[m_r]

                    # Radar -> LiDAR
                    e_r2l = radius(
                        x=p_l,
                        y=p_r,
                        r=r_cross,
                        batch_x=b_l,
                        batch_y=b_r,
                        max_num_neighbors=max_Rnum,
                    )
                    if e_r2l.numel() > 0:
                        src = idx_r[e_r2l[0]]
                        dst = idx_l[e_r2l[1]]
                        f_r2l_idx.append(torch.stack([src, dst], dim=0))

                        rel_pos = pos_r[src] - pos_l[dst]
                        dist = torch.norm(rel_pos, dim=1, keepdim=True)
                        dt_diff = (dt_r[src] - dt_l[dst]).unsqueeze(1)
                        safe_r = r_cross if r_cross > 0 else 1.0
                        attr = torch.cat([rel_pos / safe_r, dist / safe_r, dt_diff], dim=1).to(device)
                        f_r2l_attr.append(attr)

                    # LiDAR -> Radar
                    e_l2r = radius(
                        x=p_r,
                        y=p_l,
                        r=r_cross,
                        batch_x=b_r,
                        batch_y=b_l,
                        max_num_neighbors=max_Rnum,
                    )
                    if e_l2r.numel() > 0:
                        src = idx_l[e_l2r[0]]
                        dst = idx_r[e_l2r[1]]
                        f_l2r_idx.append(torch.stack([src, dst], dim=0))

                        rel_pos = pos_l[src] - pos_r[dst]
                        dist = torch.norm(rel_pos, dim=1, keepdim=True)
                        dt_diff = (dt_l[src] - dt_r[dst]).unsqueeze(1)
                        safe_r = r_cross if r_cross > 0 else 1.0
                        attr = torch.cat([rel_pos / safe_r, dist / safe_r, dt_diff], dim=1).to(device)
                        f_l2r_attr.append(attr)

            if f_r2l_idx:
                edge_index_dict[(r_nt, "to", "lidar")] = torch.cat(f_r2l_idx, dim=1).to(device)
                edge_attr_dict[(r_nt, "to", "lidar")] = torch.cat(f_r2l_attr, dim=0).to(device)

            if f_l2r_idx:
                edge_index_dict[("lidar", "to", r_nt)] = torch.cat(f_l2r_idx, dim=1).to(device)
                edge_attr_dict[("lidar", "to", r_nt)] = torch.cat(f_l2r_attr, dim=0).to(device)

    return edge_index_dict, edge_attr_dict


def apply_masking(batch, mask_ratio=0.3, lidar_pos_noise_std=0.006):
    """
    Self-supervised masking (always masks both LiDAR and Radar).

    LiDAR:
        - Mask feature x,y -> set to 0
        - Inject small noise into pos to prevent leakage via graph connectivity

    Radar:
        - Mask feature vr -> set to 0 (pos kept)

    Args:
        batch: PyG HeteroData Batch (modified in-place)
        mask_ratio: masking ratio among dt==0 nodes
        lidar_pos_noise_std: std for LiDAR pos noise in normalized units

    Returns:
        batch: masked batch (in-place)
        mask_dict: {node_type: indices}
        gt_dict: {node_type: original_values}
    """
    mask_dict = {}
    gt_dict = {}

    # 1) LiDAR masking (target: x,y -> indices 0,1)
    if "lidar" in batch.node_types:
        x = batch["lidar"].x
        current_mask = (x[:, -1].abs() < 0.05)
        current_indices = torch.where(current_mask)[0]

        num_mask = int(len(current_indices) * mask_ratio)
        if num_mask > 0:
            perm = torch.randperm(len(current_indices), device=current_indices.device)
            selected_idx = current_indices[perm[:num_mask]]

            gt_values = x[selected_idx, :2].clone()

            batch["lidar"].x[selected_idx, 0] = 0.0
            batch["lidar"].x[selected_idx, 1] = 0.0

            noise = torch.randn_like(batch["lidar"].pos[selected_idx]) * lidar_pos_noise_std
            batch["lidar"].pos[selected_idx] += noise

            mask_dict["lidar"] = selected_idx
            gt_dict["lidar"] = gt_values

    # 2) Radar masking (target: vr -> index 2)
    for r_type in ["radar1", "radar2"]:
        if r_type in batch.node_types:
            x = batch[r_type].x
            current_mask = (x[:, -1].abs() < 0.05)
            current_indices = torch.where(current_mask)[0]

            num_mask = int(len(current_indices) * mask_ratio)
            if num_mask > 0:
                perm = torch.randperm(len(current_indices), device=current_indices.device)
                selected_idx = current_indices[perm[:num_mask]]

                gt_values = x[selected_idx, 2:3].clone()
                batch[r_type].x[selected_idx, 2] = 0.0

                mask_dict[r_type] = selected_idx
                gt_dict[r_type] = gt_values

    return batch, mask_dict, gt_dict
