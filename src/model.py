import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear

class ST_HGAT(nn.Module):
    def __init__(self, hidden_dim, num_layers, heads, metadata, node_in_dims):
        """
        ST-HGAT for Dynamic Uncertainty Estimation.
        - metadata: (node_types, edge_types)
        - node_in_dims: dict of input dimensions for each node type
        """
        super(ST_HGAT, self).__init__()
        
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        # 1. Sensor-specific Encoders
        self.encoder = nn.ModuleDict()
        for node_type in self.node_types:
            self.encoder[node_type] = nn.Sequential(
                Linear(node_in_dims[node_type], hidden_dim),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden_dim)
            )

        # 2. Heterogeneous GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                src, _, dst = edge_type
                # Self-loops help preserve node identity during message passing
                self_loop = True if src == dst else False
                conv_dict[edge_type] = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    add_self_loops=self_loop
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # 3. Prediction Heads (Mu and Sigma)
        # LiDAR: predicts (x, y) and sigma_pos
        self.l_head = nn.Sequential(
            Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            Linear(32, 3) 
        )
        # Radar: predicts v_radial and sigma_vel
        self.r_head = nn.Sequential(
            Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            Linear(32, 2)
        )

    def forward(self, x_dict, edge_index_dict):
        # Initial projection
        out_dict = {}
        for nt, x in x_dict.items():
            if x.size(0) > 0:
                out_dict[nt] = self.encoder[nt](x)

        # Spatio-Temporal Message Passing
        for conv in self.convs:
            out_dict = conv(out_dict, edge_index_dict)
            out_dict = {nt: F.leaky_relu(x) for nt, x in out_dict.items()}

        device = next(self.parameters()).device

        # # --- [Debug Section: Check instantly on first batch] ---
        # if 'lidar' in x_dict and x_dict['lidar'].size(0) > 0:
        #     unique_times = torch.unique(x_dict['lidar'][:, -1]).tolist()
        #     # Force print for every batch to find the root cause immediately
        #     print(f"\n[Model Debug] Step check - Unique dt: {unique_times}")
            
        #     # Check if 0.0 exists with tolerance
        #     has_target = any(abs(t - 0.0) < 1e-5 for t in unique_times)
        #     if not has_target:
        #         print(f"[Critical] t=0.0 is NOT found in this batch!")
        # # -------------------------------------------------------
        
        # LiDAR Output: mu_l (x, y), sigma_l
        if 'lidar' in out_dict and out_dict['lidar'].size(0) > 0:
            res_l = self.l_head(out_dict['lidar'])
            mu_l = res_l[:, :2]
            sigma_l = F.softplus(res_l[:, 2:3]) + 1e-4
        else:
            mu_l = torch.empty((0, 2), device=device)
            sigma_l = torch.empty((0, 1), device=device)
        
        # Radar Output Processing
        def process_radar(key):
            if key in out_dict and out_dict[key].size(0) > 0:
                res_r = self.r_head(out_dict[key])
                mu_r = res_r[:, 0:1]
                sigma_r = F.softplus(res_r[:, 1:2]) + 1e-4
                return mu_r, sigma_r
            return torch.empty((0, 1), device=device), torch.empty((0, 1), device=device)

        mu_r1, sigma_r1 = process_radar('radar1')
        mu_r2, sigma_r2 = process_radar('radar2')

        return mu_l, sigma_l, mu_r1, sigma_r1, mu_r2, sigma_r2