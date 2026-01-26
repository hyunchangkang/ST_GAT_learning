import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, BatchNorm
from torch_geometric.data import HeteroData
from src.utils import build_graph

class ST_HGAT(nn.Module):
    def __init__(self, hidden_dim, num_layers, heads, metadata, node_in_dims, 
                 radius_ll=0.25, radius_rr=1.5, radius_cross=0.6, 
                 temporal_radius_ll=0.6, temporal_radius_rr=0.8,
                 max_num_neighbors_lidar=16, max_num_neighbors_radar=32,
                 # 물리적 제약 인자들
                 min_sigma_lidar_m=0.05, max_sigma_lidar_m=2.0,
                 min_sigma_radar_v=0.15, max_sigma_radar_v=5.0,
                 scale_pose=10.0, scale_radar_v=5.0):
        
        super(ST_HGAT, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.node_types, self.edge_types = metadata

        self.radius_ll = radius_ll
        self.radius_rr = radius_rr
        self.radius_cross = radius_cross
        self.temporal_radius_ll = temporal_radius_ll
        self.temporal_radius_rr = temporal_radius_rr
        self.max_Lnum = max_num_neighbors_lidar
        self.max_Rnum = max_num_neighbors_radar

        # Scaling Factors & Limits
        self.min_l_norm = min_sigma_lidar_m / scale_pose
        self.max_l_norm = max_sigma_lidar_m / scale_pose
        self.range_l = self.max_l_norm - self.min_l_norm

        self.min_r_norm = min_sigma_radar_v / scale_radar_v
        self.max_r_norm = max_sigma_radar_v / scale_radar_v
        self.range_r = self.max_r_norm - self.min_r_norm

        # 1. Initial Feature Projections
        self.proj_dict = nn.ModuleDict()
        for nt in self.node_types:
            self.proj_dict[nt] = nn.Linear(node_in_dims[nt], hidden_dim)

        # 2. Heterogeneous GAT Layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                # [핵심 수정] (-1, -1) -> hidden_dim 으로 변경!
                # 이렇게 하면 모델 생성 시점에 바로 가중치 행렬이 [64, 64]로 만들어집니다.
                conv_dict[edge_type] = GATv2Conv(
                    hidden_dim,         
                    hidden_dim // heads, 
                    heads=heads, 
                    add_self_loops=False,
                    edge_dim=4  
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
            bn_dict = nn.ModuleDict()
            for nt in self.node_types:
                bn_dict[nt] = BatchNorm(hidden_dim)
            self.batch_norms.append(bn_dict)

        # 3. Reconstruction Heads
        self.head_lidar = nn.Linear(hidden_dim, 3) 
        self.head_radar = nn.Linear(hidden_dim, 2)

    def forward(self, data: HeteroData):
        # [Step 1] Graph Construction
        edge_index_dict, edge_attr_dict = build_graph(
            data, 
            self.radius_ll, self.radius_rr, self.radius_cross,
            self.temporal_radius_ll, self.temporal_radius_rr,
            self.max_Lnum, self.max_Rnum,
            data['lidar'].x.device
        )

        # [Step 2] Projection
        x_dict = {}
        for nt in self.node_types:
            x_dict[nt] = F.elu(self.proj_dict[nt](data[nt].x))

        # [Step 3] Message Passing
        for i in range(self.num_layers):
            x_dict = self.convs[i](
                x_dict, 
                edge_index_dict, 
                edge_attr_dict=edge_attr_dict 
            )
            
            for nt in self.node_types:
                x_dict[nt] = F.elu(x_dict[nt])
                x_dict[nt] = F.dropout(x_dict[nt], p=0.0, training=self.training)
                x_dict[nt] = self.batch_norms[i][nt](x_dict[nt])

        # [Step 4] Prediction with Scaled Sigmoid
        
        # --- LiDAR Head ---
        out_l = self.head_lidar(x_dict['lidar'])
        mu_l = out_l[:, :2] 
        sig_l = self.range_l * torch.sigmoid(out_l[:, 2:]) + self.min_l_norm

        # --- Radar1 Head ---
        out_r1 = self.head_radar(x_dict['radar1'])
        mu_r1 = out_r1[:, :1]
        sig_r1 = self.range_r * torch.sigmoid(out_r1[:, 1:]) + self.min_r_norm

        # --- Radar2 Head ---
        out_r2 = self.head_radar(x_dict['radar2'])
        mu_r2 = out_r2[:, :1]
        sig_r2 = self.range_r * torch.sigmoid(out_r2[:, 1:]) + self.min_r_norm

        return mu_l, sig_l, mu_r1, sig_r1, mu_r2, sig_r2