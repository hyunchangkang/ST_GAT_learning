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
                 # YAML 호환성 유지 (Loss에서 쓰지만 모델 초기화 인자로 들어옴)
                 min_sigma_lidar_m=0.05, max_sigma_lidar_m=2.0,
                 min_sigma_radar_v=0.15, max_sigma_radar_v=5.0,
                 scale_pose=10.0, scale_radar_v=5.0, **kwargs):
        
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

        # 1. Projections
        self.proj_dict = nn.ModuleDict()
        for nt in self.node_types:
            self.proj_dict[nt] = nn.Linear(node_in_dims[nt], hidden_dim)

        # 2. GAT Layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    hidden_dim, hidden_dim // heads, heads=heads, 
                    add_self_loops=False, edge_dim=4
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            bn_dict = nn.ModuleDict()
            for nt in self.node_types:
                bn_dict[nt] = BatchNorm(hidden_dim)
            self.batch_norms.append(bn_dict)

        # 3. Heads (핵심 수정 포인트)
        
        # LiDAR: [pred_disp_x, pred_disp_y, log_var_pos] -> 3 channels
        # (라이다는 위치를 맞춰야 하므로 변위량 예측 필요)
        self.head_lidar = nn.Linear(hidden_dim, 3) 
        
        # Radar: [log_var_vel] -> 1 channel
        # (레이다는 속도 예측 안 함. 오직 불확실성 시그마만 출력)
        self.head_radar = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        # 1. Build Graph & Return Edges (Loss에서 방향 벡터 계산용)
        edge_index_dict, edge_attr_dict = build_graph(
            data, self.radius_ll, self.radius_rr, self.radius_cross,
            self.temporal_radius_ll, self.temporal_radius_rr,
            self.max_Lnum, self.max_Rnum, data['lidar'].x.device
        )

        # 2. Process
        x_dict = {}
        for nt in self.node_types:
            x_dict[nt] = F.elu(self.proj_dict[nt](data[nt].x))

        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            for nt in self.node_types:
                x_dict[nt] = F.elu(x_dict[nt])
                x_dict[nt] = F.dropout(x_dict[nt], p=0.0, training=self.training)
                x_dict[nt] = self.batch_norms[i][nt](x_dict[nt])

        # 3. Output
        output_dict = {}
        
        # LiDAR Output: (Displacement, Sigma)
        if 'lidar' in x_dict:
            out_l = self.head_lidar(x_dict['lidar'])
            # [:, :2]는 변위량(dx, dy), [:, 2:]는 시그마
            output_dict['lidar'] = (out_l[:, :2], out_l[:, 2:3]) 

        # Radar Output: Only Sigma!
        for r in ['radar1', 'radar2']:
            if r in x_dict:
                out_r = self.head_radar(x_dict[r])
                # 튜플 아님. 그냥 텐서(Sigma)만 리턴.
                output_dict[r] = out_r 

        return output_dict, edge_index_dict