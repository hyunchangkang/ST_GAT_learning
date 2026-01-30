import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, BatchNorm
from torch_geometric.data import HeteroData
from src.utils import build_graph

class ST_HGAT(nn.Module):
    def __init__(self, hidden_dim, num_layers, heads, metadata, node_in_dims, 
                 radius_ll, radius_rr, radius_cross, 
                 temporal_radius_ll, temporal_radius_rr,
                 max_num_neighbors_lidar, max_num_neighbors_radar, **kwargs):
        
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

        # Calculate head dimension and actual hidden dimension for stability
        self.head_dim = hidden_dim // heads
        self.actual_hidden = self.head_dim * heads 

        # 1. Projections
        self.proj_dict = nn.ModuleDict()
        for nt in self.node_types:
            self.proj_dict[nt] = nn.Linear(node_in_dims[nt], self.actual_hidden)

        # 2. GAT Layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    self.actual_hidden, self.head_dim, heads=heads, 
                    add_self_loops=False, edge_dim=4
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            bn_dict = nn.ModuleDict()
            for nt in self.node_types:
                bn_dict[nt] = BatchNorm(self.actual_hidden)
            self.batch_norms.append(bn_dict)

        # 3. Output Heads: LiDAR and Radar now only output log(sigma^2)
        self.head_lidar = nn.Linear(self.actual_hidden, 1) 
        self.head_radar = nn.Linear(self.actual_hidden, 1)

    def forward(self, data: HeteroData):
        # 1. Build Graph: Using dynamic parameters from config
        edge_index_dict, edge_attr_dict = build_graph(
            data, self.radius_ll, self.radius_rr, self.radius_cross,
            self.temporal_radius_ll, self.temporal_radius_rr,
            self.max_Lnum, self.max_Rnum, data['lidar'].x.device
        )

        x_dict = {nt: F.elu(self.proj_dict[nt](data[nt].x)) for nt in self.node_types}

        for i in range(self.num_layers):
            # Guard: HeteroConv might skip node types with no edges
            new_x_dict = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            
            for nt in self.node_types:
                # Update only if node type exists in conv output; else keep previous features
                if nt in new_x_dict:
                    x_dict[nt] = new_x_dict[nt]
                
                if x_dict[nt].size(0) > 0:
                    x_dict[nt] = F.elu(x_dict[nt])
                    x_dict[nt] = F.dropout(x_dict[nt], p=0.0, training=self.training)
                    x_dict[nt] = self.batch_norms[i][nt](x_dict[nt])

        # 3. Output
        output_dict = {}
        if 'lidar' in x_dict:
            output_dict['lidar'] = self.head_lidar(x_dict['lidar'])

        for r in ['radar1', 'radar2']:
            if r in x_dict:
                output_dict[r] = self.head_radar(x_dict[r]) 

        return output_dict, edge_index_dict