# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_softmax



class EdgeAwareGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int, dropout: float, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin_node = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim * heads, bias=False)

        self.att = nn.Parameter(torch.empty((heads, 3 * out_dim)))
        nn.init.xavier_uniform_(self.att)

        self.leaky = nn.LeakyReLU(0.2)
        self.bn = nn.LayerNorm(out_dim * heads if concat else out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        if edge_index.numel() == 0:
            h = self.lin_node(x).view(N, self.heads, self.out_dim)
            out = h.reshape(N, self.heads * self.out_dim) if self.concat else h.mean(dim=1)
            return self.bn(out)

        src, dst = edge_index[0], edge_index[1]
        h = self.lin_node(x).view(N, self.heads, self.out_dim)
        e = self.lin_edge(edge_attr).view(-1, self.heads, self.out_dim)

        hs = h[src]
        hd = h[dst]
        cat = torch.cat([hs, hd, e], dim=-1)
        logits = (cat * self.att.unsqueeze(0)).sum(dim=-1)
        logits = self.leaky(logits)

        alpha = safe_softmax(logits, dst, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = (hs + e) * alpha.unsqueeze(-1)

        out = torch.zeros((N, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg.to(x.dtype))

        if self.concat:
            out = out.reshape(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return self.bn(out)


class RelationalGATBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        self.gat1 = EdgeAwareGAT(dim, dim, edge_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = EdgeAwareGAT(dim * heads, dim, edge_dim, heads=heads, dropout=dropout, concat=True)
        self.proj = nn.Linear(dim, dim * heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x0 = self.proj(x)
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = self.drop(h)
        h2 = self.gat2(h, edge_index, edge_attr)
        h2 = self.drop(h2)
        return x0 + h2


