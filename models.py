"""
GNN model architectures for node classification.
Supports GCN, GAT, and GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            self.convs[0] = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.convs[0] = GATConv(input_dim, output_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return x


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        else:
            self.convs[0] = SAGEConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return x


def create_model(args, data):
    input_dim  = data.num_features
    output_dim = int(data.y.max().item()) + 1
    if args.gnn_model == 'GCN':
        return GCN(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    elif args.gnn_model == 'GAT':
        return GAT(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    elif args.gnn_model == 'GraphSAGE':
        return GraphSAGE(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.gnn_model}. Choose from GCN, GAT, GraphSAGE.")
