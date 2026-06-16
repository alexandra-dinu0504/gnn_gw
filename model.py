import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class GWEmbedder(nn.Module):
    """GCN that produces graph embedding."""

    def __init__(self, in_channels=1, hidden_channels=64, embed_dim=32, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))

        # Middle layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))

        # Last conv layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))

        # MLP for final embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, embed_dim)
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Mean and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)

        return self.mlp(x)