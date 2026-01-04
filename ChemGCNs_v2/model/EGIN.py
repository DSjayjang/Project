import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
import dgl

class MLP(nn.Module):
    """2-layer MLP for GINConv."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)
        self.bn1 = nn.BatchNorm1d(100)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)
        self.bn2 = nn.BatchNorm1d(20)

        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = self.bn2(h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out