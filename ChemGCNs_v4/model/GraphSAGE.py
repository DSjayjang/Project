import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
import dgl

class Net(nn.Module):
    def __init__(self, dim_in, aggr="mean"):
        """
        aggr: "mean" (default), "pool", "lstm", "gcn"
        """
        super(Net, self).__init__()
        dim_out = 1

        self.gc1 = SAGEConv(dim_in, 100, aggr)
        self.gc2 = SAGEConv(100, 20, aggr)

        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g, g.ndata["feat"]))
        h = F.relu(self.gc2(g, h))
        g.ndata["h"] = h

        hg = dgl.mean_nodes(g, "h")

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out, None