import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super(GCNConv, self).__init__()

        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps = 1e-12
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, feat: torch.Tensor):
        A_tilde = g.adj().to_dense()
        deg = A_tilde.sum(dim=1)
        D_tilde = torch.pow(deg.clamp_min(self.eps), -0.5)

        X1 = feat * D_tilde.unsqueeze(-1)
        X2 = A_tilde @ X1
        X3 = X2 * D_tilde.unsqueeze(-1)

        out = X3 @ self.weight

        return out        

# KROVEX
class Net(nn.Module):
    def __init__(self, dim_in, dim_feat_2d):
        super(Net, self).__init__()

        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, 20)

        self.fc1 = nn.Linear(20 * dim_feat_2d, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.3)

    def forward(self, g, feat_2d):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        print('feat_2d', feat_2d.shape)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        feat_2d = feat_2d.unsqueeze(1)
        hg = torch.bmm(hg, feat_2d)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out, None