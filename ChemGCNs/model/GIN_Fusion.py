import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from dgl.nn.pytorch import GINConv
import dgl
print('okkk?', GINConv)

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


class kronecker_Net_3(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)
        
        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, dim_out)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)
        
        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_Net_5(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, dim_out)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_Net_7(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, dim_out)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_Net_10(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, dim_out)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_Net_20(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super().__init__()

        mlp1 = MLP(dim_in, 100, 100)
        self.gin1 = GINConv(mlp1)

        mlp2 = MLP(100, 20, 20)
        self.gin2 = GINConv(mlp2)

        self.fc1 = nn.Linear(20 * dim_self_feat, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, g, self_feat):
        h = g.ndata['feat']

        h = self.gin1(g, h)
        h = F.relu(h)

        h = self.gin2(g, h)
        h = F.relu(h)

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    


# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

# class kronecker_Net_3(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)
        
#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 32)
#         self.fc2 = nn.Linear(32, 8)
#         self.fc3 = nn.Linear(8, dim_out)

#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(8)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)
    
#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out

# class kronecker_Net_5(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)

#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 32)
#         self.fc2 = nn.Linear(32, 8)
#         self.fc3 = nn.Linear(8, dim_out)

#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(8)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)

#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out
    

# class kronecker_Net_7(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)

#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 64)
#         self.fc2 = nn.Linear(64, 16)
#         self.fc3 = nn.Linear(16, dim_out)

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(16)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)

#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out
    

# class kronecker_Net_10(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)

#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 64)
#         self.fc2 = nn.Linear(64, 16)
#         self.fc3 = nn.Linear(16, dim_out)

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(16)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)

#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out
    

# class kronecker_Net_20(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)

#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 256)
#         self.fc2 = nn.Linear(256, 32)
#         self.fc3 = nn.Linear(32, dim_out)

#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)

#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out
    

# class Net(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super().__init__()

#         mlp1 = MLP(dim_in, 100, 100)
#         self.gin1 = GINConv(mlp1)

#         mlp2 = MLP(100, 20, 20)
#         self.gin2 = GINConv(mlp2)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, dim_out)

#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(0.3)
    
#         self.gin_bn1 = nn.BatchNorm1d(100)
#         self.gin_bn2 = nn.BatchNorm1d(20)

#     def forward(self, g, self_feat):
#         h = g.ndata['feat']

#         h = self.gin1(g, h)
#         h = self.gin_bn1(h)
#         h = F.relu(h)
        
#         h = self.gin_bn1(h)
#         h = self.gin2(g, h)
#         h = F.relu(h)

#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # fully connected networks
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out
    