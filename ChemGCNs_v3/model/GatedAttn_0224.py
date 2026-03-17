import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

# class NodeApplyModule(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)

#     def forward(self, node):
#         h = self.linear(node.data['h'])

#         return {'h': h}

# class GCNLayer(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(GCNLayer, self).__init__()
#         self.msg = fn.copy_u('h', 'm')
#         self.apply_mod = NodeApplyModule(dim_in, dim_out)

#     def reduce(self, nodes):
#         mbox = nodes.mailbox['m']
#         accum = torch.mean(mbox, dim = 1)

#         return {'h': accum}     

#     def forward(self, g, feature):
#         g.ndata['h'] = feature
#         g.update_all(self.msg, self.reduce)
#         g.apply_nodes(func = self.apply_mod)

#         return g.ndata.pop('h')
    
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

class GatedAtten_d_g(nn.Module):
    def __init__(self, d_g, d_d):
        super(GatedAtten_d_g, self).__init__()
        self.d_g = d_g
        self.d_d = d_d

        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.randn(self.d_d, self.d_g))
    
    def forward(self, hg, desc):

        A = self.sigmoid(desc @ self.W)        # (B, d_2d) @ (d_2d, d_g)
        hg = A * hg                     # (B, d_g)

        return hg

class GatedAtten_g_d(nn.Module):
    def __init__(self, d_g, d_d):
        super(GatedAtten_g_d, self).__init__()
        self.d_g = d_g
        self.d_d = d_d

        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.randn(self.d_g, self.d_d))
    
    def forward(self, hg, desc):

        A = self.sigmoid(hg @ self.W)        # (B, d_g) @ (d_g, d_d)
        desc = A * desc                      # (B, d_d)

        return desc

# Gated Attention
class Net_2d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 64, rank: int = 8):
        super(Net_2d, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 128
        dim_out_fc1 = 128
        dim_out_fc2 = 32
        dim_out = 1
        drop_out = 0.1

        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, dim_graph_emb)

        # # desc -> graph
        # self.gated = GatedAtten_d_g(dim_graph_emb, dim_2d_desc)
        # self.fc1 = nn.Linear(dim_graph_emb, dim_out_fc1)

        # graph -> desc
        self.gated = GatedAtten_g_d(dim_graph_emb, dim_2d_desc)
        self.fc1 = nn.Linear(dim_2d_desc, dim_out_fc1)


        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # # desc -> graph
        # out = self.gated(hg, desc_2d)

        # graph -> desc
        out = self.gated(hg, desc_2d)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out

class Net_3d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 64, rank: int = 8):
        super(Net_3d, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 20
        dim_out_fc1 = 256
        dim_out_fc2 = 64
        dim_out = 1
        drop_out = 0.3

        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_3d_desc))
        self.fc1 = nn.Linear(dim_3d_desc, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        concat = torch.cat([desc_2d, desc_3d], dim=1)

        A = hg @ self.W
        hg = A * concat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out


class Net(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 64, rank: int = 8):
        super(Net, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 20
        dim_out_fc1 = 256
        dim_out_fc2 = 64
        dim_out = 1
        drop_out = 0.3

        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_2d_desc))

        self.fc1 = nn.Linear(dim_2d_desc, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * desc_2d

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out