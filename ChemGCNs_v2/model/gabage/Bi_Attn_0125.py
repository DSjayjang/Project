import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class NodeApplyModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, node):
        h = self.linear(node.data['h'])

        return {'h': h}

class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCNLayer, self).__init__()
        self.msg = fn.copy_u('h', 'm')
        self.apply_mod = NodeApplyModule(dim_in, dim_out)

    def reduce(self, nodes):
        mbox = nodes.mailbox['m']
        accum = torch.mean(mbox, dim = 1)

        return {'h': accum}     

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.msg, self.reduce)
        g.apply_nodes(func = self.apply_mod)

        return g.ndata.pop('h')


# good 성능 좋음
# class Net(nn.Module):
#     def __init__(self, dim_in, dim_2d_desc, dim_3d_desc):
#         super(Net, self).__init__()

#         self.dim_in = dim_in
#         self.dim_out = 1
#         self.dim_2d_desc = dim_2d_desc
#         self.dim_3d_desc = dim_3d_desc
#         d_g = 20
#         d_h = 64
#         drop = 0.3

#         # GCN layer
#         self.gc1 = GCNLayer(dim_in, 100)
#         self.gc2 = GCNLayer(100, d_g)

#         # Bilinear attention blocks
#         self.proj_g = nn.Linear(d_g, d_h, bias=True)
#         self.proj_d = nn.Linear(dim_2d_desc, d_h, bias=True)

#         # bilinear weight W: (d_h, d_h)
#         self.W = nn.Parameter(torch.empty(d_h, d_h))
#         nn.init.xavier_uniform_(self.W)

#         # MLP head
#         self.fc1 = nn.Linear(20 * dim_2d_desc, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 1)

#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(drop)

#     def forward(self, g, desc_2d, desc_3d):
#         """
#         g: DGLGraph (batched)
#         desc_2d: (B, dim_2d_desc)  descriptor vector
#         """

#         # ---- 1) GCN + readout ----
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')   # (B, 20)

#         # ---- 2) Bilinear attention between hg and desc_2d ----
#         # shared space
#         h_g = self.proj_g(hg)         # (B, d_h)
#         h_d = self.proj_d(desc_2d)  # (B, d_h)

#         # s = h_g^T W h_d  (per sample)
#         t = h_g @ self.W              # (B, d_h)
#         s = (t * h_d).sum(dim=-1)     # (B,)
#         a = torch.sigmoid(s).unsqueeze(-1)  # (B, 1)

#         # gate descriptors (sample-wise scalar gate)
#         desc_2d_gate = a * desc_2d      # (B, dim_2d_desc)

#         # ---- 3) Keep your original outer-product fusion ----
#         hg_ = hg.unsqueeze(2)               # (B, 20, 1)
#         sf_ = desc_2d.unsqueeze(1)   # (B, 1, dim_2d_desc)

#         fusion = torch.bmm(hg_, sf_)        # (B, 20, dim_2d_desc)
#         fusion = fusion.view(fusion.size(0), -1)  # (B, 20*dim_2d_desc)

#         # MLP
#         out = F.relu(self.bn1(self.fc1(fusion)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))
#         out = self.fc3(out)                # (B, dim_out)

#         return out


class Net(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc):
        super(Net, self).__init__()

        self.dim_in = dim_in
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        d_g = 20
        d_h = 64
        drop = 0.3

        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # GCN layer
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # Bilinear attention blocks
        self.proj_g = nn.Linear(d_g, d_h)
        self.proj_2d = nn.Linear(dim_2d_desc, d_h)
        self.proj_3d = nn.Linear(dim_3d_desc, d_h)

        # bilinear weight W: (d_h, d_h)
        self.W2 = nn.Parameter(torch.empty(d_h, d_h))
        self.W3 = nn.Parameter(torch.empty(d_h, d_h))
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)

        # MLP head
        # self.fc1 = nn.Linear(d_g + dim_2d_desc, mlp_hidden1)
        self.fc1 = nn.Linear(2 * d_h, mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)

        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d):
        """
        g:       (B, d_g)          graph embedding
        desc_2d: (B, dim_2d_desc)  3d descriptors
        desc_3d: (B, dim_3d_desc)  2d descriptors
        """
        # ---- 1) GCN ----
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')   # (B, 20)

        # ---- 2) Bilinear attention between hg and desc_2d ----
        # shared space
        h_g = self.proj_g(hg)         # (B, d_h)
        h_d = self.proj_2d(desc_2d)    # (B, d_h)

        # s = h_g^T W h_d  (per sample)
        t = h_g @ self.W2             # (B, d_h)
        s = (t * h_d).sum(dim=-1)     # (B,)
        a = torch.sigmoid(s).unsqueeze(-1)  # (B, 1)

        # # gate descriptors (sample-wise scalar gate)
        # desc_2d_gate = a * desc_2d      # (B, dim_2d_desc)
        # fusion = torch.cat([hg, desc_2d_gate], dim=-1)   
        # # ---- 3) Keep your original outer-product fusion ----
        # hg_ = hg.unsqueeze(2)               # (B, 20, 1)
        # sf_ = desc_2d.unsqueeze(1)   # (B, 1, dim_2d_desc)

        # fusion = torch.bmm(hg_, sf_)        # (B, 20, dim_2d_desc)
        # fusion = fusion.view(fusion.size(0), -1)  # (B, 20*dim_2d_desc)


        # test 중
        # ---- 3) bilinear attention (vector gate) ----
        # gate_vec = sigmoid( (h_g W) ⊙ h_d )  -> (B, d_h)
        gate_vec = torch.sigmoid((h_g @ self.W2) * h_d)   # (B, d_h)
        # bilinear-attended descriptor vector
        v2 = gate_vec * h_d                                # (B, d_h)
        # ---- 4) MLP regression ----
        fusion = torch.cat([h_g, v2], dim=-1)                   # (B, 2*d_h)



        # MLP
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)                # (B, dim_out)

        return out
