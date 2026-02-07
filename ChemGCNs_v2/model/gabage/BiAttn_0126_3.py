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


class Net_ContextualGate(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc=None, d_g=20, drop=0.1):
        super().__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_g = d_g

        # GCN
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # Contextual gates: g = sigmoid(W [hg; x])
        self.gate_2d = nn.Sequential(
            nn.Linear(d_g + dim_2d_desc, dim_2d_desc),
            nn.Sigmoid()
        )

        # if dim_3d_desc is not None:
        #     self.gate_3d = nn.Sequential(
        #         nn.Linear(d_g + dim_3d_desc, dim_3d_desc),
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.gate_3d = None
        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # Outer-product fusion (2D 기준)
        self.fc1 = nn.Linear((d_g + 1) * (dim_2d_desc + 1), mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)
        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d):
        # 1) Graph embedding
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)

        # 2) Contextual gating for 2D: g2 = sigmoid(W [hg; x2])
        gate_in_2d = torch.cat([hg, desc_2d], dim=1)   # (B, d_g + p2)
        g2 = self.gate_2d(gate_in_2d)                  # (B, p2)
        v2 = g2 * desc_2d                              # (B, p2)

        # # (옵션) 3D도 같은 방식으로 gate 생성 (예측엔 아직 미사용)
        # if (self.gate_3d is not None) and (desc_3d is not None):
        #     gate_in_3d = torch.cat([hg, desc_3d], dim=1)  # (B, d_g + p3)
        #     g3 = self.gate_3d(gate_in_3d)                 # (B, p3)
        #     v3 = g3 * desc_3d
        # else:
        #     g3, v3 = None, None

        # 3) Outer-product fusion (hg, v2)
        B = g.batch_size
        ones = torch.ones(B, 1, device=hg.device, dtype=hg.dtype)

        hg_aug = torch.cat([hg, ones], dim=1)  # (B, d_g+1)
        v2_aug = torch.cat([v2, ones], dim=1)  # (B, p2+1)

        fusion = torch.bmm(hg_aug.unsqueeze(2), v2_aug.unsqueeze(1)).view(B, -1)

        # 4) MLP head
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        # gate를 같이 반환하면 샘플별 중요도(XAI) 바로 뽑기 좋음
        return out

class Net_ContextualGate2(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc=None, d_g=20, drop=0.1):
        super().__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_g = d_g

        # GCN
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # Contextual gates: g = sigmoid(W [hg; x])
        self.gate_3d = nn.Sequential(
            nn.Linear(d_g + dim_3d_desc, dim_3d_desc),
            nn.Sigmoid()
        )

        # if dim_3d_desc is not None:
        #     self.gate_3d = nn.Sequential(
        #         nn.Linear(d_g + dim_3d_desc, dim_3d_desc),
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.gate_3d = None
        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # Outer-product fusion (2D 기준)
        self.fc1 = nn.Linear((d_g + 1) * (dim_3d_desc + 1), mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)
        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d):
        # 1) Graph embedding
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)

        # 2) Contextual gating for 2D: g2 = sigmoid(W [hg; x2])
        gate_in_3d = torch.cat([hg, desc_3d], dim=1)   # (B, d_g + p2)
        g3 = self.gate_3d(gate_in_3d)                  # (B, p2)
        v3 = g3 * desc_3d                              # (B, p2)

        # # (옵션) 3D도 같은 방식으로 gate 생성 (예측엔 아직 미사용)
        # if (self.gate_3d is not None) and (desc_3d is not None):
        #     gate_in_3d = torch.cat([hg, desc_3d], dim=1)  # (B, d_g + p3)
        #     g3 = self.gate_3d(gate_in_3d)                 # (B, p3)
        #     v3 = g3 * desc_3d
        # else:
        #     g3, v3 = None, None

        # 3) Outer-product fusion (hg, v2)
        B = g.batch_size
        ones = torch.ones(B, 1, device=hg.device, dtype=hg.dtype)

        hg_aug = torch.cat([hg, ones], dim=1)  # (B, d_g+1)
        v3_aug = torch.cat([v3, ones], dim=1)  # (B, p2+1)

        fusion = torch.bmm(hg_aug.unsqueeze(2), v3_aug.unsqueeze(1)).view(B, -1)

        # 4) MLP head
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        # gate를 같이 반환하면 샘플별 중요도(XAI) 바로 뽑기 좋음
        return out
