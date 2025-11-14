import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import torch.nn.functional as F
import math

import dgl
import dgl.function as fn


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# =======================
# =======================
# =======================

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

# class MLPDecoder(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super(MLPDecoder, self).__init__()
#         self.fc1 = nn.Linear(20, 10)
#         self.fc2 = nn.Linear(10, dim_out)

#     def forward(self, x):
#         out = F.relu(self.fc1(x))
#         out = F.relu(self.fc2(x))
        
#         return out


class MLPDecoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(dim_in, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
    
# class Net(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super(Net, self).__init__()

#         self.gc1 = GCNLayer(dim_in, 100)
#         self.gc2 = GCNLayer(100, 20)

#         self.mlp = MLPDecoder(dim_in, dim_out, dim_self_feat)

#     def forward(self, g, self_feat):
#         # graph convolutional networks
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h

#         # graph embedding
#         hg = dgl.mean_nodes(g, 'h')

#         # bilinear attention
#         """
#         구현필요
#         """
#         ban_layer = BANLayer(v_dim=2, q_dim=1, h_dim=20, h_out=1)
#         bcn = weight_norm(ban_layer, name = 'h_mat', dim = None)
#         z = self_feat

#         f, attn = bcn(hg, z)

#         # fully connected networks
#         out = self.mlp(f)

#         return out

class BilinearGate(nn.Module):
    def __init__(self, dim_g, dim_self_feat):
        """
        dim_g        : graph embedding 차원 (여기서는 20)
        dim_self_feat: descriptor 차원
        """
        super(BilinearGate, self).__init__()
        # z -> graph 차원으로 projection
        self.proj = nn.Linear(dim_self_feat, dim_g)

    def forward(self, hg, self_feat):
        """
        hg        : (B, dim_g)
        self_feat : (B, dim_self_feat)
        return:
            f     : (B, dim_g)  - fused representation
            gate  : (B, dim_g)  - attention/gating 값
        """
        # descriptor를 graph 차원으로 사상
        z_proj = self.proj(self_feat)          # (B, dim_g)

        # bilinear-like interaction (element-wise product)
        score = hg * z_proj                    # (B, dim_g)

        # sigmoid gate
        gate = torch.sigmoid(score)            # (B, dim_g)

        # fusion
        f = gate * hg + (1.0 - gate) * z_proj  # (B, dim_g)

        return f, gate


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        # bilinear gate: graph(20) + descriptor(dim_self_feat)
        self.bilinear_gate = BilinearGate(dim_g=20, dim_self_feat=dim_self_feat)

        # fused representation f: (B, 20)
        self.mlp = MLPDecoder(dim_in=20, dim_out=dim_out)

    def forward(self, g, self_feat):
        # 1) graph convolution
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        # 2) graph embedding
        hg = dgl.mean_nodes(g, 'h')    # (B, 20)

        # 3) bilinear attention-like fusion
        f, gate = self.bilinear_gate(hg, self_feat)  # f: (B, 20)

        # (원하면 여기서 gate를 저장해서 나중에 꺼내 쓸 수도 있음)
        self.last_gate = gate

        # 4) fully connected
        out = self.mlp(f)              # (B, dim_out)

        return out
