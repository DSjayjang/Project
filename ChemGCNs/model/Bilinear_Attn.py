import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
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
            fusion_logits = fusion_logits.unsqueeze(1)           # (B, 1, h_dim*k)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) # (B, h_dim)
            fusion_logits = fusion_logits * self.k               # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        """
        v: (B, v_num, v_dim)
        q: (B, q_num, q_dim)
        return:
            logits:   (B, h_dim)
            att_maps: (B, h_out, v_num, q_num)
        """
        v_num = v.size(1)
        q_num = q.size(1)

        if self.h_out <= self.c:
            v_ = self.v_net(v)  # (B, v_num, h_dim*k)
            q_ = self.q_net(q)  # (B, q_num, h_dim*k)

            # h_mat: (1, h_out, 1, h_dim*k)
            # einsum -> (B, h_out, v_num, q_num)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)        # b x h_out x v x q

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        # 첫 번째 head에 대해 pooling
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        # logits: (B, h_dim)
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
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

class MLPDecoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(dim_in, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)

        return out

# Bilinear Attn
class Net_New(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat, hidden_in = 256, hidden_out = 512):
        super(Net_New, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(hidden_in, hidden_out))

        self.W_q = nn.Parameter(torch.randn(hidden_in, dim_self_feat)) # descriptor
        self.W_k = nn.Parameter(torch.randn(hidden_out, 20)) # graph emb
        self.W_v = nn.Parameter(torch.randn(hidden_out, 20)) # graph emb

        self.fc1 = nn.Linear(hidden_in + hidden_out, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # query, key, value

        # descriptor
        q = self_feat @ self.W_q.T # (B, dim_self_feat) x (dim_self_feat, hidden_in) = (B, hidden_in)

        # graph embedding
        k = hg @ self.W_k.T # (B, 20) x (20, hidden_out) = (B, hidden_out)

        # graph embedding
        v = hg @ self.W_v.T # (B, 20) x (20, hidden_out) = (B, hidden_out)

        # attention score
        Wk = self.W @ k.T # (hidden_in, hidden_out) x (hidden_out, B)
        bilinear = torch.sum(q * Wk.T, dim = 1, keepdim = True) # (B, hidden_in) * (B, hidden_out) = (B, hidden_in or out)
        alpha = torch.sigmoid(bilinear)
        # print(alpha)
        z = alpha * v
        z = torch.cat([z, q], dim=1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(z)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out