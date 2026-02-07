import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

import math


from torch.nn.utils.weight_norm import weight_norm
from dgl.nn import GraphConv



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


# ==================================================================
# =========================================================
# (Fix #1) Descriptor Tokenizer 강화:
#   token = Linear(value) + Embedding(index) -> LayerNorm
# =========================================================
class Descriptor_Tokenizer(nn.Module):
    def __init__(self, d_desc: int, d_t: int, dropout: float = 0.0):
        super().__init__()
        self.d_desc = d_desc
        self.d_t = d_t

        self.val_proj = nn.Linear(1, d_t)              # value -> token
        self.idx_emb  = nn.Embedding(d_desc, d_t)      # index -> token
        self.ln = nn.LayerNorm(d_t)
        self.drop = nn.Dropout(dropout)

        # buffer for indices [0..D-1]
        self.register_buffer("idx", torch.arange(d_desc), persistent=False)

    def forward(self, desc: torch.Tensor):
        """
        desc: (B, D)
        return: (B, D, d_t)
        """
        B, D = desc.shape
        assert D == self.d_desc, f"Expected D={self.d_desc}, got {D}"

        # (B,D,1) -> (B,D,d_t)
        val_tok = self.val_proj(desc.unsqueeze(-1))

        # (D,) -> (1,D,d_t) -> (B,D,d_t)
        idx_tok = self.idx_emb(self.idx).unsqueeze(0).expand(B, -1, -1)

        tok = val_tok + idx_tok
        tok = self.ln(tok)
        tok = self.drop(tok)
        return tok


class FCNet(nn.Module):
    """
    dims = [in, h1, h2, ..., out]
    - hidden layers: Linear + act
    - final layer: Linear (+ last_act if True)
    """
    def __init__(self, dims, act='ReLU', dropout=0.0, last_act=False):
        super().__init__()
        assert len(dims) >= 2
        layers = []

        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i+1]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

            is_last = (i == len(dims) - 2)
            if act and (not is_last or last_act):
                layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# =========================================================
# BCNet: BAN bilinear logits (원 구조 유지, q에도 dropout 적용)
# =========================================================
class BCNet(nn.Module):
    """
    v: (B, N, v_dim)
    q: (B, D, q_dim)
    logits: (B, G, N, D)
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU',
                 dropout_v=0.2, dropout_q=0.2, dropout_att=0.0, k=3):
        super().__init__()
        self.c = 32
        self.k = k
        self.h_out = h_out
        self.h_dim = h_dim

        # (Fix #3) 마지막 act 제거: bilinear 입력 분포 왜곡 줄이기
        self.v_net = FCNet([v_dim, h_dim * k], act=act, dropout=dropout_v, last_act=False)
        self.q_net = FCNet([q_dim, h_dim * k], act=act, dropout=dropout_q, last_act=False)

        self.drop_att = nn.Dropout(dropout_att)

        assert h_out <= self.c, "Set glimpse(h_out) <= 32 for this minimal path"
        self.h_mat  = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * k))
        self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1))
        nn.init.xavier_normal_(self.h_mat)
        nn.init.zeros_(self.h_bias)

    def forward(self, v, q):
        v_ = self.drop_att(self.v_net(v))    # (B,N,h_dim*k)
        q_ = self.q_net(q)                   # (B,D,h_dim*k)
        logits = torch.einsum('xhyk,bnk,bdk->bhnd', self.h_mat, v_, q_) + self.h_bias
        return logits

# =========================================================
# (Fix #2) 2-stage attention:
#  - joint softmax(N*D) 대신
#  - node-score / desc-score 각각 softmax
#  - p ≈ A_node ⊗ A_desc  (rank-1 factorized joint)
# =========================================================
class BiAttention_2Stage(nn.Module):
    """
    Returns:
      p:      (B, G, N, D)  factorized joint attention
      logits: (B, G, N, D)  bilinear logits (for debug/regularization)
      A_node: (B, G, N)
      A_desc: (B, G, D)
    """
    def __init__(self, v_dim, q_dim, z_dim, glimpse=4,
                 dropout_v=0.2, dropout_q=0.2, dropout_att=0.0, k=3,
                 temp_node=1.0, temp_desc=1.0):
        super().__init__()
        self.glimpse = glimpse
        self.temp_node = temp_node
        self.temp_desc = temp_desc

        self.bc = BCNet(v_dim, q_dim, z_dim, glimpse,
                        dropout_v=dropout_v, dropout_q=dropout_q,
                        dropout_att=dropout_att, k=k)

    def forward(self, v, q, v_mask=None, mask_with=-1e9):
        """
        v: (B,N,v_dim)
        q: (B,D,q_dim)
        v_mask: (B,N) True valid, False padded
        """
        logits = self.bc(v, q)  # (B,G,N,D)

        if v_mask is not None:
            mask = (~v_mask).unsqueeze(1).unsqueeze(3).expand_as(logits)
            logits = logits.masked_fill(mask, mask_with)

        # node scores: sum over descriptors
        node_score = logits.sum(dim=3)  # (B,G,N)
        # desc scores: sum over nodes
        desc_score = logits.sum(dim=2)  # (B,G,D)

        A_node = F.softmax(node_score / self.temp_node, dim=-1)  # (B,G,N)
        A_desc = F.softmax(desc_score / self.temp_desc, dim=-1)  # (B,G,D)

        # factorized joint: outer product
        p = A_node.unsqueeze(3) * A_desc.unsqueeze(2)  # (B,G,N,D)

        return p, logits, A_node, A_desc


# =========================================================
# Net: 노드 토큰×디스크립터 토큰 BAN(2-stage) 적용
# =========================================================
def dgl_nodes_to_dense(h, batch_num_nodes):
    if not torch.is_tensor(batch_num_nodes):
        batch_num_nodes = torch.tensor(batch_num_nodes, device=h.device)

    B = batch_num_nodes.numel()
    d = h.size(-1)
    Nmax = int(batch_num_nodes.max().item())

    v = h.new_zeros((B, Nmax, d))
    mask = torch.zeros((B, Nmax), device=h.device, dtype=torch.bool)

    start = 0
    for b in range(B):
        n = int(batch_num_nodes[b].item())
        v[b, :n] = h[start:start+n]
        mask[b, :n] = True
        start += n
    return v, mask


class Net_BAN_Stable(nn.Module):
    def __init__(self, dim_in, dim_out, dim_2d_desc,
                 d_g=64, d_t=32, z_dim=64, glimpse=4):
        super().__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)
        self.d_g = d_g

        # (Fix #1) descriptor tokenizer 강화 (+ LN)
        self.desc_tok = Descriptor_Tokenizer(d_desc=dim_2d_desc, d_t=d_t, dropout=0.1)

        # (Fix #2) 2-stage BAN attention
        self.ban = BiAttention_2Stage(v_dim=d_g, q_dim=d_t, z_dim=z_dim, glimpse=glimpse,
                                      dropout_v=0.1, dropout_q=0.1, dropout_att=0.0, k=3,
                                      temp_node=1.0, temp_desc=1.0)

        # head
        self.fc1 = nn.Linear(glimpse * (d_g + d_t), 128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, desc_2d, return_attn=False):
        # GNN -> node tokens
        x = g.ndata['feat']
        h = F.relu(self.gc1(g, x))
        h = F.relu(self.gc2(g, h))  # (total_nodes, d_g)

        # dense node tokens + mask
        batch_num_nodes = g.batch_num_nodes()
        v_nodes, v_mask = dgl_nodes_to_dense(h, batch_num_nodes)  # (B,Nmax,d_g)

        # descriptor tokens
        q_desc = self.desc_tok(desc_2d)  # (B,D,d_t)

        # BAN attention (2-stage)
        p, logits, A_node, A_desc = self.ban(v_nodes, q_desc, v_mask=v_mask)  # p:(B,G,N,D)

        # contexts
        node_ctx = torch.einsum('bgn,bnd->bgd', A_node, v_nodes)  # (B,G,d_g)
        desc_ctx = torch.einsum('bgd,bdt->bgt', A_desc, q_desc)   # (B,G,d_t)

        fused = torch.cat([node_ctx, desc_ctx], dim=-1)           # (B,G,d_g+d_t)
        fused = fused.reshape(fused.size(0), -1)                  # (B, G*(d_g+d_t))

        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        if return_attn:
            return out, {
                "p": p, "logits": logits,
                "A_node": A_node, "A_desc": A_desc,
                "v_mask": v_mask
            }
        return out
    
class BiAttention_RankR(nn.Module):
    """
    Rank-R factorized joint attention
    p = sum_r A_node[r] ⊗ A_desc[r]

    Returns:
      p      : (B, G, N, D)
      A_node : (B, G, R, N)
      A_desc : (B, G, R, D)
    """
    def __init__(self, v_dim, q_dim, z_dim, glimpse=4, rank=4,
                 dropout_v=0.1, dropout_q=0.1, k=3,
                 temp_node=1.0, temp_desc=1.0):
        super().__init__()
        self.glimpse = glimpse
        self.rank = rank
        self.temp_node = temp_node
        self.temp_desc = temp_desc

        self.bc = BCNet(
            v_dim=v_dim,
            q_dim=q_dim,
            h_dim=z_dim,
            h_out=glimpse * rank,   # G * R
            dropout_v=dropout_v,
            dropout_q=dropout_q,
            k=k
        )

    def forward(self, v, q, v_mask=None, mask_with=-1e9):
        """
        v: (B, N, d_g)
        q: (B, D, d_t)
        """
        B, N, _ = v.shape
        D = q.size(1)

        logits = self.bc(v, q)  # (B, G*R, N, D)
        logits = logits.view(B, self.glimpse, self.rank, N, D)

        if v_mask is not None:
            mask = (~v_mask).unsqueeze(1).unsqueeze(2).unsqueeze(4)
            logits = logits.masked_fill(mask, mask_with)

        # node / desc scores
        node_score = logits.sum(dim=4)   # (B,G,R,N)
        desc_score = logits.sum(dim=3)   # (B,G,R,D)

        A_node = F.softmax(node_score / self.temp_node, dim=-1)
        A_desc = F.softmax(desc_score / self.temp_desc, dim=-1)

        # rank-R factorized joint
        p = torch.einsum('bgrn,bgrd->bgnd', A_node, A_desc)

        return p, A_node, A_desc
class Net_BAN_RankR(nn.Module):
    def __init__(self, dim_in, dim_out, dim_2d_desc,
                 d_g=64, d_t=32, z_dim=64,
                 glimpse=4, rank=4):
        super().__init__()

        # GNN
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # descriptor tokenizer
        self.desc_tok = Descriptor_Tokenizer(
            d_desc=dim_2d_desc,
            d_t=d_t,
            dropout=0.1
        )

        # Rank-R BAN
        self.ban = BiAttention_RankR(
            v_dim=d_g,
            q_dim=d_t,
            z_dim=z_dim,
            glimpse=glimpse,
            rank=rank,
            k=3
        )

        # head
        fused_dim = glimpse * (d_g + d_t) + d_g   # + hg
        self.fc1 = nn.Linear(fused_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, desc_2d, return_attn=False):
        # ----- GNN -----
        x = g.ndata['feat']
        h = F.relu(self.gc1(g, x))
        h = F.relu(self.gc2(g, h))  # (total_nodes, d_g)
        g.ndata['h'] = h

        # graph pooled token
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)

        # dense node tokens
        v_nodes, v_mask = dgl_nodes_to_dense(h, g.batch_num_nodes())

        # descriptor tokens
        q_desc = self.desc_tok(desc_2d)  # (B,D,d_t)

        # ----- Rank-R BAN -----
        p, A_node, A_desc = self.ban(v_nodes, q_desc, v_mask=v_mask)

        # contexts
        node_ctx = torch.einsum('bgn,bnd->bgd', A_node.mean(dim=2), v_nodes)
        desc_ctx = torch.einsum('bgd,bdt->bgt', A_desc.mean(dim=2), q_desc)

        fused = torch.cat([node_ctx, desc_ctx], dim=-1)
        fused = fused.reshape(fused.size(0), -1)

        # 🔥 hg concat (실전 핵심)
        fused = torch.cat([fused, hg], dim=-1)

        # ----- MLP -----
        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        if return_attn:
            return out, {
                "p": p,
                "A_node": A_node,
                "A_desc": A_desc,
                "hg": hg
            }
        return out
