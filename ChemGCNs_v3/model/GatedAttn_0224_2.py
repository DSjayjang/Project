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
        self.softmax = nn.softmax()
        self.W = nn.Parameter(torch.randn(self.d_g, self.d_d))

    def forward(self, hg, desc):

        A = self.sigmoid(hg @ self.W)        # (B, d_g) @ (d_g, d_d)
        desc = A * desc                      # (B, d_d)
        score = self.softmax(desc)

        return score


# =========================
# Gated Attention: graph -> desc (descriptor-token attention + gate)
# =========================
class GatedAttention_DescTokens(nn.Module):
    def __init__(self, d_g, d_d, d_model=64, ctx_inject_scale=0.1, residual_gate=True):
        super().__init__()
        self.d_d = d_d
        self.d_model = d_model
        self.ctx_inject_scale = ctx_inject_scale
        self.residual_gate = residual_gate

        # descriptor scalar -> token embedding
        self.desc_tok = nn.Linear(1, d_model)

        # q from graph, k/v from descriptor tokens
        self.Wq = nn.Linear(d_g, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # gate from (hg, context): per-descriptor gate
        self.gate = nn.Sequential(
            nn.Linear(d_g + d_model, d_d),
            nn.Sigmoid()
        )

        # project context back to d_d
        self.ctx_to_desc = nn.Linear(d_model, d_d)

        # optional: stabilize scales (helps sigmoid saturation a bit)
        self.norm_hg = nn.LayerNorm(d_g)
        self.norm_ctx = nn.LayerNorm(d_model)

    def forward(self, hg, desc):   # hg: (B,d_g), desc: (B,d_d)
        # Ensure floating type (sometimes desc comes as double)
        if desc.dtype != hg.dtype:
            desc = desc.to(hg.dtype)

        # (B,d_d,1) -> (B,d_d,d_model)
        tok = self.desc_tok(desc.unsqueeze(-1))

        hg_n = self.norm_hg(hg)
        q = self.Wq(hg_n).unsqueeze(1)          # (B,1,d_model)
        k = self.Wk(tok)                       # (B,d_d,d_model)
        v = self.Wv(tok)                       # (B,d_d,d_model)

        # attention over descriptor dims
        attn_logits = (q * k).sum(-1) / (self.d_model ** 0.5)  # (B,d_d)
        alpha = F.softmax(attn_logits, dim=-1)                 # (B,d_d)

        # context summary
        ctx = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)      # (B,d_model)
        ctx = self.norm_ctx(ctx)

        # gate per descriptor dim
        g = self.gate(torch.cat([hg_n, ctx], dim=-1))          # (B,d_d)

        # gated output: (option) residual gate to preserve base info
        if self.residual_gate:
            out_desc = desc * (1.0 + g)
        else:
            out_desc = desc * g

        # optional context injection
        out_desc = out_desc + (self.ctx_inject_scale * self.ctx_to_desc(ctx))  # (B,d_d)

        return out_desc, alpha, g


# =========================
# Gated Attention Network (2D desc)
# =========================
class Net_2d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc,
                 d_h: int = 64, rank: int = 8,
                 d_model: int = 64):
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

        # graph -> desc (gated attention)
        self.gated = GatedAttention_DescTokens(
            d_g=dim_graph_emb,
            d_d=dim_2d_desc,
            d_model=d_model,
            ctx_inject_scale=0.1,
            residual_gate=True
        )

        self.fc1 = nn.Linear(dim_2d_desc, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

        # 저장용(원하면 외부에서 확인)
        self.last_alpha = None
        self.last_gate = None

    def forward(self, g, desc_2d, desc_3d):
        # Graph encoder
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')  # (B, dim_graph_emb)

        # Gated attention: graph -> desc
        desc_2d_gated, alpha, gate = self.gated(hg, desc_2d)

        # (optional) keep for analysis/visualization
        self.last_alpha = alpha
        self.last_gate = gate

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(desc_2d_gated)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out

class Net_3d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc,
                 d_h: int = 64, rank: int = 8,
                 d_model: int = 64):
        super(Net_3d, self).__init__()
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

        # graph -> desc (gated attention)
        self.gated = GatedAttention_DescTokens(
            d_g=dim_graph_emb,
            d_d=dim_3d_desc,
            d_model=d_model,
            ctx_inject_scale=0.1,
            residual_gate=True
        )

        self.fc1 = nn.Linear(dim_3d_desc, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

        # 저장용(원하면 외부에서 확인)
        self.last_alpha = None
        self.last_gate = None

    def forward(self, g, desc_2d, desc_3d):
        # Graph encoder
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')  # (B, dim_graph_emb)

        # Gated attention: graph -> desc
        desc_3d_gated, alpha, gate = self.gated(hg, desc_3d)

        # (optional) keep for analysis/visualization
        self.last_alpha = alpha
        self.last_gate = gate

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(desc_3d_gated)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out
    
class Net_total(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc,
                 d_h: int = 64, rank: int = 8,
                 d_model: int = 64):
        super(Net_total, self).__init__()
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

        # graph -> desc (gated attention)
        self.gated = GatedAttention_DescTokens(
            d_g=dim_graph_emb,
            d_d=dim_3d_desc,
            d_model=d_model,
            ctx_inject_scale=0.1,
            residual_gate=True
        )

        self.fc1 = nn.Linear(dim_3d_desc, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

        # 저장용(원하면 외부에서 확인)
        self.last_alpha = None
        self.last_gate = None

    def forward(self, g, desc_2d, desc_3d):
        # Graph encoder
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')  # (B, dim_graph_emb)

        # Gated attention: graph -> desc
        desc_3d_gated, alpha, gate = self.gated(hg, desc_3d)

        # (optional) keep for analysis/visualization
        self.last_alpha = alpha
        self.last_gate = gate

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(desc_3d_gated)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out
  