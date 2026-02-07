import math
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

class BilinearAttentionFusion2D(nn.Module):
    """
    Graph embedding (hg) as query, 2D descriptors as tokens (scalar -> token).
    - Adds descriptor ID embedding so each descriptor has its own identity.
    - Bilinear scoring via low-rank factorization: e_i = <Uq(q), Vk(k_i)> / sqrt(d_att)
    - Attention type: "softmax" (competitive) or "sigmoid" (independent, optionally renormed)
    Returns:
      c     : (B, d_att)  attended context vector
      alpha : (B, P)      importance weights per descriptor
    """
    def __init__(
        self,
        d_g: int,
        dim_2d_desc: int,
        d_att: int = 64,
        attn_type: str = "sigmoid",     # "softmax" | "sigmoid"
        sigmoid_renorm: bool = True,
        use_layernorm: bool = True,
        drop: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        assert attn_type in ["softmax", "sigmoid"]
        self.p2 = dim_2d_desc
        self.d_att = d_att
        self.attn_type = attn_type
        self.sigmoid_renorm = sigmoid_renorm
        self.eps = eps

        # hg -> q
        self.proj_q = nn.Linear(d_g, d_att)

        # descriptor scalar -> k,v (shared)
        self.proj_k_val = nn.Linear(1, d_att, bias=True)
        self.proj_v_val = nn.Linear(1, d_att, bias=True)

        # descriptor ID embeddings (critical!)
        self.emb_k_id = nn.Embedding(self.p2, d_att)
        self.emb_v_id = nn.Embedding(self.p2, d_att)

        # low-rank bilinear (more stable than full W_bi)
        self.Uq = nn.Linear(d_att, d_att, bias=False)
        self.Vk = nn.Linear(d_att, d_att, bias=False)

        self.ln_q = nn.LayerNorm(d_att) if use_layernorm else nn.Identity()
        self.ln_k = nn.LayerNorm(d_att) if use_layernorm else nn.Identity()
        self.ln_v = nn.LayerNorm(d_att) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(drop)

        self.register_buffer("desc_ids", torch.arange(self.p2, dtype=torch.long), persistent=False)

    def forward(self, hg: torch.Tensor, desc_2d: torch.Tensor):
        """
        hg      : (B, d_g)
        desc_2d : (B, P)
        """
        B, P = desc_2d.shape
        if P != self.p2:
            raise ValueError(f"Expected dim_2d_desc={self.p2}, got {P}")

        # (1) query
        q = self.proj_q(hg)           # (B, d_att)
        q = self.ln_q(q)
        q = self.drop(q)

        # (2) tokens: value proj + ID embedding
        x = desc_2d.unsqueeze(-1)     # (B, P, 1)
        k = self.proj_k_val(x)        # (B, P, d_att)
        v = self.proj_v_val(x)        # (B, P, d_att)

        ids = self.desc_ids.unsqueeze(0).expand(B, P)  # (B, P)
        k = k + self.emb_k_id(ids)
        v = v + self.emb_v_id(ids)

        k = self.ln_k(k)
        v = self.ln_v(v)
        k = self.drop(k)
        v = self.drop(v)

        # (3) bilinear score (scaled)
        q2 = self.Uq(q)                      # (B, d_att)
        k2 = self.Vk(k)                      # (B, P, d_att)
        e = (k2 * q2.unsqueeze(1)).sum(-1) / math.sqrt(self.d_att)  # (B, P)

        # (4) attention weights
        if self.attn_type == "softmax":
            alpha = torch.softmax(e, dim=1)  # (B, P)
        else:
            alpha = torch.sigmoid(e)         # (B, P)
            if self.sigmoid_renorm:
                alpha = alpha / (alpha.sum(dim=1, keepdim=True) + self.eps)

        # (5) context vector
        c = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)  # (B, d_att)
        return c, alpha



class Net_BiAttn2D(nn.Module):
    """
    네가 준 Net_ContextualGate "형식"을 그대로 따르되,
    (2) 단계의 contextual gating 대신 bilinear attention fusion을 넣은 2D-only 모델.
    - GCNConv 사용
    - Outer-product fusion 유지
    - forward 시 desc_3d는 받지만 미사용(호환성)
    """
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc=None, d_g=20, drop=0.1, d_att=64,
                 attn_type="sigmoid", sigmoid_renorm=True):
        super().__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_g = d_g
        self.d_att = d_att

        # GCN
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, d_g)

        # 2D Bilinear Attention Fusion
        self.bifuse_2d = BilinearAttentionFusion2D(
            d_g=d_g,
            dim_2d_desc=dim_2d_desc,
            d_att=d_att,
            attn_type=attn_type,          # "softmax" or "sigmoid"
            sigmoid_renorm=sigmoid_renorm,
            use_layernorm=True,
            drop=drop,
        )

        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # Outer-product fusion (hg, c)
        self.fc1 = nn.Linear((d_g + 1) * (d_att + 1), mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)
        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d=None, return_aux=False):
        # 1) Graph embedding
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)

        # 2) Bilinear attention for 2D: c = sum_i alpha_i v_i
        c2, alpha_2d = self.bifuse_2d(hg, desc_2d)  # c2: (B, d_att), alpha_2d: (B, p2)

        # 3) Outer-product fusion (hg, c2)
        B = g.batch_size
        ones = torch.ones(B, 1, device=hg.device, dtype=hg.dtype)

        hg_aug = torch.cat([hg, ones], dim=1)   # (B, d_g+1)
        c2_aug = torch.cat([c2, ones], dim=1)   # (B, d_att+1)

        fusion = torch.bmm(hg_aug.unsqueeze(2), c2_aug.unsqueeze(1)).view(B, -1)

        # 4) MLP head
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        if return_aux:
            aux = {"alpha_2d": alpha_2d, "context_2d": c2}
            return out, aux
        return out