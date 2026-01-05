import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

import math
from typing import Optional, Tuple

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

# TFN
class LowRankTriLinearFusionAttn(nn.Module):
    """
    Low-rank tri-linear (CP) fusion with rank-wise attention.

    Given three modality vectors:
      h_g   : (B, d_g)
      h_2d  : (B, d_2d)
      h_3d  : (B, d_3d)

    1) Project each to a common fusion dim d_f
    2) For r in [1..R], compute elementwise interaction:
         z_r = (U_r h_g) ⊙ (V_r h_2d) ⊙ (S_r h_3d)   where each term is (B, d_f)
    3) Compute attention over ranks:
         beta = softmax(MLP([h_g; h_2d; h_3d]))  -> (B, R)
    4) Fuse:
         z = sum_r beta_r * z_r   -> (B, d_f)
    5) Optionally output to d_out with an MLP head (handled outside or inside).

    This avoids the full outer-product tensor of size (d_g*d_2d*d_3d).
    """

    def __init__(
        self,
        d_g: int,
        d_2d: int,
        d_3d: int,
        d_f: int = 64,
        rank: int = 8,
        attn_hidden: int = 128,
        dropout: float = 0.15,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.d_g, self.d_2d, self.d_3d = d_g, d_2d, d_3d
        self.d_f = d_f
        self.rank = rank

        # Modality projections (shared fusion dimension)
        self.proj_g = nn.Linear(d_g, d_f, bias=True)
        self.proj_2d = nn.Linear(d_2d, d_f, bias=True)
        self.proj_3d = nn.Linear(d_3d, d_f, bias=True)

        self.norm_g = nn.LayerNorm(d_f) if use_layernorm else nn.Identity()
        self.norm_2d = nn.LayerNorm(d_f) if use_layernorm else nn.Identity()
        self.norm_3d = nn.LayerNorm(d_f) if use_layernorm else nn.Identity()

        # Rank-specific linear maps:
        # We implement CP factors as one big linear per modality producing (B, R*d_f),
        # then reshape to (B, R, d_f). This is fast and simple.
        self.U = nn.Linear(d_f, rank * d_f, bias=False)
        self.V = nn.Linear(d_f, rank * d_f, bias=False)
        self.S = nn.Linear(d_f, rank * d_f, bias=False)

        # Rank-wise attention: beta in R^R per sample
        self.attn = nn.Sequential(
            nn.Linear(d_g + d_2d + d_3d, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, rank),
        )

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Reasonable init for stability
        for m in [self.proj_g, self.proj_2d, self.proj_3d]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        for m in [self.U, self.V, self.S]:
            nn.init.xavier_uniform_(m.weight)

        for m in self.attn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h_g: torch.Tensor,
        h_2d: torch.Tensor,
        h_3d: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          z: (B, d_f)
          beta (optional): (B, R) rank attention weights
        """
        # Project + normalize
        g = self.norm_g(self.proj_g(h_g))
        d2 = self.norm_2d(self.proj_2d(h_2d))
        d3 = self.norm_3d(self.proj_3d(h_3d))

        g = self.dropout(F.relu(g))
        d2 = self.dropout(F.relu(d2))
        d3 = self.dropout(F.relu(d3))

        B = g.size(0)

        # Produce rank-specific embeddings: (B, R, d_f)
        g_r = self.U(g).view(B, self.rank, self.d_f)
        d2_r = self.V(d2).view(B, self.rank, self.d_f)
        d3_r = self.S(d3).view(B, self.rank, self.d_f)

        # Elementwise tri-linear interactions per rank
        z_r = g_r * d2_r * d3_r  # (B, R, d_f)

        # Rank-wise attention from raw (unprojected) inputs to keep semantics
        beta_logits = self.attn(torch.cat([h_g, h_2d, h_3d], dim=-1))  # (B, R)
        beta = F.softmax(beta_logits, dim=-1)  # (B, R)

        # Weighted sum across ranks
        z = torch.sum(z_r * beta.unsqueeze(-1), dim=1)  # (B, d_f)
        z = self.dropout(z)

        if return_attn:
            return z, beta
        return z, None


class KROVEX_LowRankTFN(nn.Module):
    """
    - GCN -> graph embedding hg
    - Low-rank tri-linear fusion with rank-attention over (hg, self_feat, x3d)
    - MLP head for prediction
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_self_feat: int,
        dim_3d_feat: int,
        dim_graph_emb: int = 20,
        fusion_dim: int = 64,
        fusion_rank: int = 8,
        attn_hidden: int = 128,
        mlp_hidden1: int = 128,
        mlp_hidden2: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dim_graph_emb = dim_graph_emb
        self.dim_self_feat = dim_self_feat
        self.dim_3d_feat = dim_3d_feat

        # --- GCN backbone ---
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, dim_graph_emb)

        # --- Low-rank tri-linear fusion + attention ---
        self.fusion = LowRankTriLinearFusionAttn(
            d_g = dim_graph_emb,
            d_2d = dim_self_feat,
            d_3d = dim_3d_feat,
            d_f = fusion_dim,
            rank = fusion_rank,
            attn_hidden = attn_hidden,
            dropout = dropout,
            use_layernorm = True,
        )

        # --- Prediction head ---
        self.fc1 = nn.Linear(fusion_dim, mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, self_feat: torch.Tensor, x3d: torch.Tensor, return_attn: bool = False):
        """
        Args:
          g: DGLGraph (batched)
          self_feat: (B, dim_self_feat)  -- 2D descriptors
          x3d:      (B, dim_3d_feat)     -- 3D descriptors
          return_attn: if True, also returns rank-attention weights beta (B, R)

        Returns:
          out: (B, dim_out)
          beta (optional): (B, R)
        """
        # --- Graph embedding ---
        h = F.relu(self.gc1(g, g.ndata["feat"]))
        h = F.relu(self.gc2(g, h))
        g.ndata["h"] = h
        
        hg = dgl.mean_nodes(g, "h")  # (B, dim_graph_emb)

        # --- Low-rank TFN fusion ---
        fused, beta = self.fusion(hg, self_feat, x3d, return_attn=return_attn)  # (B, fusion_dim)

        # --- MLP head ---
        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        if return_attn:
            return out, beta
        return out


# -----------------------------
# Optional: if you also want modality-gating before TFN
# -----------------------------
class ModalityGate(nn.Module):
    """
    Computes modality weights alpha over {g, 2d, 3d} and scales each modality.
    Useful if you want: modality importance + rank importance (two-level interpretability).
    """

    def __init__(self, d_g: int, d_2d: int, d_3d: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_g + d_2d + d_3d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, h_g: torch.Tensor, h_2d: torch.Tensor, h_3d: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(torch.cat([h_g, h_2d, h_3d], dim=-1))  # (B,3)
        return F.softmax(logits, dim=-1)  # (B,3)


class KROVEX_LowRankTFN_TwoLevelAttn(nn.Module):
    """
    Two-level attention:
      1) modality gate alpha_g, alpha_2d, alpha_3d
      2) rank-wise attention beta_1..beta_R inside low-rank TFN
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_self_feat: int,
        dim_3d_feat: int,
        dim_graph_emb: int = 20,
        fusion_dim: int = 64,
        fusion_rank: int = 8,
        attn_hidden: int = 128,
        gate_hidden: int = 128,
        mlp_hidden1: int = 128,
        mlp_hidden2: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.dim_graph_emb = dim_graph_emb

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, dim_graph_emb)

        self.gate = ModalityGate(dim_graph_emb, dim_self_feat, dim_3d_feat, hidden=gate_hidden, dropout=dropout)

        self.fusion = LowRankTriLinearFusionAttn(
            d_g=dim_graph_emb,
            d_2d=dim_self_feat,
            d_3d=dim_3d_feat,
            d_f=fusion_dim,
            rank=fusion_rank,
            attn_hidden=attn_hidden,
            dropout=dropout,
            use_layernorm=True,
        )

        self.fc1 = nn.Linear(fusion_dim, mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, self_feat: torch.Tensor, x3d: torch.Tensor, return_attn: bool = False):
        h = F.relu(self.gc1(g, g.ndata["feat"]))
        h = F.relu(self.gc2(g, h))
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")

        alpha = self.gate(hg, self_feat, x3d)  # (B,3)
        hg2 = hg * alpha[:, 0:1]
        sf2 = self_feat * alpha[:, 1:2]
        x3d2 = x3d * alpha[:, 2:3]

        fused, beta = self.fusion(hg2, sf2, x3d2, return_attn=return_attn)

        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        if return_attn:
            # return both modality and rank attention for interpretability
            return out, {"alpha": alpha, "beta": beta}
        return out