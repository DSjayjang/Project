"""
Improved Net_PPF (Full Code)
- Normalized Residual GCN (self-loop + D^{-1/2} A D^{-1/2})
- Posterior Predictive Fusion (PPF): mu/logvar + calibrated residual z
- Reliability gate: precision (1/var) * sigmoid(a) (해석/성능 둘 다)
- Low-Rank Bilinear Fusion (LRBF): (U hg) ⊙ (V v2_red)  (outer-product보다 과적합 덜함)
- Training: task loss + lambda_desc * NLL(desc | hg)
"""

import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
import dgl.function as fn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold


# -----------------------------
# 1) Normalized Residual GCN Layer
#    h = D^{-1/2} A_hat D^{-1/2} x W  (+ residual) -> LN -> ReLU
# -----------------------------
class GCNLayerNorm(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out, bias=False)
        self.res = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim_out) if use_ln else nn.Identity()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        g: batched DGLGraph
        x: (num_nodes_in_batch, dim_in)
        """
        with g.local_scope():
            # self-loop to include self-message (important for molecular graphs)
            g2 = dgl.add_self_loop(g)

            deg = g2.in_degrees().float().clamp(min=1).to(x.device)  # (N,)
            norm = torch.pow(deg, -0.5).unsqueeze(1)                 # (N,1)

            h = self.lin(x)          # (N, dim_out)
            h = h * norm             # left normalization

            g2.ndata["h"] = h
            g2.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            h = g2.ndata["h"]        # (N, dim_out)

            h = h * norm             # right normalization

            h = h + self.res(x)      # residual
            h = self.ln(h)
            h = F.relu(self.dropout(h))
            return h


# -----------------------------
# 2) Net: PPF + LRBF
# -----------------------------
class Net_PPF_LRBF(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        d_g: int = 64,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,          # descriptor representation reduction
        rank: int = 64,           # low-rank bilinear fusion dimension
        mlp_hidden: int = 128,
        dim_out: int = 1,
        logv_clip: float = 8.0,
        use_layernorm: bool = True,
        gate_max_precision: float = 50.0,
    ):
        super().__init__()
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = gate_max_precision

        # ---- GNN backbone ----
        self.gc1 = GCNLayerNorm(dim_in, 128, dropout=drop_gnn, use_ln=use_layernorm)
        self.gc2 = GCNLayerNorm(128, d_g,  dropout=drop_gnn, use_ln=use_layernorm)

        # ---- PPF heads (2D) ----
        self.mu_2d   = nn.Linear(d_g, dim_2d_desc)
        self.logv_2d = nn.Linear(d_g, dim_2d_desc)
        self.a_2d    = nn.Linear(d_g, dim_2d_desc)   # gate logits

        # ---- reduce descriptor channel before fusion ----
        self.v_reduction = nn.Sequential(
            nn.Linear(dim_2d_desc, v_red),
            nn.LayerNorm(v_red),
            nn.ReLU(),
            nn.Dropout(drop_head),
        )

        # ---- Low-Rank Bilinear Fusion: (U hg) ⊙ (V v2_red) ----
        self.U = nn.Linear(d_g, rank, bias=False)
        self.V = nn.Linear(v_red, rank, bias=False)
        self.fuse_ln = nn.LayerNorm(rank)

        # ---- Prediction head ----
        self.head = nn.Sequential(
            nn.Linear(rank, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(drop_head),
            nn.Linear(mlp_hidden, dim_out),
        )

    def _ppf_2d(self, hg: torch.Tensor, x2d: torch.Tensor):
        """
        hg:  (B, d_g)
        x2d: (B, D2)
        returns:
          v2:   (B, D2) fused descriptor representation (w * z)
          nll:  (B,)    Gaussian NLL per sample (sum over D2)
          aux:  dict    (mu, logv, sigma, w, z)
        """
        mu = self.mu_2d(hg)  # (B, D2)
        logv = self.logv_2d(hg).clamp(-self.logv_clip, self.logv_clip)
        var = torch.exp(logv) + self.eps
        sigma = torch.sqrt(var)

        # calibrated residual
        z = (x2d - mu) / (sigma + self.eps)

        # reliability: precision * sigmoid(gate)
        precision = 1.0 / var
        precision = precision.clamp(max=self.gate_max_precision)
        gate = torch.sigmoid(self.a_2d(hg))
        w = gate * precision

        v2 = w * z

        # Gaussian NLL (omit constant)
        nll_elem = 0.5 * ((x2d - mu) ** 2 / var + torch.log(var))
        nll = nll_elem.sum(dim=1)  # (B,)

        aux = {"mu": mu, "logv": logv, "sigma": sigma, "w": w, "z": z, "gate": gate, "precision": precision}
        return v2, nll, aux

    def forward(self, g: dgl.DGLGraph, desc_2d: torch.Tensor, desc_3d: torch.Tensor, return_aux: bool = False):
        # ---- GNN ----
        x = g.ndata["feat"]
        h = self.gc1(g, x)
        h = self.gc2(g, h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")  # (B, d_g)

        # ---- PPF (2D) ----
        v2, nll2, aux2 = self._ppf_2d(hg, desc_2d)

        # ---- reduce + LRBF ----
        v2_red = self.v_reduction(v2)               # (B, v_red)
        fuse = self.U(hg) * self.V(v2_red)          # (B, rank)
        fuse = self.fuse_ln(fuse)

        out = self.head(fuse)                       # (B, 1)

        if not return_aux:
            return out
        return out, {"hg": hg, "nll_2d": nll2, "aux_2d": aux2, "v2_red": v2_red, "fuse": fuse}


class Net_PPF_LRBF2(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        dim_3d_desc: int,
        d_g: int = 64,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,          # descriptor representation reduction
        rank: int = 64,           # low-rank bilinear fusion dimension
        mlp_hidden: int = 128,
        dim_out: int = 1,
        logv_clip: float = 8.0,
        use_layernorm: bool = True,
        gate_max_precision: float = 50.0,
    ):
        super().__init__()
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = gate_max_precision

        # ---- GNN backbone ----
        self.gc1 = GCNLayerNorm(dim_in, 128, dropout=drop_gnn, use_ln=use_layernorm)
        self.gc2 = GCNLayerNorm(128, d_g,  dropout=drop_gnn, use_ln=use_layernorm)

        # ---- PPF heads (2D) ----
        self.mu_3d   = nn.Linear(d_g, dim_3d_desc)
        self.logv_3d = nn.Linear(d_g, dim_3d_desc)
        self.a_3d    = nn.Linear(d_g, dim_3d_desc)   # gate logits

        # ---- reduce descriptor channel before fusion ----
        self.v_reduction = nn.Sequential(
            nn.Linear(dim_3d_desc, v_red),
            nn.LayerNorm(v_red),
            nn.ReLU(),
            nn.Dropout(drop_head),
        )

        # ---- Low-Rank Bilinear Fusion: (U hg) ⊙ (V v2_red) ----
        self.U = nn.Linear(d_g, rank, bias=False)
        self.V = nn.Linear(v_red, rank, bias=False)
        self.fuse_ln = nn.LayerNorm(rank)

        # ---- Prediction head ----
        self.head = nn.Sequential(
            nn.Linear(rank, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(drop_head),
            nn.Linear(mlp_hidden, dim_out),
        )

    def _ppf_3d(self, hg: torch.Tensor, x3d: torch.Tensor):
        """
        hg:  (B, d_g)
        x2d: (B, D2)
        returns:
          v2:   (B, D2) fused descriptor representation (w * z)
          nll:  (B,)    Gaussian NLL per sample (sum over D2)
          aux:  dict    (mu, logv, sigma, w, z)
        """
        mu = self.mu_3d(hg)  # (B, D2)
        logv = self.logv_3d(hg).clamp(-self.logv_clip, self.logv_clip)
        var = torch.exp(logv) + self.eps
        sigma = torch.sqrt(var)

        # calibrated residual
        z = (x3d - mu) / (sigma + self.eps)

        # reliability: precision * sigmoid(gate)
        precision = 1.0 / var
        precision = precision.clamp(max=self.gate_max_precision)
        gate = torch.sigmoid(self.a_3d(hg))
        w = gate * precision

        v3 = w * z

        # Gaussian NLL (omit constant)
        nll_elem = 0.5 * ((x3d - mu) ** 2 / var + torch.log(var))
        nll = nll_elem.sum(dim=1)  # (B,)

        aux = {"mu": mu, "logv": logv, "sigma": sigma, "w": w, "z": z, "gate": gate, "precision": precision}
        return v3, nll, aux

    def forward(self, g: dgl.DGLGraph, desc_2d: torch.Tensor, desc_3d: torch.Tensor, return_aux: bool = False):
        # ---- GNN ----
        x = g.ndata["feat"]
        h = self.gc1(g, x)
        h = self.gc2(g, h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")  # (B, d_g)

        # ---- PPF (2D) ----
        v3, nll3, aux3 = self._ppf_3d(hg, desc_3d)

        # ---- reduce + LRBF ----
        v3_red = self.v_reduction(v3)               # (B, v_red)
        fuse = self.U(hg) * self.V(v3_red)          # (B, rank)
        fuse = self.fuse_ln(fuse)

        out = self.head(fuse)                       # (B, 1)

        if not return_aux:
            return out
        return out, {"hg": hg, "nll_2d": nll3, "aux_2d": aux3, "v2_red": v3_red, "fuse": fuse}

