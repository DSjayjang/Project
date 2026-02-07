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
"""
Improved Net_PPF (Full Code)
- Normalized Residual GCN (self-loop + D^{-1/2} A D^{-1/2})
- Posterior Predictive Fusion (PPF): mu/logvar + calibrated residual z
- Reliability gate: precision (1/var) * sigmoid(a) (해석/성능 둘 다)
- Low-Rank Bilinear Fusion (LRBF): (U hg) ⊙ (V v2_red)  (outer-product보다 과적합 덜함)
- Training: task loss + lambda_desc * NLL(desc | hg)
"""

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


# -----------------------------
# 2) Net: PPF + LRBF
# -----------------------------
class Net_PPF_LRBF(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        d_g: int = 60,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,          # descriptor representation reduction
        rank: int = 64,           # low-rank bilinear fusion dimension
        mlp_hidden: int = 128,
        dim_out: int = 1,
        logv_clip: float = 8.0,
        use_layernorm: bool = True,
        gate_max_precision: float = 10.0,
    ):
        super().__init__()
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = nn.Parameter(torch.tensor(gate_max_precision))

        # ---- GNN backbone ----
        self.gc1 = GCNConv(dim_in, 128)
        self.gc2 = GCNConv(128, d_g)

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
        
        # precision = precision.clamp(max=self.gate_max_precision)
        # 2. clamp 시 self.gate_max_precision 사용
        # 단, precision 상한선은 항상 양수여야 하므로 F.softplus나 .abs()를 고려할 수 있음
        max_p = torch.clamp(self.gate_max_precision, min=1e-3) 
        precision = precision.clamp(max=max_p)
        
        
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
