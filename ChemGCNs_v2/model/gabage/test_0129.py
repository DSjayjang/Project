import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

# =============================
# 1) Your original GCNLayer
# =============================
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
           
class GCNConvLayerNorm(nn.Module):
    """
    기존 GCNLayerNorm과 같은 구조:
      Kipf GCNConv (+residual) -> LN -> ReLU -> Dropout
    """
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.conv = GCNConv(dim_in, dim_out, bias=False)  # Kipf conv
        self.res  = nn.Linear(dim_in, dim_out, bias=False) if dim_in != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_out) if use_ln else nn.Identity()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            h = self.conv(g, x)      # D^-1/2 A~ D^-1/2 X W
            h = h + self.res(x)      # residual
            h = self.ln(h)           # layernorm (optional)
            h = F.relu(h)
            h = self.dropout(h)
            return h
class Net_PPF_LRBF(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        d_g: int = 64,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,
        rank: int = 64,
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

        # ---- GNN backbone ----  (교체됨)
        # self.gc1 = GCNConvLayerNorm(dim_in, 128, dropout=drop_gnn, use_ln=use_layernorm)
        # self.gc2 = GCNConvLayerNorm(128, d_g,  dropout=drop_gnn, use_ln=use_layernorm)
        self.gc1 = GCNConv(dim_in, 128)
        self.gc2 = GCNConv(128, d_g)

        # ---- PPF heads (2D) ----
        self.mu_2d   = nn.Linear(d_g, dim_2d_desc)
        self.logv_2d = nn.Linear(d_g, dim_2d_desc)
        self.a_2d    = nn.Linear(d_g, dim_2d_desc)

        # ---- reduce descriptor channel before fusion ----
        self.v_reduction = nn.Sequential(
            nn.Linear(dim_2d_desc, v_red),
            nn.LayerNorm(v_red),
            nn.ReLU(),
            nn.Dropout(drop_head),
        )

        # ---- Low-Rank Bilinear Fusion ----
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
        mu = self.mu_2d(hg)
        logv = self.logv_2d(hg).clamp(-self.logv_clip, self.logv_clip)
        var = torch.exp(logv) + self.eps
        sigma = torch.sqrt(var)

        z = (x2d - mu) / (sigma + self.eps)

        precision = 1.0 / var
        precision = precision.clamp(max=self.gate_max_precision)
        gate = torch.sigmoid(self.a_2d(hg))
        w = gate * precision

        v2 = w * z

        nll_elem = 0.5 * ((x2d - mu) ** 2 / var + torch.log(var))
        nll = nll_elem.sum(dim=1)

        aux = {"mu": mu, "logv": logv, "sigma": sigma, "w": w, "z": z, "gate": gate, "precision": precision}
        return v2, nll, aux

    def forward(self, g: dgl.DGLGraph, desc_2d: torch.Tensor, desc_3d: torch.Tensor, return_aux: bool = False):
        x = g.ndata["feat"]
        h = F.relu(self.gc1(g, x))
        h = F.relu(self.gc2(g, h))
        
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")

        v2, nll2, aux2 = self._ppf_2d(hg, desc_2d)

        v2_red = self.v_reduction(v2)
        fuse = self.U(hg) * self.V(v2_red)
        fuse = self.fuse_ln(fuse)

        out = self.head(fuse)

        if not return_aux:
            return out
        return out, {"hg": hg, "nll_2d": nll2, "aux_2d": aux2, "v2_red": v2_red, "fuse": fuse}
