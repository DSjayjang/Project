"""
Net_PPF_LRBF using the PROVIDED GCNConv backbone (DO NOT TOUCH GCNConv)

- Graph embedding hg is produced by GCNConv (2-layer) + graph readout
- PPF on 2D (or 3D) descriptors conditioned on hg
- LRBF fusion: (U hg) ⊙ (V v_red)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


# -----------------------------
# (0) Provided GCNConv (DO NOT TOUCH)
# -----------------------------
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
# 2) Net: PPF + LRBF (2D descriptor)
# -----------------------------
class Net_PPF_LRBF(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        d_g: int = 20,
        d_hidden: int = 128,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,          # descriptor representation reduction
        rank: int = 64,           # low-rank bilinear fusion dimension
        mlp_hidden: int = 128,
        dim_out: int = 1,
        logv_clip: float = 8.0,
        gate_max_precision: float = 50.0,
    ):
        super().__init__()
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = gate_max_precision

        # ---- GNN backbone (GCNConv) Don't touch ----
        self.gc1 = GCNConv(dim_in, d_hidden)
        self.gc2 = GCNConv(d_hidden, d_g)

        # ---- PPF heads (2D) conditioned on hg ----
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

    def forward(self, g: dgl.DGLGraph, desc_2d: torch.Tensor, desc_3d: torch.Tensor = None, return_aux: bool = False):
        # ---- GNN (GCNConv) ----
        x = g.ndata["feat"]
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

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


# -----------------------------
# Net: PPF + LRBF (2D descriptor)
#   - d_g is FIXED to 20
#   - "최적"을 목표로 한 실전 기본값 세팅:
#       * hidden=96 (20-dim bottleneck이라 과한 hidden은 과적합 유발)
#       * drop_gnn=0.15, drop_head=0.30 (분자 데이터셋에서 흔히 안정적)
#       * v_red=48 (descriptor가 크면 32는 정보손실이 커지는 경우 많음)
#       * rank=96 (LRBF 표현력 보강; d_g=20일 때 64보다 안정적 성능 흔함)
#       * mlp_hidden=192 (fusion 이후 비선형 헤드 용량 확보)
#       * logv_clip=6.0 (var 폭주 방지 + calibration 안정)
#       * gate_max_precision=25.0 (precision 과대 -> v2 폭주 방지)
#       * gate bias를 음수로 초기화해 초반 과신 방지
# -----------------------------
class Net_PPF_LRBF3(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        # graph embedding dimension FIXED
        d_g: int = 20,

        # ---- tuned defaults (you can override) ----
        d_hidden: int = 96,
        drop_gnn: float = 0.15,
        drop_head: float = 0.30,
        v_red: int = 48,
        rank: int = 96,
        mlp_hidden: int = 192,
        dim_out: int = 1,

        logv_clip: float = 6.0,
        gate_max_precision: float = 25.0,
    ):
        super().__init__()
        # assert d_g == 20, "d_g는 20으로 고정해서 쓰는 설정입니다."
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = gate_max_precision
        self.drop_gnn = nn.Dropout(drop_gnn)

        # ---- GNN backbone (GCNConv) ----
        self.gc1 = GCNConv(dim_in, d_hidden)
        self.gc2 = GCNConv(d_hidden, d_g)

        # (optional but strongly recommended) stabilize 20-dim embedding
        self.hg_ln = nn.LayerNorm(d_g)

        # ---- PPF heads (2D) conditioned on hg ----
        self.mu_2d   = nn.Linear(d_g, dim_2d_desc)
        self.logv_2d = nn.Linear(d_g, dim_2d_desc)
        self.a_2d    = nn.Linear(d_g, dim_2d_desc)   # gate logits

        # gate를 초반에 보수적으로: sigmoid(bias) < 0.5
        nn.init.constant_(self.a_2d.bias, -1.5)

        # ---- reduce descriptor channel before fusion ----
        self.v_reduction = nn.Sequential(
            nn.Linear(dim_2d_desc, v_red),
            nn.LayerNorm(v_red),
            nn.GELU(),
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
            nn.GELU(),
            nn.Dropout(drop_head),
            nn.Linear(mlp_hidden, dim_out),
        )

    def _ppf_2d(self, hg: torch.Tensor, x2d: torch.Tensor):
        """
        hg:  (B, 20)
        x2d: (B, D2)
        returns:
          v2:   (B, D2) fused descriptor representation (w * z)
          nll:  (B,)    Gaussian NLL per sample (sum over D2)
          aux:  dict
        """
        mu = self.mu_2d(hg)  # (B, D2)

        logv = self.logv_2d(hg).clamp(-self.logv_clip, self.logv_clip)
        var = torch.exp(logv) + self.eps
        sigma = torch.sqrt(var)

        # calibrated residual
        z = (x2d - mu) / (sigma + self.eps)

        # reliability: precision * sigmoid(gate)
        precision = (1.0 / var).clamp(max=self.gate_max_precision)
        gate = torch.sigmoid(self.a_2d(hg))
        w = gate * precision

        v2 = w * z

        # Gaussian NLL (omit constant)
        nll_elem = 0.5 * (((x2d - mu) ** 2) / var + torch.log(var))
        nll = nll_elem.sum(dim=1)

        aux = {
            "mu": mu,
            "logv": logv,
            "sigma": sigma,
            "precision": precision,
            "gate": gate,
            "w": w,
            "z": z,
        }
        return v2, nll, aux

    def forward(
        self,
        g: dgl.DGLGraph,
        desc_2d: torch.Tensor,
        desc_3d: torch.Tensor = None,
        return_aux: bool = False
    ):
        # ---- GNN (GCNConv) ----
        x = g.ndata["feat"]

        h = self.gc1(g, x)
        h = F.relu(self.drop_gnn(h))
        h = self.gc2(g, h)
        h = F.relu(self.drop_gnn(h))

        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")          # (B, 20)
        hg = self.hg_ln(hg)                  # 안정화 (추천)

        # ---- PPF (2D) ----
        v2, nll2, aux2 = self._ppf_2d(hg, desc_2d)

        # ---- reduce + LRBF ----
        v2_red = self.v_reduction(v2)        # (B, v_red)
        fuse = self.U(hg) * self.V(v2_red)   # (B, rank)
        fuse = self.fuse_ln(fuse)

        out = self.head(fuse)                # (B, 1)

        if not return_aux:
            return out
        return out, {"hg": hg, "nll_2d": nll2, "aux_2d": aux2, "v2_red": v2_red, "fuse": fuse}






# -----------------------------
# 3) Net: PPF + LRBF (3D descriptor only) — 원문 스타일 유지
# -----------------------------
class Net_PPF_LRBF2(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,   # signature 유지 (unused)
        dim_3d_desc: int,
        d_g: int = 64,
        d_hidden: int = 128,
        drop_gnn: float = 0.1,
        drop_head: float = 0.2,
        v_red: int = 32,
        rank: int = 64,
        mlp_hidden: int = 128,
        dim_out: int = 1,
        logv_clip: float = 8.0,
        gate_max_precision: float = 50.0,
    ):
        super().__init__()
        self.eps = 1e-6
        self.logv_clip = logv_clip
        self.gate_max_precision = gate_max_precision

        # ---- GNN backbone (GCNConv) ----
        self.backbone = GCNBackbone(dim_in=dim_in, d_hidden=d_hidden, d_g=d_g, drop_gnn=drop_gnn)

        # ---- PPF heads (3D) ----
        self.mu_3d   = nn.Linear(d_g, dim_3d_desc)
        self.logv_3d = nn.Linear(d_g, dim_3d_desc)
        self.a_3d    = nn.Linear(d_g, dim_3d_desc)

        # ---- reduce descriptor channel before fusion ----
        self.v_reduction = nn.Sequential(
            nn.Linear(dim_3d_desc, v_red),
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

    def _ppf_3d(self, hg: torch.Tensor, x3d: torch.Tensor):
        mu = self.mu_3d(hg)
        logv = self.logv_3d(hg).clamp(-self.logv_clip, self.logv_clip)
        var = torch.exp(logv) + self.eps
        sigma = torch.sqrt(var)

        z = (x3d - mu) / (sigma + self.eps)

        precision = (1.0 / var).clamp(max=self.gate_max_precision)
        gate = torch.sigmoid(self.a_3d(hg))
        w = gate * precision

        v3 = w * z

        nll_elem = 0.5 * ((x3d - mu) ** 2 / var + torch.log(var))
        nll = nll_elem.sum(dim=1)

        aux = {"mu": mu, "logv": logv, "sigma": sigma, "w": w, "z": z, "gate": gate, "precision": precision}
        return v3, nll, aux

    def forward(self, g: dgl.DGLGraph, desc_2d: torch.Tensor, desc_3d: torch.Tensor, return_aux: bool = False):
        # ---- GNN (GCNConv) ----
        x = g.ndata["feat"]
        _, hg = self.backbone(g, x)

        # ---- PPF (3D) ----
        v3, nll3, aux3 = self._ppf_3d(hg, desc_3d)

        # ---- reduce + LRBF ----
        v3_red = self.v_reduction(v3)
        fuse = self.U(hg) * self.V(v3_red)
        fuse = self.fuse_ln(fuse)

        out = self.head(fuse)

        if not return_aux:
            return out

        # 원문 코드의 key naming을 그대로 유지
        return out, {"hg": hg, "nll_2d": nll3, "aux_2d": aux3, "v2_red": v3_red, "fuse": fuse}
