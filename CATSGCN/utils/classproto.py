from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple

class ClassPrototypePriorLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        num_joints: int,
        num_frames: int,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        reduce_heads: Literal["mean", "sum", "max"] = "mean",
        use_ema_prototypes: bool = True,
        ema_momentum: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.C = num_classes
        self.d = embed_dim
        self.N = num_joints
        self.T = num_frames
        self.lambda_pos = float(lambda_pos)
        self.lambda_neg = float(lambda_neg)
        self.reduce_heads = reduce_heads
        self.use_ema = use_ema_prototypes
        self.m = float(ema_momentum)
        self.eps = eps

        self.proj_N = nn.Linear(self.d, self.N, bias=True)
        self.proj_T = nn.Linear(self.d, self.T, bias=True)

        if self.use_ema:
            self.register_buffer("prototypes", F.normalize(torch.randn(self.C, self.d), dim=1))
            self.register_buffer("proto_counts", torch.zeros(self.C, dtype=torch.long))
        else:
            self.prototypes = nn.Parameter(F.normalize(torch.randn(self.C, self.d), dim=1))
            self.register_buffer("proto_counts", torch.zeros(self.C, dtype=torch.long))

    @torch.no_grad()
    def ema_update(self, feats: torch.Tensor, targets: torch.Tensor) -> None:
        if not self.use_ema:
            return
        
        f = F.normalize(feats, dim=1)  # (B, d)
        for c in targets.unique():
            c = c.item()
            mask = (targets == c)
            if torch.any(mask):
                f_c = f[mask].mean(dim=0)

                self.prototypes[c] = F.normalize(self.m * self.prototypes[c] + (1.0 - self.m) * f_c, dim=0)
                self.proto_counts[c] += int(mask.sum().item())

    def _reduce_heads(self, A: torch.Tensor) -> torch.Tensor:
        if A.ndim == 4:
            if self.reduce_heads == "mean":
                A = A.mean(dim=1)
            elif self.reduce_heads == "sum":
                A = A.sum(dim=1)
            elif self.reduce_heads == "max":
                A, _ = A.max(dim=1)
            else:
                raise ValueError(f"Unknown reduce_heads: {self.reduce_heads}")
        elif A.ndim != 3:
            raise ValueError("Attention A must be (B,N,T) or (B,H,N,T).")
        return A

    def _pick_rival(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        P = F.normalize(self.prototypes, dim=1)  # (C, d)
        f = F.normalize(feats, dim=1)  # (B, d)

        sim = torch.matmul(f, P.t())  # (B, C)
        sim.scatter_(1, targets.view(-1, 1), float("-inf"))
        rival_idx = torch.argmax(sim, dim=1)  # (B,)
        return P[rival_idx]  # (B, d)

    def _project_to_attention_space(self, vecs: torch.Tensor, N: int, T: int) -> torch.Tensor:

        B = vecs.size(0)
        rN = self.proj_N(vecs)[:, :N] # (B,N)
        rT = self.proj_T(vecs)[:, :T] # (B,T)

        rel = torch.einsum("bn,bt->bnt", rN, rT).reshape(B, N * T)
        rel = F.normalize(rel, dim=1, eps=self.eps)
        return rel

    def forward(self, A: torch.Tensor, feats: torch.Tensor, targets: torch.Tensor, update_prototypes: bool = True) -> Tuple[torch.Tensor, dict]:
        B = A.size(0)

        if update_prototypes and self.use_ema:
            self.ema_update(feats.detach(), targets.detach())

        A = self._reduce_heads(A)

        _, N_in, T_in = A.shape
        A_flat = A.reshape(B, N_in * T_in)
        A_flat = F.normalize(A_flat, dim=1, eps=self.eps)

        P = self.prototypes # (C, d)
        p_pos = F.normalize(P[targets], dim=1, eps=self.eps)
        p_riv = self._pick_rival(feats, targets)

        R_pos = self._project_to_attention_space(p_pos, N=N_in, T=T_in)
        R_riv = self._project_to_attention_space(p_riv, N=N_in, T=T_in)

        cos_pos = torch.sum(A_flat * R_pos, dim=1)
        cos_riv = torch.sum(A_flat * R_riv, dim=1)

        loss = (-self.lambda_pos * cos_pos + self.lambda_neg * cos_riv).mean()

        stats = {
            "prior/cos_pos": cos_pos.mean().item(),
            "prior/cos_riv": cos_riv.mean().item(),
            "prior/loss": loss.item(),
        }
        return loss, stats
