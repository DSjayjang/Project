from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .classproto import ClassPrototypePriorLoss
from .sparsity import MultiHeadSparsityLoss
from .diversity import MultiHeadDiversityLoss

class CATSUnifiedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        num_joints: int = 18,
        num_frames: int = 300,
        lambda_cp: float = 1.0,
        lambda_s: float = 1.0,
        lambda_d: float = 0.1,
        use_ema_prototypes: bool = True,
        ema_momentum: float = 0.99,
        sparse_reduction: str = "mean",
        div_reduction: str = "mean",
        reduce_heads_for_prior: str = "mean",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.lambda_cp = float(lambda_cp)
        self.lambda_s = float(lambda_s)
        self.lambda_d = float(lambda_d)

        # Cross-entropy
        self.ce = nn.CrossEntropyLoss()

        # Prior (class-prototype guided)
        self.prior = ClassPrototypePriorLoss(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_joints=num_joints,
            num_frames=num_frames,
            lambda_pos=1.0,
            lambda_neg=1.0,
            reduce_heads=reduce_heads_for_prior,
            use_ema_prototypes=use_ema_prototypes,
            ema_momentum=ema_momentum,
            eps=eps,
        )

        # Sparsity (multi-head)
        self.sparse = MultiHeadSparsityLoss(eps=eps, reduction=sparse_reduction)

        # Diversity (multi-head)
        self.div = MultiHeadDiversityLoss(eps=eps, reduction=div_reduction)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention: torch.Tensor,
        feats: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        update_prototypes: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        L_ce = self.ce(logits, targets)
        L_prior, prior_stats = self.prior(
            A=attention, feats=feats, targets=targets, update_prototypes=update_prototypes
        )
        L_sparse = self.sparse(attention, frame_mask=frame_mask)
        L_div = self.div(attention, frame_mask=frame_mask)

        loss = (
            L_ce
            + self.lambda_cp * L_prior
            + self.lambda_s * L_sparse
            + self.lambda_d * L_div
        )

        logs = {
            "loss/total": float(loss.item()),
            "loss/ce": float(L_ce.item()),
            "loss/prior": float(L_prior.item()),
            "loss/sparse": float(L_sparse.item()),
            "loss/div": float(L_div.item()),
            "prior/cos_pos": prior_stats.get("prior/cos_pos", 0.0),
            "prior/cos_riv": prior_stats.get("prior/cos_riv", 0.0),
        }
        return loss, logs