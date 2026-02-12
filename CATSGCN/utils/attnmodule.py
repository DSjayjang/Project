import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AttentionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        num_person: int = 2,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.H = num_heads
        self.M = num_person
        self.eps = eps

        self.score = nn.Conv2d(in_channels, num_heads, kernel_size=1)

    def forward(self, h: torch.Tensor, B: Optional[int] = None) -> torch.Tensor:
        if h.ndim != 4:
            raise ValueError(f"h must be 4D (B*M,C,T,V). Got {h.shape}")
        BM, C, T, V = h.shape

        if B is None:
            if BM % self.M != 0:
                raise ValueError(f"Cannot infer B: BM({BM}) not divisible by M({self.M}).")
            B = BM // self.M

        s = self.score(h)
        s = s.view(B, self.M, self.H, T, V).mean(dim=1)
        a = F.softplus(s) + self.eps
        A = a.permute(0, 1, 3, 2).contiguous()
        
        return A
