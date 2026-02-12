import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadDiversityLoss(nn.Module):
    def __init__(self, eps: float = 1e-12, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.eps = eps
        self.reduction = reduction

    def forward(self, A: torch.Tensor, frame_mask: torch.Tensor = None) -> torch.Tensor:
        if A.ndim == 3:
            A = A.unsqueeze(1) # (B,N,T) -> (B,1,N,T)
        elif A.ndim != 4:
            raise ValueError("A must be (B,H,N,T) or (B,N,T).")

        B, H, N, T = A.shape

        # frame_mask: allow (B,T), (B,1,T), (B,1,1,T)
        if frame_mask is not None:
            if frame_mask.ndim == 2:          # (B,T)
                frame_mask = frame_mask[:, None, None, :]     # (B,1,1,T)
            elif frame_mask.ndim == 3:        # (B,1,T)
                if frame_mask.shape[1] != 1:
                    raise ValueError("frame_mask with ndim=3 must be (B,1,T).")
                frame_mask = frame_mask[:, None, :, :]        # (B,1,1,T) if middle dim==1
            elif frame_mask.ndim == 4:        # (B,1,1,T)
                if frame_mask.shape[1:3] != (1,1):
                    raise ValueError("frame_mask with ndim=4 must be (B,1,1,T).")
            else:
                raise ValueError("frame_mask must be (B,T), (B,1,T), or (B,1,1,T).")

            frame_mask = frame_mask.view(B, 1, 1, T)
            A = A * frame_mask  # zero-out invalid frames

        # Head-wise normalization by total sum over (N,T)
        sums = A.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps) # (B,H,1,1)
        A_tilde = A / sums # (B,H,N,T)

        # Vectorize per head
        A_vec = A_tilde.reshape(B, H, N*T)

        # L2-normalize per head vector
        A_hat = F.normalize(A_vec, dim=2, eps=self.eps) # (B,H,NT)

        # If H == 1, no pair exists -> loss = 0
        if H <= 1:
            zero = A_hat.new_zeros(B)
            if self.reduction == "mean":
                return zero.mean()
            elif self.reduction == "sum":
                return zero.sum()
            else:
                return zero  # (B,)

        # squared cosine similarity
        G = torch.bmm(A_hat, A_hat.transpose(1, 2)) # (B,H,H)
        G2 = G.pow(2)                                        

        triu_mask = torch.triu(torch.ones(H, H, dtype=torch.bool, device=A_hat.device), diagonal=1)
        pair_sums = G2[:, triu_mask].sum(dim=1) # (B,)

        L_div_per_sample = (2.0 / (H * (H - 1))) * pair_sums # (B,)

        # Reduction over batch
        if self.reduction == "mean":
            return L_div_per_sample.mean()
        elif self.reduction == "sum":
            return L_div_per_sample.sum()
        else:
            return L_div_per_sample  # (B,)