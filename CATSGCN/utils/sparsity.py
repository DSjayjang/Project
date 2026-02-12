import torch
import torch.nn as nn

class MultiHeadSparsityLoss(nn.Module):
    def __init__(self, eps: float = 1e-12, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        assert reduction in ("mean", "sum", "none")
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

        # Entropy
        L_sparse_h = -torch.sum(A_tilde * torch.log(A_tilde+self.eps), axis=(2,3))

        # Average over heads
        L_sparse = L_sparse_h.mean(dim=1) # (B,)

        if self.reduction == "mean":
            return L_sparse.mean()
        elif self.reduction == "sum":
            return L_sparse.sum()
        else:
            return L_sparse  # (B,)