import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryGaussian(nn.Module):
    """
    Balanced BCE:
      - w_pos = min(1, S_neg / S_pos)   (scalar, per-batch)
      - w_neg = (S_pos / S_neg) + w_bdry(x,y)   (map on BG; Gaussian distance from boundary)
    """
    def __init__(self, sigma=2.0, eps=1e-6):
        super().__init__()
        self.sigma = float(sigma)
        self.eps   = float(eps)

    @torch.no_grad()
    def _neg_boundary_weights(self, y01):
        """
        y01: (B,H,W) binary mask (1=FG).
        Returns w_bdry (B,H,W) with weights ONLY on the background.
        """
        B,H,W = y01.shape
        ws = []
        for i in range(B):
            m = y01[i].detach().float().cpu().numpy().astype(np.uint8)  # 1=FG, 0=BG
            dist_bg = distance_transform_edt(m == 0)                     # BG → nearest FG
            w_bdry = np.exp(-(dist_bg**2) / (2.0 * (self.sigma**2))).astype(np.float32)
            ws.append(torch.from_numpy(w_bdry))
        return torch.stack(ws, 0)  # (B,H,W)

    def forward(self, probs, targets):
        """
        probs  : (B,1,H,W) probabilities in (0,1)
        targets: (B,1,H,W) binary mask {0,1}
        """
        eps = self.eps
        y   = (targets > 0.5).float()           # (B,1,H,W)
        B, _, H, W = probs.shape

        # Batch statistics
        S_pos = y.sum()
        S_tot = y.numel()
        S_neg = S_tot - S_pos

        # Class weights (scalars)
        w_pos_batch = ((S_neg + eps) / (S_pos + eps)).clamp(max=1.0)   # ≤ 1
        w_neg_base  = (S_pos + eps) / (S_neg + eps)                    # small (<1) usually

        w_pos = w_pos_batch                                            # scalar tensor

        # Boundary weights (B,1,H,W)
        w_bdry = self._neg_boundary_weights(y[:, 0]).to(probs.device).unsqueeze(1)

        # Final negative weight (map)
        w_neg = (w_neg_base + w_bdry).clamp(max=1)                     # (B,1,H,W)

        # Numerical stability in logs
        p      = probs.clamp(min=eps, max=1.0 - eps)
        log_p  = torch.log(p)
        log_1p = torch.log1p(-p)

        # BCE terms
        loss_pos = y        * (-log_p)
        loss_neg = (1.0-y)  * (-log_1p)

        # Apply weights
        loss = w_neg * loss_neg + w_pos * loss_pos   # (B,1,H,W)

        return loss.mean()


