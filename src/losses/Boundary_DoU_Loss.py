import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# ============================================================
#  1. Boundary Gaussian Loss
# ============================================================
class BoundaryGaussian(nn.Module):
    """
    Balanced BCE:
      - w_pos = min(1, S_neg / S_pos)   (scalar, per batch)
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
            dist_bg = distance_transform_edt(m == 0)                     # BG -> nearest FG
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
        w_neg_base  = (S_pos + eps) / (S_neg + eps)                    # usually small (<1)

        w_pos = w_pos_batch                                            # scalar tensor

        # Boundary weights (B,1,H,W)
        w_bdry = self._neg_boundary_weights(y[:, 0]).to(probs.device).unsqueeze(1)

        # Final negative weight (map)
        w_neg = (w_neg_base + w_bdry).clamp(max=1)                     # (B,1,H,W)

        # Log safety
        p      = probs.clamp(min=eps, max=1.0 - eps)
        log_p  = torch.log(p)
        log_1p = torch.log1p(-p)

        # BCE terms
        loss_pos = y        * (-log_p)
        loss_neg = (1.0-y)  * (-log_1p)

        # Apply weights
        loss = w_neg * loss_neg + w_pos * loss_pos   # (B,1,H,W)

        return loss.mean()


# ============================================================
# 2. Boundary Difference-over-Union Loss (DoU)
# ============================================================
class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes, from_logits=True):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes
        self.from_logits = bool(from_logits)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        device = target.device
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]]).to(device)
        h, w = 3, 3

        # Padding to handle boundaries
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2]+2, target.shape[-1]+2),
            device=device
        )
        padding_out[:, 1:-1, 1:-1] = target

        # Convolution to detect boundary neighborhood
        Y = torch.zeros(
            (padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1),
            device=device
        )
        for i in range(Y.shape[0]):
            Y[i, :, :] = F.conv2d(
                target[i].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            )
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        alpha = torch.clamp(alpha, max=0.8)

        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )
        return loss

    def forward(self, inputs, target):
        """
        from_logits=False → expects probabilities:
          - If [B,1,H,W]: will convert to [B,2,H,W] as (1-p, p)
          - If [B,2,H,W]: used as is (no softmax)
        target: [B,H,W] or [B,1,H,W]
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]  # -> [B,H,W]

        if self.from_logits:
            inputs = torch.softmax(inputs, dim=1)
        else:
            if inputs.size(1) == 1 and self.n_classes == 2:
                p = inputs[:, 0].clamp(1e-7, 1-1e-7)
                inputs = torch.stack([1.0 - p, p], dim=1)  # [B,2,H,W]
            else:
                inputs = inputs.clamp(1e-7, 1-1e-7)

        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), (
            f"predict {inputs.size()} & target {target.size()} shape do not match"
        )

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


# ============================================================
# 3. Combined Loss: L = L_BG + λ * L_DoU
# ============================================================
class MixedBoundaryGaussian_BoundaryDoU(nn.Module):
    """
    Final combined loss for binary segmentation:
        L = L_BG + λ * L_DoU
    Takes probs [B,1,H,W] (sigmoid output)
    """
    def __init__(self, lambda_dou: float = 1.0, sigma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.lambda_dou = float(lambda_dou)
        self.balanced_bce = BoundaryGaussian(sigma=sigma, eps=eps)
        self.dou = BoundaryDoULoss(n_classes=2, from_logits=False)

    def forward(self, probs: torch.Tensor, targets: torch.Tensor):
        loss_dyn = self.balanced_bce(probs, targets)
        loss_dou = self.dou(probs, targets)
        return loss_dyn + self.lambda_dou * loss_dou
