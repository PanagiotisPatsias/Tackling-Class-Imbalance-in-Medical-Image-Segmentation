import torch
import torch.nn as nn


def identify_axis_torch(shape):
    """
    Identify axes for 2D or 3D segmentation (NCHW or NCDHW)
    """
    if len(shape) == 5:  # 3D: (B, C, D, H, W)
        return (2, 3, 4)
    elif len(shape) == 4:  # 2D: (B, C, H, W)
        return (2, 3)
    else:
        raise ValueError('Shape of tensor is neither 2D or 3D.')

def dice_coefficient_torch(y_true, y_pred, delta=0.5, smooth=1e-6):
    """
    Compute Dice coefficient for PyTorch
    """
    axis = identify_axis_torch(y_true.shape)
    tp = torch.sum(y_true * y_pred, dim=axis)
    fn = torch.sum(y_true * (1 - y_pred), dim=axis)
    fp = torch.sum((1 - y_true) * y_pred, dim=axis)

    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    return torch.mean(dice_class)

class ComboLoss(nn.Module):
    """
    Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    https://arxiv.org/abs/1805.02798
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: [B, C, H, W] or [B, C, D, H, W] - probabilities (after sigmoid/softmax)
        y_true: one-hot encoded target of same shape
        """
        # Ensure probabilities are clipped
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        # Dice coefficient
        dice = dice_coefficient_torch(y_true, y_pred, delta=0.5, smooth=self.smooth)

        # Weighted Cross Entropy
        ce = -y_true * torch.log(y_pred)
        if self.beta is not None:
            beta_weight = torch.tensor([self.beta, 1 - self.beta], device=y_true.device)
            beta_weight = beta_weight.view(1, -1, *([1] * (y_true.ndim - 2)))  # reshape for broadcasting
            ce = beta_weight * ce

        ce = torch.mean(torch.sum(ce, dim=1))  # sum over classes, mean over batch

        # Combo Loss
        if self.alpha is not None:
            combo = (self.alpha * ce) + ((1 - self.alpha) * (1 - dice))
        else:
            combo = ce + (1 - dice)

        return combo