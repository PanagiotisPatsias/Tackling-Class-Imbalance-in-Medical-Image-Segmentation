import torch

def identify_axis(shape):
    # Three dimensional (B, C, D, H, W)
    if len(shape) == 5:
        return [2, 3, 4]
    # Two dimensional (B, C, H, W)
    elif len(shape) == 4:
        return [2, 3]
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D nor 3D.')

def Focal_tversky_loss(delta=0.3, gamma=4/3, smooth=1e-6):
    """
    Focal Tversky loss (PyTorch version)
    Paper: https://arxiv.org/abs/1810.07842
    
    Parameters
    ----------
    delta : float
        Controls weight given to FN vs FP. delta > 0.5 penalizes FN more.
    gamma : float
        Focal parameter; higher values down-weight easy examples.
    smooth : float
        Small constant to avoid division by zero.
    """
    def loss_function(y_true, y_pred):
        """
        y_true: one-hot encoded ground truth tensor (B, C, H, W) or (B, C, D, H, W)
        y_pred: probability tensor (after sigmoid/softmax) of same shape
        """
        # Clip values to prevent division by zero
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.shape)

        # True positives, false negatives, false positives
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=axis)

        # Tversky index
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)

        # Focal Tversky loss
        focal_loss = torch.pow((1 - tversky_class), gamma)

        return focal_loss.mean()

    return loss_function