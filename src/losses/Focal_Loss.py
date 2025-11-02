import numpy as np
import torch

def identify_axis(shape):
    # shape is a torch.Size object in PyTorch
    if len(shape) == 5:   # e.g., (B, C, D, H, W) - 3D volumes
        return [2, 3, 4]
    elif len(shape) == 4: # e.g., (B, C, H, W) - 2D images
        return [2, 3]
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def focal_loss(alpha=0.25, gamma=2.0):
    """
    PyTorch equivalent of the provided Keras focal loss implementation.
    
    Parameters
    ----------
    alpha : float, list, or None
        Class weight(s) for balancing. If >0.5 for binary, penalizes FN more than FP.
    gamma_f : float
        Focal parameter controlling down-weighting of easy examples.
    """
    def loss_function(y_true, y_pred):
        # y_true: one-hot encoded tensor [B, C, H, W] or [B, C, D, H, W]
        # y_pred: probability tensor (softmax or sigmoid applied) of same shape

        axis = identify_axis(y_true.shape)
        epsilon = 1e-7  # Similar to K.epsilon()

        # Clip probabilities for numerical stability
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        # Cross entropy term
        cross_entropy = -y_true * torch.log(y_pred)

        # Apply alpha weighting if provided
        if alpha is not None:
            alpha_weight = torch.tensor(np.array(alpha, dtype=np.float32),
                                        device=y_pred.device).view([1, -1] + [1] * (len(y_true.shape) - 2))
            focal_loss_tensor = alpha_weight * (1 - y_pred) ** gamma * cross_entropy
        else:
            focal_loss_tensor = (1 - y_pred) ** gamma * cross_entropy

        # Sum over classes, then take mean
        focal_loss_tensor = focal_loss_tensor.sum(dim=1) 
        return focal_loss_tensor.mean()

    return loss_function