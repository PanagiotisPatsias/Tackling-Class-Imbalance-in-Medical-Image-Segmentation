import torch
import torch.nn.functional as F
import torch.nn as nn

class DynaMit_BCE_ImageWise(nn.Module):
    def __init__(self):
        super(DynaMit_BCE_ImageWise, self).__init__()

    def forward(self, predictions, targets):
        """
        predictions: logits, shape [B, 1, H, W]
        targets: binary masks, shape [B, 1, H, W]
        """
        batch_size = predictions.size(0)
        loss_list = []

        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i].float() # Cast target to float

            S_pos = (target == 1).sum().float()
            S_neg = (target == 0).sum().float()

            pos_coeff = min(1.0, S_neg / S_pos) if S_pos > 0 else 0.0
            neg_coeff = min(1.0, S_pos / S_neg) if S_neg > 0 else 0.0

            weights = torch.where(target == 1, pos_coeff, neg_coeff)

            loss = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='mean')
            loss_list.append(loss)

        loss_final = torch.stack(loss_list).mean()
        return loss_final