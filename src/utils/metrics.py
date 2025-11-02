import torch
from sklearn.metrics import precision_score, recall_score, jaccard_score


def compute_batch_metrics_binary(pred, target):
    # pred/target: (B,1,H,W) in {0,1}
    b = pred.shape[0]
    preds = pred.view(b, -1).cpu().numpy()
    tgts = target.view(b, -1).cpu().numpy()
    dices, ious, precs, recs = [], [], [], []
    for p, t in zip(preds, tgts):
        tp = (p * t).sum()
        dice = 2*tp / (p.sum() + t.sum() + 1e-6)
        iou = jaccard_score(t, p, zero_division=0)
        prec = precision_score(t, p, zero_division=0)
        rec = recall_score(t, p, zero_division=0)
        dices.append(dice); ious.append(iou); precs.append(prec); recs.append(rec)
    return dices, ious, precs, recs