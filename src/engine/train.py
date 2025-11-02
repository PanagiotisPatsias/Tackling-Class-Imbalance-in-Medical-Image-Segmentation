import argparse, yaml, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.utils.seed import set_seed
from src.utils.logging import log_print
from src.utils.augment import default_aug
from src.utils.metrics import compute_batch_metrics_binary
from src.datasets.cvc import CVC as _CVC
from pathlib import Path as _P
from src.models.unet import UNet2D
from src.datasets.isic import ISICDataset
from src.datasets.drive import DRIVE
import numpy as _np
from src.utils.metrics import compute_batch_metrics_binary
import numpy as np, scipy.stats as st
from src.datasets.cvc import CVC
from sklearn.model_selection import train_test_split
from src.utils.loss_router import (
get_loss_spec, apply_activation_for_loss, prepare_targets, probs_for_metrics
)


# ---- loss builder ----
import torch.nn.functional as F
from src.losses.Dice import DSC
from src.losses.Focal_Loss import focal_loss
from src.losses.Tversky import tversky_loss
from src.losses.Focal_Tversky import Focal_tversky_loss
from src.losses.Combo import ComboLoss
from src.losses.Dynamit_BCE_ImageWise import DynaMit_BCE_ImageWise
from src.losses.Dynamit_BCE_BatchWise import DynaMit_BCE_BatchWise
from src.losses.Boundary_DoU_Loss import MixedBoundaryGaussian_BoundaryDoU
from src.losses.Boundary_Gaussian_Loss import BoundaryGaussian
from src.losses.Unified_focal_loss import AsymmetricUnifiedFocalLoss
from src.losses.Unified_focal_loss import SymmetricUnifiedFocalLoss
from src.losses.CE import CE
import torch
import torch.nn as nn
import inspect, math

def make_criterion(sym, params=None, device=None):
    """
    Normalize any loss into: criterion(pred, target) -> scalar Tensor
    Supports:
      - function: loss(pred, target, **params) -> Tensor
      - class(nn.Module): Loss(**params); then call instance(pred, target) -> Tensor
      - factory: LossFactory(**params) -> (pred, target)->Tensor
    """
    params = params or {}

    # nn.Module class → instantiate once
    if isinstance(sym, type) and issubclass(sym, nn.Module):
        inst = sym(**params)
        try:
            next(inst.parameters())
            if device is not None:
                inst.to(device)
        except StopIteration:
            pass
        return lambda pred, tgt: inst(pred, tgt)

    if callable(sym):
        
        def as_func(pred, tgt):
            return sym(pred, tgt, **params)
        try:
            
            inspect.signature(sym).bind_partial(None, None, **params)
            
            out = as_func(torch.zeros(1,1,1,1), torch.zeros(1,1,1,1))
            if callable(out):  # factory misused
                built = sym(**params)
                return lambda pred, tgt: built(pred, tgt)
            return lambda pred, tgt: sym(pred, tgt, **params)
        except TypeError:
            
            built = sym(**params)
            if not callable(built):
                raise
            return lambda pred, tgt: built(pred, tgt)

    raise TypeError(f"Cannot adapt loss symbol {sym} ({type(sym)})")



def build_loss(name, params):
    if name == 'dice':
        return make_criterion(DSC, params, device)
    if name == 'focal_loss':
        return make_criterion(focal_loss, params, device)
    if name == 'tversky':
        return make_criterion(tversky_loss, params, device)
    if name == 'focal_tversky':
        return make_criterion(Focal_tversky_loss, params, device)
    if name == 'combo':
        return make_criterion(ComboLoss, params, device)
    if name == 'ce':
        return make_criterion(CE, params, device)
    if name == 'dynamit_bce_batch':
        return make_criterion(DynaMit_BCE_BatchWise, params, device)
    if name == 'dynamit_bce_image':
        return make_criterion(DynaMit_BCE_ImageWise, params, device)  
    if name == 'boundary_gaussian':
        return make_criterion(BoundaryGaussian, params, device)
    if name == 'boundary_dou':
        return make_criterion(MixedBoundaryGaussian_BoundaryDoU, params, device)
    if name == 'asymmetric_unified_focal_loss':
        return make_criterion(AsymmetricUnifiedFocalLoss, params, device)
    if name == 'symmetric_unified_focal_loss':
        return make_criterion(SymmetricUnifiedFocalLoss, params, device)
    raise ValueError('Unknown loss')


# ---- helpers for monitoring ----
def get_monitored_value(metric_name, va_loss, va_dice):
    if metric_name == 'val_loss':
        return va_loss
    if metric_name == 'dice':
        return va_dice
    raise ValueError(f"Unknown monitor metric: {metric_name}")

def make_is_better(mode):
    if mode == 'min':
        return lambda new, best: new < best
    if mode == 'max':
        return lambda new, best: new > best
    raise ValueError(f"Unknown monitor mode: {mode}")

def is_nan(x):
    return isinstance(x, float) and math.isnan(x)

# ---- args ----
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))


# ---- seed/logs ----
set_seed(cfg.get('seed',42))
out_dir = Path(cfg['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
loss_name = cfg['loss']['name']
log_file = out_dir / f"{loss_name}.txt"


# ---- dataset/loaders ----
aug = default_aug()
size = tuple(cfg['image_size'])
bs = cfg['batch_size']; nw = cfg['num_workers']
root = cfg['data_dir']


if cfg['dataset'] == 'isic':
    train_ds = ISICDataset(root, split='training', aug=None, image_size=size)
    val_ds = ISICDataset(root, split='validation', image_size=size)
    test_ds = ISICDataset(root, split='test', image_size=size)
elif cfg['dataset'] == 'drive':
    full = DRIVE(root, mode='training', size=size)
    idx = list(range(len(full)))
    train_idx, val_idx = train_test_split(idx, test_size=cfg.get('val_size',4), random_state=42, shuffle=True)
    train_ds = Subset(DRIVE(root, mode='training', size=size, aug=aug), train_idx)
    val_ds = Subset(DRIVE(root, mode='training', size=size), val_idx)
    test_ds = DRIVE(root, mode='test', size=size)
elif cfg['dataset'] == 'cvc':
    IMG_DIR = _P(root)/'CVC-ClinicDB'/'Original'
    MSK_DIR = _P(root)/'CVC-ClinicDB'/'Ground Truth'
    img_paths = sorted(list(IMG_DIR.glob('*.tif')))
    msk_paths = [MSK_DIR / p.with_suffix('.tif').name for p in img_paths]
    idx = np.arange(len(img_paths)); np.random.seed(42); np.random.shuffle(idx)
    ntr, nval = cfg['splits']['train'], cfg['splits']['val']
    train_idx, val_idx, test_idx = idx[:ntr], idx[ntr:ntr+nval], idx[ntr+nval:]
    train_ds = _CVC(train_idx, img_paths, msk_paths, target_size=size, aug=aug)
    val_ds = _CVC(val_idx, img_paths, msk_paths, target_size=size)
    test_ds = _CVC(test_idx, img_paths, msk_paths, target_size=size)
else:
    raise ValueError('Unknown dataset')


train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

# ---- model via loss spec ----
loss_name = cfg['loss']['name']
spec = get_loss_spec(loss_name)
mcfg = cfg['model']
Cout = spec.required_out_classes #how many output channels the model must have
model = UNet2D(in_channels=mcfg['in_channels'], out_classes=Cout, base=mcfg.get('base_channels',64))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ---- monitoring config ----
mon_cfg = cfg.get('monitor', {})
monitor_metric = mon_cfg.get('metric', 'val_loss')  # 'val_loss' or 'dice'
monitor_mode   = mon_cfg.get('mode',   'min')       # 'min' or 'max'
is_better = make_is_better(monitor_mode)
best_val = float('inf') if monitor_mode == 'min' else -float('inf')
bad_epochs = 0
patience = cfg['early_stopping']['patience']

# ---- loss/opt/sched ----
criterion = build_loss(loss_name, cfg['loss'].get('params', {}))
opt = torch.optim.SGD(model.parameters(), lr=cfg['optimizer']['lr'], momentum=cfg['optimizer']['momentum'], weight_decay=float(cfg['optimizer']['weight_decay']))
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode=monitor_mode, factor=cfg['lr_scheduler']['factor'],
    patience=cfg['lr_scheduler']['patience']
)



# ---- train/val loops ----
for epoch in range(1, cfg['max_epochs']+1):
    # train
    model.train(); total=0.0
    for img, msk in train_dl:
        img, msk = img.to(device), msk.to(device)
        target_for_loss = prepare_targets(msk, spec, Cout)
        logits = model(img)
        input_for_loss = apply_activation_for_loss(logits, spec)
        opt.zero_grad(set_to_none=True)
        loss = criterion(input_for_loss, target_for_loss)
        loss.backward(); opt.step()
        total += loss.item()*img.size(0)
    tr_loss = total/len(train_dl.dataset)
    # val
    model.eval(); total=0.0; dices=[]
    with torch.no_grad():
        for img, msk in val_dl:
            img, msk = img.to(device), msk.to(device)
            target_for_loss = prepare_targets(msk, spec, Cout)
            logits = model(img)
            input_for_loss = apply_activation_for_loss(logits, spec)
            total += criterion(input_for_loss, target_for_loss).item()*img.size(0)

            probs = probs_for_metrics(logits, spec)
            if Cout == 1:
                pred = (probs > cfg.get('threshold',0.5)).float()
                target = (msk.unsqueeze(1) == 1).float()
            else:
                pred_idx = probs.argmax(dim=1)
                pred = (pred_idx == 1).float().unsqueeze(1)
                target = (msk == 1).float().unsqueeze(1)
            d,_,_,_ = compute_batch_metrics_binary(pred, target)
            dices.append(float(_np.mean(d)))
    va_loss = total/len(val_dl.dataset)
    va_dice = (sum(dices)/len(dices)) if dices else float('nan')
    # value to monitor (dice or val_loss)
    monitored = get_monitored_value(monitor_metric, va_loss, va_dice)
    # scheduler step (skip if NaN)
    if not is_nan(monitored):
        sched.step(monitored)


    line = f"Epoch {epoch:03d} Train {tr_loss:.4f} Val {va_loss:.4f} Dice {va_dice:.4f} LR {opt.param_groups[0]['lr']:.3g}"
    log_print(line, str(log_file))
    # ckpt
    ckpt = out_dir / f"{loss_name}.ckpt"
    if not is_nan(monitored) and is_better(monitored, best_val):
        best_val = monitored; bad_epochs = 0
        torch.save({'model': model.state_dict(), 'cfg': cfg}, ckpt)
    else:
        
        if not is_nan(monitored):
            bad_epochs += 1

    if bad_epochs >= patience:
        log_print('Early stopping', str(log_file))
        break

# ---- evaluate best checkpoint on VALIDATION set ----
state = torch.load(out_dir / f"{loss_name}.ckpt", map_location=device)
model.load_state_dict(state['model'])
model.eval()

all_d, all_i, all_p, all_r = [], [], [], []

with torch.no_grad():
    for img, msk in val_dl: 
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        probs = probs_for_metrics(logits, spec)

        if Cout == 1:
            pred = (probs > cfg.get('threshold', 0.5)).float()
            target = (msk.unsqueeze(1) == 1).float()
        else:
            pred_idx = probs.argmax(dim=1)
            pred = (pred_idx == 1).float().unsqueeze(1)
            target = (msk == 1).float().unsqueeze(1)

        d, i, p, r = compute_batch_metrics_binary(pred, target)
        all_d += d
        all_i += i
        all_p += p
        all_r += r

def ci(arr):
    arr = np.array(arr)
    return arr.mean(), 1.96 * st.sem(arr)

log_print("\n=== Validation metrics (best weights) ===", str(log_file))
for name, arr in [('Dice', all_d), ('IoU', all_i), ('Precision', all_p), ('Recall', all_r)]:
    m, c = ci(arr)
    line = f"{name:9s}: {m:.3f} ± {c:.3f}"
    print(line)
    log_print(line, str(log_file))
