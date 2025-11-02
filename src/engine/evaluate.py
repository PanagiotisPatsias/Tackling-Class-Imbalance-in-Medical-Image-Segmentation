import argparse
import yaml
import torch
import numpy as np
import scipy.stats as st
from pathlib import Path
from torch.utils.data import DataLoader

# imports from your project
from src.models.unet import UNet2D
from src.utils.loss_router import get_loss_spec, probs_for_metrics
from src.utils.metrics import compute_batch_metrics_binary
from src.datasets.isic import ISICDataset
from src.datasets.drive import DRIVE
from src.datasets.cvc import CVC as CVCData


def build_test_loader(cfg):
    """Rebuild test loader exactly as in train.py"""
    size = tuple(cfg['image_size'])
    root = cfg['data_dir']

    if cfg['dataset'] == 'isic':
        test_ds = ISICDataset(root, split='test', image_size=size)
    elif cfg['dataset'] == 'drive':
        test_ds = DRIVE(root, mode='test', size=size)
    elif cfg['dataset'] == 'cvc':
        IMG_DIR = Path(root) / 'CVC-ClinicDB' / 'Original'
        MSK_DIR = Path(root) / 'CVC-ClinicDB' / 'Ground Truth'
        img_paths = sorted(list(IMG_DIR.glob('*.tif')))
        msk_paths = [MSK_DIR / p.with_suffix('.tif').name for p in img_paths]
        idx = np.arange(len(img_paths))
        test_ds = CVCData(idx, img_paths, msk_paths, target_size=size)
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=True,
    )
    return test_dl


def main():
    # ---- Parse args ----
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    ap.add_argument('--weights', default=None, help='Path to checkpoint (.ckpt)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] Using: {device}")

    # ---- Model setup ----
    spec = get_loss_spec(cfg['loss']['name'])
    Cout = spec.required_out_classes
    mcfg = cfg['model']
    model = UNet2D(
        in_channels=mcfg['in_channels'],
        out_classes=Cout,
        base=mcfg.get('base_channels', 64)
    ).to(device)

    # ---- Data ----
    test_dl = build_test_loader(cfg)

    # ---- Checkpoint path ----
    loss_name = cfg['loss']['name']
    default_ckpt = Path(cfg['out_dir']) / f"{loss_name}.ckpt"
    ckpt_path = Path(args.weights) if args.weights is not None else default_ckpt

    print(f"[eval] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    # ---- Evaluation ----
    all_d, all_i, all_p, all_r = [], [], [], []
    thr = cfg.get('threshold', 0.5)

    with torch.no_grad():
        for img, msk in test_dl:
            img, msk = img.to(device), msk.to(device)
            logits = model(img)
            probs = probs_for_metrics(logits, spec)

            # Binary vs. multiclass
            if Cout == 1:
                pred = (probs > thr).float()
                target = (msk.unsqueeze(1) == 1).float()
            else:
                pred_idx = probs.argmax(dim=1)
                pred = (pred_idx == 1).float().unsqueeze(1)
                target = (msk == 1).float().unsqueeze(1)

            d, i, p, r = compute_batch_metrics_binary(pred, target)
            all_d += list(d)
            all_i += list(i)
            all_p += list(p)
            all_r += list(r)

    # ---- Report ----
    def ci(arr):
        arr = np.array(arr, dtype=float)
        m = arr.mean() if arr.size else float('nan')
        c = 1.96 * st.sem(arr) if arr.size > 1 else 0.0
        return m, c

    print("\n=== Test metrics ===")
    for name, arr in [('Dice', all_d), ('IoU', all_i), ('Precision', all_p), ('Recall', all_r)]:
        m, c = ci(arr)
        print(f"{name:9s}: {m:.3f} ± {c:.3f}")

    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / f"test_{loss_name}.txt"

    with open(results_file, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write("=== Test metrics ===\n")
        for name, arr in [('Dice', all_d), ('IoU', all_i), ('Precision', all_p), ('Recall', all_r)]:
            m, c = ci(arr)
            line = f"{name:9s}: {m:.3f} ± {c:.3f}\n"
            f.write(line)

    print(f"\n Results saved to {results_file}")


if __name__ == "__main__":
    main()
