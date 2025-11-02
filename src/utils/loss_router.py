from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class LossSpec:
    required_out_classes: int # 1 for binary, >=2 for multiclass
    expects: str # 'logits' | 'probs'
    activation: str | None # 'sigmoid' | 'softmax' | None
    target_format: str # 'binary_mask' | 'one_hot' | 'index'


LOSS_REGISTRY = {
    # Binary (1-channel)
    'dynamit_bce_batch': LossSpec(1, 'logits', 'sigmoid', 'binary_mask'),
    'dynamit_bce_image': LossSpec(1, 'logits', 'sigmoid', 'binary_mask'),
    'boundary_gaussian': LossSpec(1, 'probs', 'sigmoid', 'binary_mask'),
    'boundary_dou': LossSpec(1, 'probs', 'sigmoid', 'binary_mask'),


    # (softmax)
    'ce': LossSpec(2, 'logits', None, 'index'),
    'focal_loss': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'dice': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'focal_tversky': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'combo': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'tversky': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'asymmetric_unified_focal_loss': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    'symmetric_unified_focal_loss': LossSpec(2, 'probs', 'softmax', 'one_hot'),
    }


def get_loss_spec(name: str) -> LossSpec:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Add it to LOSS_REGISTRY.")
    return LOSS_REGISTRY[name]


    # ---------- helpers ----------
def apply_activation_for_loss(logits: torch.Tensor, spec: LossSpec) -> torch.Tensor:
    if spec.expects == 'logits':
        return logits
    if spec.expects == 'probs':
        if spec.activation == 'sigmoid':
            return torch.sigmoid(logits)
        if spec.activation == 'softmax':
            return torch.softmax(logits, dim=1)
        raise ValueError('Loss expects probs but activation is None/unknown.')
    raise ValueError("spec.expects must be 'logits' or 'probs'.")


def prepare_targets(msk: torch.Tensor, spec: LossSpec, num_classes: int) -> torch.Tensor:
    if spec.target_format == 'binary_mask':
        return (msk.unsqueeze(1) == 1).float()
    if spec.target_format == 'one_hot':
        return F.one_hot(msk, num_classes).permute(0,3,1,2).float()
    if spec.target_format == 'index':
        return msk.long()
    raise ValueError('Unknown target_format.')


def probs_for_metrics(logits: torch.Tensor, spec: LossSpec) -> torch.Tensor:
    if spec.activation == 'sigmoid':
        return torch.sigmoid(logits)
    if spec.activation == 'softmax':
        return torch.softmax(logits, dim=1)
    # default: infer
    C = logits.shape[1]
    return torch.softmax(logits, dim=1) if C > 1 else torch.sigmoid(logits)