from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISICDataset(Dataset):
    def __init__(self, base_dir, split='training', aug=None, image_size=(384, 384)):
        self.base_dir = Path(base_dir)
        self.split = split
        self.aug = aug
        self.H, self.W = image_size
        self.img_dir = self.base_dir / f'ISIC2018_Task1-2_{split.capitalize()}_Input'
        self.msk_dir = self.base_dir / f'ISIC2018_Task1_{split.capitalize()}_GroundTruth'
        self.imgs = sorted(list(self.img_dir.glob("*.jpg")))
        self.msks = [self.msk_dir / f"{p.stem}_segmentation.png" for p in self.imgs]
        assert all(m.exists() for m in self.msks), f"Missing masks in {split} set"
        self.resize = A.Resize(height=self.H, width=self.W)


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB")).astype(np.float32)
        msk = np.array(Image.open(self.msks[idx]).convert("L"))
        msk = (msk > 0).astype(np.uint8)
        sample = self.resize(image=img, mask=msk)
        img, msk = sample['image'], sample['mask']
        img = (img - img.mean(axis=(0, 1), keepdims=True)) / (img.std(axis=(0, 1), keepdims=True) + 1e-8)
        if self.aug:
            sample = self.aug(image=img, mask=msk)
            img, msk = sample['image'], sample['mask']
        sample = ToTensorV2()(image=img, mask=msk)
        img, msk = sample['image'], sample['mask']
        return img, msk.long()