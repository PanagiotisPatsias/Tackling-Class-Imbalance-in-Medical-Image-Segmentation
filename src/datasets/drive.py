from pathlib import Path
import numpy as np
import cv2
import tifffile
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset


class DRIVE(Dataset):
    def __init__(self, root_dir, mode='training', size=(512, 512), aug=None):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.size = size
        self.aug = aug
        self.img_paths = []
        self.mask_paths = []


        if mode == 'training':
            img_dir = self.root_dir / 'training' / 'images'
            manual_dir = self.root_dir / 'training' / '1st_manual'
            for img_path in sorted(img_dir.glob("*_training.tif")):
                number = img_path.name.split("_")[0]
                mask_path = manual_dir / f"{number}_manual1.gif"
                if mask_path.exists():
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        elif mode == 'test':
            img_dir = self.root_dir / 'test' / 'images'
            manual_dir = self.root_dir / 'test' / '1st_manual'
            for img_path in sorted(img_dir.glob("*_test.tif")):
                number = img_path.name.split("_")[0]
                mask_path = manual_dir / f"{number}_manual1.gif"
                if mask_path.exists():
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        else:
            raise ValueError("mode must be 'training' or 'test'")


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img = tifffile.imread(str(self.img_paths[idx])).astype(np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        msk = imageio.imread(str(self.mask_paths[idx]))
        msk = (msk > 0).astype(np.uint8)
        if self.aug:
            sample = self.aug(image=img, mask=msk)
            img, msk = sample["image"], sample["mask"]
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, torch.from_numpy(msk).long()