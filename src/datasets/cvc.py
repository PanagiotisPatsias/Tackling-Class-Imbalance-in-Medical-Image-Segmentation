from pathlib import Path
import numpy as np
import cv2
import tifffile
import torch
from torch.utils.data import Dataset


class CVC(Dataset):
    def __init__(self, indices, img_paths, msk_paths, target_size=(288, 384), aug=None):
        self.idx = indices
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.aug = aug
        self.H, self.W = target_size


    def __len__(self):
        return len(self.idx)


    def __getitem__(self, i):
        j = self.idx[i]
        img = tifffile.imread(str(self.img_paths[j])).astype(np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        msk = cv2.imread(str(self.msk_paths[j]), cv2.IMREAD_GRAYSCALE)
        msk = (msk > 0).astype(np.uint8)
        if self.aug:
            sample = self.aug(image=img, mask=msk)
            img, msk = sample["image"], sample["mask"]
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, torch.from_numpy(msk).long()