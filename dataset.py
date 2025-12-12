import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class InvoiceDataset(Dataset):
    def __init__(self, img_dir="fixed_images", mask_dir="fixed_masks"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = [
            f.rsplit(".",1)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg",".png"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # 讀取影像
        img = cv2.imread(f"{self.img_dir}/{name}.jpg")
        if img is None:
            img = cv2.imread(f"{self.img_dir}/{name}.png")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # (3,H,W)

        # 讀取 mask = (H,W,3) 0/255
        mask = np.load(f"{self.mask_dir}/{name}.npy")              # np.uint8
        mask = mask.astype(np.float32) / 255.0                     # → 0/1
        mask = torch.from_numpy(mask).permute(2,0,1).float()       # (3,H,W)

        return img, mask