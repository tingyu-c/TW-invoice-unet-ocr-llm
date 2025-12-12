# train_v3.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from dataset import InvoiceDataset
from unet_model import UNet


# ===========================================================
# Loss
# ===========================================================
class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        inter = (pred * target).sum(-1)
        union = pred.sum(-1) + target.sum(-1)
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        return dice.mean()


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)

        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


class InvoiceLoss(nn.Module):
    def __init__(self, dice_weight=0.85, focal_weight=0.15, focal_alpha=0.8, gamma=2.0):
        super().__init__()
        self.dice = MultiLabelDiceLoss()
        self.focal = MultiLabelFocalLoss(alpha=focal_alpha, gamma=gamma)
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        return self.dw * self.dice(pred, target) + self.fw * self.focal(pred, target)


# ===========================================================
# Visualization
# ===========================================================
def visualize(img, true_mask, pred_prob, name):
    os.makedirs("visualize", exist_ok=True)

    img_np = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(f"visualize/{name}_img.png")

    H, W = true_mask.shape[1:]

    # True mask
    t = true_mask.cpu().numpy()
    true_color = np.zeros((H, W, 3), dtype=np.uint8)
    true_color[t[0] > 0.5] = [255,0,0]
    true_color[t[1] > 0.5] = [0,255,0]
    true_color[t[2] > 0.5] = [0,0,255]
    Image.fromarray(true_color).save(f"visualize/{name}_true.png")

    # Predicted mask
    p = (pred_prob.cpu().numpy() > 0.3)
    pred_color = np.zeros((H, W, 3), dtype=np.uint8)
    pred_color[p[0]] = [255,0,0]
    pred_color[p[1]] = [0,255,0]
    pred_color[p[2]] = [0,0,255]
    Image.fromarray(pred_color).save(f"visualize/{name}_pred.png")


# ===========================================================
# Training Main
# ===========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dataset
    dataset = InvoiceDataset("fixed_images", "fixed_masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Total training samples:", len(dataset))

    # Model
    model = UNet(n_channels=3, n_classes=3).to(device)

    # ★ 加上你要求的「最後一層 bias = -4 初始化」★
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 3:
            nn.init.constant_(m.bias, -4.0)
            print("已初始化 UNet 最終層 bias = -4.0（預設強烈偏向背景）")

    # Loss + Optimizer + Scheduler
    criterion = InvoiceLoss(
        dice_weight=0.85,
        focal_weight=0.15,
        focal_alpha=0.80,
        gamma=2.0
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = 999

    # Training Loop
    for epoch in range(1, 51):
        model.train()
        total_loss = 0

        for i, (img, mask) in enumerate(loader):
            img  = img.to(device)
            mask = mask.to(device)

            logits = model(img)
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 可視化
            if i == 0:
                pred_prob = torch.sigmoid(logits[0].detach())
                visualize(img[0], mask[0], pred_prob, f"epoch{epoch:03d}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss = {avg:.6f}")

        scheduler.step()

        # Save best model
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "checkpoints/best_unet.pth")
            print(">>> Best model updated!")

    print("Training Finished!")


if __name__ == "__main__":
    main()
