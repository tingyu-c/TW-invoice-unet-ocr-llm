import torch
import numpy as np
from PIL import Image
from unet_model import UNet

# ============================================================
# Settings
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

FIELDS = ["invoice_no", "date", "total_amount"]

# ============================================================
# Load UNet model
# ============================================================
def load_model(checkpoint_path: str):
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)

    model.eval()
    return model


# ============================================================
# Preprocess image → Tensor
# ============================================================
def preprocess(pil_img: Image.Image):
    """
    Input : PIL.Image (RGB)
    Output: Tensor [1, 3, 512, 512]
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0  # (H,W,3)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {arr.shape}")

    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    return tensor


# ============================================================
# UNet inference (SAFE VERSION)
# ============================================================
def run_unet(pil_img: Image.Image, checkpoint_path: str):
    """
    最終正確版 UNet inference:
    - 正確對齊原圖和 mask
    - 保證 crop 不會是空白
    - 保證 OCR 拿到的裁剪是真實 ROI
    """

    model = load_model(checkpoint_path)
    model.eval()

    # ========= Preprocess =========
    ow, oh = pil_img.size
    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    x = preprocess(img_resized)  # → (1,3,H,W)

    with torch.no_grad():
        out = model(x)

        if isinstance(out, (list, tuple)):
            out = out[0]

        prob = torch.sigmoid(out.squeeze(0)).cpu().numpy()  # (3,H,W)

    # ========= Masks =========
    masks = {
        "invoice_no":   prob[0] > 0.25,
        "date":         prob[1] > 0.40,
        "total_amount": prob[2] > 0.30,
    }

    crops = {}

    # ========= Mask → Crop（正確 mapping） =========
    for key, mask in masks.items():
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            crops[key] = None
            continue

        # mask 座標（相對於 IMG_SIZE）
        mx1, mx2 = xs.min(), xs.max()
        my1, my2 = ys.min(), ys.max()

        # ===== 正確尺度換算：mask → 原圖 =====
        # 原因：mask 是 (IMG_SIZE × IMG_SIZE)
        scale_x = ow / IMG_SIZE
        scale_y = oh / IMG_SIZE

        x1 = int(mx1 * scale_x)
        x2 = int(mx2 * scale_x)
        y1 = int(my1 * scale_y)
        y2 = int(my2 * scale_y)

        # ===== 補強 padding =====
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(ow, x2 + pad_x)
        y2 = min(oh, y2 + pad_y)

        # ===== 安全檢查（避免空裁切） =====
        if x2 <= x1 or y2 <= y1:
            crops[key] = None
            continue

        crop = pil_img.crop((x1, y1, x2, y2))

        # 二次檢查：不能是 0×0 或黑白空圖
        arr = np.array(crop)
        if arr.size == 0 or arr.mean() < 3:  # 避免全黑
            crops[key] = None
            continue

        crops[key] = crop

    return masks, crops

