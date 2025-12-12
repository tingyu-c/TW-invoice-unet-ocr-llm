# rescue_masks_from_json_final_FIXED.py
import os
import json
import cv2
import numpy as np
from glob import glob
from PIL import Image, ImageDraw


TRAIN_SIZE = (512, 512)   # 例如 (640, 640), (768, 1024) 都行


os.makedirs("fixed_images", exist_ok=True)
os.makedirs("fixed_masks", exist_ok=True)

LABEL_TO_CH = {
    "invoice_no":   0,
    "date":         1,
    "total_amount": 2,
}

def fix_mask(json_path, img_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W0 = data["imageWidth"]
    H0 = data["imageHeight"]

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    sx = w / W0
    sy = h / H0

    # 這裡改用三張獨立的 PIL 畫布，絕對不會有 contiguous 問題
    channels = {
        0: Image.new("L", (w, h), 0),
        1: Image.new("L", (w, h), 0),
        2: Image.new("L", (w, h), 0),
    }
    draws = {k: ImageDraw.Draw(v) for k, v in channels.items()}

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in LABEL_TO_CH:
            continue
        ch = LABEL_TO_CH[label]
        pts = [(p[0]*sx, p[1]*sy) for p in shape["points"]]
        draws[ch].polygon(pts, fill=255)

    # 合併成 numpy (h, w, 3)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for ch, pil_img in channels.items():
        mask[:, :, ch] = np.array(pil_img)

    # 統一一 resize
    img_resized = img.resize(TRAIN_SIZE, Image.BILINEAR)
    mask_resized = cv2.resize(mask, TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

    base = os.path.basename(img_path).rsplit(".", 1)[0]
    img_resized.save(f"fixed_images/{base}.jpg")
    np.save(f"fixed_masks/{base}.npy", mask_resized)

    print(f"成功: {base}")

# ===================== 主程式 =====================
for json_path in glob("json/*.json"):
    base = os.path.basename(json_path).replace(".json", "")
    candidates = [
        f"images/{base}.jpg",
        f"images/{base}.jpeg",
        f"images/{base}.JPG",
        f"images/{base}.png",
    ]
    img_path = None
    for p in candidates:
        if os.path.exists(p):
            img_path = p
            break
    if img_path:
        fix_mask(json_path, img_path)
    else:
        print("找不到圖片:", base)

print("\n全部完成！fixed_images + fixed_masks 準備就緒，直接拿去訓練吧！")