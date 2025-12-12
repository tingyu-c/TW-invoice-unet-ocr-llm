# json_to_mask.py —— 完全修好版（支援紅綠藍三類）

import os
import json
from PIL import Image, ImageDraw

input_dir = r"C:\Users\88697\Desktop\invoice_project\json"
output_dir = r"C:\Users\88697\Desktop\invoice_project\masks"

# 定義 Labelme label → 顏色對應表（千萬不要改錯！）
LABEL_TO_COLOR = {
    "invoice_no":   (255, 0,   0),    # 紅色
    "date":         (0,   255, 0),    # 綠色
    "total_amount": (0,   0, 255),    # 藍色
}

def json_to_mask(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])

    # 改成 RGB 模式（不是 L！）
    mask = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        label = shape["label"]
        points = [(float(p[0]), float(p[1])) for p in shape["points"]]
        
        if label in LABEL_TO_COLOR:
            color = LABEL_TO_COLOR[label]
            draw.polygon(points, fill=color, outline=color)
        else:
            print(f"警告：未知 label → {label}")

    return mask

def main():
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, filename)
        try:
            mask_img = json_to_mask(json_path)
            base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base + ".png")
            mask_img.save(output_path)
            print("生成彩色 mask：", output_path)
        except Exception as e:
            print("失敗：", filename, e)

    print("\n全部完成！現在你的 masks 資料夾裡都是紅綠藍彩色 mask 了！")

if __name__ == "__main__":
    main()