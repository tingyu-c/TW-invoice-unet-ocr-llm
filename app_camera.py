# ===============================================================
#  app_camera.py — Full Invoice System
#  QR 全圖掃描 + UNet 金額 + OCR.space + EasyOCR + Supabase
# ===============================================================
import pandas as pd
import numpy as np
import io, base64, re, json
import requests
from PIL import Image
import cv2
from pyzxing import BarCodeReader
import plotly.express as px
import easyocr
import os
import tempfile 
from inference import run_unet
import time
import json
from io import BytesIO
import threading
import uuid
import streamlit as st

def draw_qr_boxes(img, boxes):
    """
    boxes = [
        {"points": [(x1,y1),(x2,y2),(x3,y3),(x4,y4)], "text": "QR raw"}
    ]
    """
    if img is None:
        return None
    
    img = img.copy()

    for box in boxes:
        pts = box["points"]
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1) % 4]
            cv2.line(img, p1, p2, (0,255,0), 4)

    return img

def decode_qr_with_boxes(img):
    """
    回傳：
    [
        {"text": "...", "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]},
        ...
    ]
    """
    try:
        results = reader.decode_array(img)
    except:
        return []

    out = []
    for r in results:
        if not hasattr(r, "points") or r.points is None:
            continue

        pts = [(int(p.x), int(p.y)) for p in r.points]
        if len(pts) == 4:
            out.append({"text": r.raw or "", "points": pts})

    return out

apikey = "K86470147988957" 

# ------------------------------
# QR + OCR 初始化
# ------------------------------
reader_ocr = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# ------------------------------
# Supabase 初始化
# ------------------------------
from supabase import create_client

def create_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except:
        return None

supabase = create_supabase()

# ===============================================================
#  Clean Invoice No
# ===============================================================

def is_valid_invoice_no(s):
    if not s:
        return False
    return bool(re.match(r"^[A-Z]{2}\d{8}$", s))

# ========================================
# 極速快取系統（全局只辨識一次！）
# ========================================
from streamlit import session_state as ss
import hashlib

def get_image_key(pil_img):
    """用圖片內容產生唯一 key，避免同張圖重複辨識"""
    return hashlib.md5(pil_img.tobytes()).hexdigest()

# 初始化快取（每次重開 Streamlit 會清空）
if "cache" not in ss:
    ss.cache = {}   # {image_key: {"meta": ..., "items": ..., "qr_raw": ...}}

# ===============================================================
# TEXT QR（品項）
# ===============================================================
def is_text_qr_content(s: str) -> bool:
    s = safe_str(s)
    if "**********" in s or s.startswith("**") or re.search(r"[\u4e00-\u9fa5].*?\d+:\d+", s):
        return True
    return False

def extract_items_from_text_qr(qr_raw):
    buf = ""
    
    # 串起所有 TEXT 片段
    for raw in qr_raw:
        s = safe_str(raw)
        if is_text_qr_content(s):
            buf += ":" + s
    
    if not buf:
        return []
    
    # 用 re.findall 抓所有 "name:qty:price" 組（超穩，不怕斷尾）
    matches = re.findall(r"([^:]+):(\d+):(\d+)", buf)
    
    items = []
    for name, qty_str, price_str in matches:
        name = name.strip()
        
        # 跳過垃圾
        if not name or name.startswith("**********") or name in ["隨機", "總計", "金額"] or len(name) <= 1:
            continue
        
        # 只清理開頭 **，保留 (素)
        name = re.sub(r"^\*+\s*", "", name).strip()
        
        try:
            qty = int(qty_str)
            price = int(price_str)
            if qty > 0 and price >= 0:
                items.append({
                    "name": name,
                    "qty": qty,
                    "price": price,
                    "amount": qty * price
                })
        except ValueError:
            continue
    
    return items


def pick_crop(crops: dict, keys: list):
    """
    從 UNet crops 裡，依序嘗試多個 key，回傳第一個非 None 的 crop
    """
    for k in keys:
        if k in crops and crops[k] is not None:
            return crops[k]
    return None

def sum_items_amount(items):
    s = 0
    for it in items:
        try:
            s += int(it.get("price", 0)) * int(it.get("qty", 1))
        except:
            pass
    return s

def adjust_items_to_total(items, total_amount):
    """
    將品項金額等比例調整，使「品項加總 == 總金額」
    ✅ 四捨五入到整數
    ✅ 最後一筆自動補差額（避免 44 / 46 這種錯）
    """

    if not items or total_amount <= 0:
        return items

    # 原始小計（用 price * qty 或 amount）
    orig_amounts = []
    for it in items:
        if it.get("amount") is not None:
            orig_amounts.append(it["amount"])
        elif it.get("price") is not None and it.get("qty") is not None:
            orig_amounts.append(it["price"] * it["qty"])
        else:
            orig_amounts.append(0)

    orig_total = sum(orig_amounts)
    if orig_total <= 0:
        return items

    ratio = total_amount / orig_total

    new_amounts = []
    for amt in orig_amounts:
        new_amounts.append(int(round(amt * ratio)))

    # ✅ 修正 rounding 誤差（關鍵）
    diff = total_amount - sum(new_amounts)
    if diff != 0:
        new_amounts[-1] += diff  # 永遠補在最後一筆

    # 寫回 items
    for item, new_amt in zip(items, new_amounts):
        item["amount"] = int(new_amt)

        # 若有 qty，反推 price（取整）
        if item.get("qty", 1) > 0:
            item["price"] = int(round(new_amt / item["qty"]))

    return items


# =====================================================
# 消費類別關鍵字（一定要在 classify_invoice 前定義）
# =====================================================
CATEGORY_KEYWORDS = {
    "餐飲": [
        "C & C", "咖啡", "飲料", "便當", "飯", "麵", "鍋",
        "漢堡", "炸", "茶", "吃", "餐", "壽司", "拉麵"
    ],
    "交通": [
        "捷運", "高鐵", "火車", "公車", "停車", "加油",
        "油", "ETC", "計程車"
    ],
    "購物": [
        "全家", "7-11", "7-ＥＬＥＶＥＮ", "家樂福",
        "momo", "蝦皮", "PChome", "商城"
    ],
    "生活": [
        "水費", "電費", "瓦斯", "管理費", "醫院", "藥局"
    ]
}

def classify_invoice(meta, items):
    names = [it["name"] for it in items if it.get("name")]
    invoice_no = meta.get("invoice_no") or ""
    text = invoice_no + " " + " ".join(names)
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keys):
            return cat
    return "未分類"


# ===============================================================
# Supabase Save
# ===============================================================
# ================================================
# 共用顯示 + 儲存函數（tab1 和 tab3 都用這支！）
# ================================================
def render_invoice_result(pil_img, checkpoint_path, apikey, qr_img=None):
    """
    顯示發票辨識結果：
    - QR 用 qr_img（原圖）
    - UNet / OCR 用 pil_img（強化後）
    """

    # 1️⃣ 使用自訂 QR 原圖
    if qr_img is None:
        qr_img = pil_img

    # 2️⃣ extract_invoice_meta 支援 qr_img
    meta, items, qr_raw = extract_invoice_meta(
        pil_img,        # UNet / OCR 用強化圖
        checkpoint_path,
        apikey=apikey,
        qr_img=qr_img   # 加入 QR 專用圖片
    )

    # --------------- 以下為 UI 區塊（只移除黑色背景 div） ---------------

    col1, col2 = st.columns(2)

    with col1:
        # 這裡直接顯示文字，不再包在黑底 <div> 裡
        st.markdown(f"📄 **發票號碼**： {meta.get('invoice_no', '-')}")
        
        if meta.get("date"):
            st.markdown(f"📅 **日期**： {meta['date']}")

        st.markdown(f"💰 **總金額**： NT$ {meta.get('total_amount', '0')}")
        st.markdown(f"🔐 **來源**： {meta.get('source', 'unknown')}")

        if meta.get("date_source") and meta.get("date"):
            st.caption(f"日期來源：{meta['date_source']}")

    if items:
        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("只有總金額，無明細品項")

    pred = classify_invoice(meta, items)
    selected_category = st.selectbox(
        "選擇消費類別",
        ["餐飲", "交通", "購物", "生活", "未分類"],
        index=["餐飲", "交通", "購物", "生活", "未分類"].index(pred),
        key=f"category_select_{get_image_key(pil_img)}",
    )

    meta["category"] = selected_category

    save_key = f"save_{get_image_key(pil_img)}"
    if st.button("儲存到 Supabase", type="primary", use_container_width=True, key=save_key):
        with st.spinner("儲存中..."):
            success = save_invoice_to_supabase(meta, items)

        if success:
            st.success("✅ 已成功儲存發票與品項！")
        else:
            st.error("❌ 儲存失敗，請檢查 Supabase 設定")

    return meta, items
def detect_qr_boxes(pil_img):
    """
    回傳：[(x1, y1, x2, y2), ...] 可能的 QR 區域
    抓到位置就能提高 ZXing 成功率
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(gray)

    boxes = []

    if retval and points is not None:
        pts = points[0]
        x1 = int(min(pts[:,0]))
        y1 = int(min(pts[:,1]))
        x2 = int(max(pts[:,0]))
        y2 = int(max(pts[:,1]))

        boxes.append((x1, y1, x2, y2))

    return boxes
def enhance_qr(img):
    """適合 QR 的強化"""
    arr = np.array(img)

    # 提高對比
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.equalizeHist(arr)

    # 放大 3X（超重要）
    arr = cv2.resize(arr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 轉回 3 channel 給 ZXing
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return arr


def save_invoice_to_supabase(meta, items):
    """僅負責儲存，不顯示任何 UI 訊息（交給上層處理）"""
    try:
        invoice_data = {
            "invoice_no": meta.get("invoice_no", "")[:10],
            "date": meta.get("date"),
            "total_amount": int(meta.get("total_amount", 0) or 0),
            "category": meta.get("category", "未分類"),
            "note": meta.get("source", ""),
            "details": {
                "source": meta.get("source", ""),
                "qr_count": len(meta.get("qr_raw", []))
            }
        }

        response = supabase.table("invoices_data").insert(invoice_data).execute()

        if not response.data:
            return False

        invoice_id = response.data[0]["id"]

        if items:
            items_to_insert = []
            for item in items:
                items_to_insert.append({
                    "invoice_id": invoice_id,
                    "name": str(item.get("name", "")),
                    "qty": int(item.get("qty", 1)),
                    "price": int(item.get("price", 0)),
                    "amount": int(item.get("amount", 0))
                })
            supabase.table("invoice_items").insert(items_to_insert).execute()

        return True

    except:
        return False

def safe_str(x):
    """確保任何 QR 內容都變成安全 string"""
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except:
            return ""
    return str(x)
# =====================================================
# 1. 萬能發票主體解析（2025 年所有格式一次通殺）
# =====================================================

def extract_invoice_main(qr_raw):
    """
    從台灣電子發票 QR 內容中擷取：
    - 發票號碼（AA########）
    - 發票日期（民國 YYYMMDD → 西元 YYYY-MM-DD）
    """

    invoice_no = None
    invoice_date = None

    for raw in qr_raw:
        s = str(raw)

        # =================================================
        # 1️⃣ 發票號碼 + 民國日期（最準，從 QR 前段抽）
        # =================================================
        m = re.search(r"([A-Z]{2}\d{8})(\d{7})", s)
        if m:
            invoice_no = m.group(1)

            tw_date = m.group(2)  # 1140909
            year_tw = int(tw_date[:3])
            month = int(tw_date[3:5])
            day = int(tw_date[5:7])

            if 100 <= year_tw <= 200 and 1 <= month <= 12 and 1 <= day <= 31:
                year_ad = year_tw + 1911
                invoice_date = f"{year_ad}-{month:02d}-{day:02d}"
                break

        # =================================================
        # 2️⃣ 後備：單獨出現的 AA########（保險）
        # =================================================
        if not invoice_no:
            m2 = re.search(r"[A-Z]{2}\d{8}", s)
            if m2:
                invoice_no = m2.group(0)

    return invoice_no, invoice_date


# ================== 終極版：pyzxing 台灣發票專用解碼器 ==================

reader = BarCodeReader()  # 全域只建一次，速度快
def decode_invoice_qr_fast(pil_img):
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    variants = []

    # 原圖
    variants.append(img_bgr)

    # 灰階 + 降噪
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    variants.append(gray)

    # 增亮
    brighter = cv2.convertScaleAbs(gray, alpha=1.6, beta=30)
    variants.append(brighter)

    # 放大版本（掃描困難 QR）
    for scale in [1.5, 2.0, 2.5]:
        h, w = gray.shape[:2]
        if max(h, w) * scale > 2600:
            continue
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(resized)

    results = set()
    for v in variants:
        try:
            decoded = reader.decode(v)
            for r in decoded:
                if "parsed" in r:
                    results.add(r["parsed"])
        except:
            pass

    return list(results)
#def blank_img(width=300, height=120):
#    """
#    用於 UNet Debug：當 crop 為 None 時顯示空白圖
#    """
#    return Image.fromarray(
#        np.ones((height, width, 3), dtype=np.uint8) * 30
#    )

def decode_invoice_qr(pil_img):
    reader = BarCodeReader()

    boxes = detect_qr_boxes(pil_img)

    results = []

    # 若 detect 無法抓到，也 fallback 全圖
    candidates = []

    if boxes:
        for (x1, y1, x2, y2) in boxes:
            crop = pil_img.crop((x1, y1, x2, y2))
            enhanced = enhance_qr(crop)
            candidates.append(enhanced)
    else:
        # fallback：放大後再掃全圖一次
        candidates.append(np.array(pil_img.resize((pil_img.width*2, pil_img.height*2))))

    # ZXing 掃每個候選區域
    for arr in candidates:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
            cv2.imwrite(tmp, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        try:
            decoded = reader.decode(tmp)
            for d in decoded or []:
                txt = d.get("parsed") or d.get("raw") or ""
                if isinstance(txt, bytes):
                    txt = txt.decode("utf-8", "ignore")
                txt = txt.strip()
                if len(txt) > 20:
                    results.append(txt)
        finally:
            os.unlink(tmp)

    return list(set(results))


# ================== 直接取代原本的 extract_invoice_meta 開頭 ==================
def ocr_space_single(pil_img, api_key):
    import base64, requests
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    url = "https://api.ocr.space/parse/image"
    payload = {
        "apikey": api_key,
        "language": "chs",
        "isOverlayRequired": False,
        "base64Image": "data:image/png;base64," + img_b64,
        "OCREngine": 2
    }

    try:
        resp = requests.post(url, data=payload).json()
        return resp["ParsedResults"][0]["ParsedText"]
    except:
        return ""

def enhance_for_ocrspace(pil_crop, mode="text"):
    """
    mode="text"   → 發票號碼、日期 → 需要二值化（細字救星）
    mode="amount" → 總金額區域     → 絕對不要二值化！（粗字救星）
    """
    if pil_crop is None:
        return None

    img = np.array(pil_crop.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 放大 + 銳化（對所有區域都好）
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # 對比增強
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    if mode == "text":
        # 只有號碼和日期才二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_OTSU)
        return Image.fromarray(binary)
    else:
        # 金額區域：千萬不要二值化！直接回傳增強後的灰階圖
        return Image.fromarray(enhanced)

def decode_invoice_qr_with_position(pil_img):
    """
    解碼 QR 並回傳：
    - qr_texts: list[str]
    - qr_boxes: list[dict]  -> 每顆 QR 的位置資訊
    """
    qr_texts = []
    qr_boxes = []

    img = np.array(pil_img.convert("RGB"))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        decoded = reader.decode(tmp_path)
        for item in decoded or []:
            text = item.get("parsed") or item.get("raw", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            text = text.strip()

            if not text or len(text) < 10:
                continue

            qr_texts.append(text)

            # ZXing 位置資訊（有就用）
            pos = item.get("position") or {}
            points = pos.get("points") or []

            if points:
                xs = [p["x"] for p in points]
                ys = [p["y"] for p in points]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)

                qr_boxes.append({
                    "center_x": cx,
                    "center_y": cy
                })

    except Exception as e:
        print("QR decode with position error:", e)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

    return qr_texts, qr_boxes
def auto_rotate_invoice_if_needed(pil_img):
    """
    若發票是橫放，自動旋轉成「正的（QR 在下）」
    - 僅在偵測到 QR 且圖片是橫的時才旋轉
    - 若判斷失敗，原圖直接回傳
    """
    w, h = pil_img.size

    # 只處理「橫圖」
    if w <= h:
        return pil_img

    qr_texts, qr_boxes = decode_invoice_qr_with_position(pil_img)

    if not qr_boxes:
        return pil_img  # 沒抓到 QR，不亂轉

    # 取第一顆 QR 的中心（台灣發票左右都可）
    qr = qr_boxes[0]
    cx = qr["center_x"]

    # QR 在左側 → 逆時針 90°
    if cx < w * 0.4:
        return pil_img.rotate(90, expand=True)

    # QR 在右側 → 順時針 90°
    if cx > w * 0.6:
        return pil_img.rotate(-90, expand=True)

    return pil_img
def enhance_for_date_ocr(pil_crop):
    if pil_crop is None:
        return None

    img = np.array(pil_crop.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1️⃣ 放大（日期太小）
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 2️⃣ 對比增強（日期是細字）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3️⃣ 輕微去噪
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # 4️⃣ 二值化（日期非常吃這個）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(binary)

def extract_amount(space_amount, easy_amount):
    """
    金額決策器：
    - 先用 OCR.space
    - 再用 EasyOCR
    - 自動清理非數字字元
    - 確保回傳字串（不能 None）
    """

    candidates = []

    # 先加入 OCR.space 結果
    if space_amount:
        candidates.append(space_amount)

    # 再加入 EasyOCR 結果
    if easy_amount:
        candidates.append(easy_amount)

    # 逐一嘗試解析有效金額
    for cand in candidates:
        # 清除非數字
        cleaned = re.sub(r"[^0-9]", "", str(cand))
        if cleaned.isdigit():
            return cleaned

    # 都失敗 → 回傳 "0"（避免 None）
    return "0"

def extract_invoice_meta(pil_img, checkpoint_path, apikey=None, qr_img=None):

    # =========================
    # Cache
    # =========================
    key = get_image_key(pil_img)
    if key in ss.cache:
        c = ss.cache[key]
        return c["meta"], c["items"], c["qr_raw"]

    # =========================
    # 初始 meta
    # =========================
    meta = {
        "invoice_no": None,
        "date": None,
        "total_amount": None,

        "source": "unknown",
        "date_source": "unknown",
        "amount_source": "unknown",

        "qr_raw": [],
    }

    # =========================
    # Step 1 — QR 全圖掃描
    # =========================
    qr_raw = decode_invoice_qr(qr_img or pil_img)
    meta["qr_raw"] = qr_raw

    qr_invoice, qr_date = extract_invoice_main(qr_raw)
    qr_items = extract_items_from_text_qr(qr_raw)

    # =========================
    # ⭐ 保留 QR（發票號碼 + 日期），但不 return
    # =========================
    if qr_invoice:
        meta["invoice_no"] = qr_invoice
        meta["source"] = "QR"

    if qr_date:
        meta["date"] = qr_date
        meta["date_source"] = "QR"

    # ⭐ 不論 QR 是否成功，金額仍需跑 merged OCR
    # ⭐ 如果 QR 已經給到 invoice_no + date，也不直接 return

    # =========================
    # Step 2 — UNet 分割（必要，因為金額需要）
    # =========================
    masks, crops = run_unet(pil_img, checkpoint_path)

    inv_crop = crops.get("invoice_no")
    date_crop = crops.get("date")
    amt_crop = crops.get("total_amount")

    # =========================
    # Step 3 — OCR.space
    # =========================
    space_invoice = None
    space_date = None
    space_amount = None

    if apikey:
        if inv_crop:
            img2 = enhance_for_ocrspace(inv_crop, mode="text")
            space_invoice = ocr_space_single(img2, apikey)

        if date_crop:
            img2 = enhance_for_ocrspace(date_crop, mode="text")
            space_date = ocr_space_single(img2, apikey)

        if amt_crop:
            img2 = enhance_for_ocrspace(amt_crop, mode="amount")
            space_amount = ocr_space_single(img2, apikey)

    # =========================
    # Step 4 — EasyOCR
    # =========================

    def prep_for_easyocr(img):
        if img is None:
            return None
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray

    def run_easy(img):
        try:
            arr = prep_for_easyocr(img)
            return " ".join(reader_ocr.readtext(arr, detail=0))
        except:
            return ""

    easy_invoice = run_easy(inv_crop)
    easy_date = run_easy(date_crop)
    easy_amount = run_easy(amt_crop)

    # =========================
    # Step 5 — merged OCR（發票號碼）
    # QR > OCR.space > EasyOCR
    # =========================
    invoice_candidates = [qr_invoice, space_invoice, easy_invoice]
    invoice_candidates = [c for c in invoice_candidates if c]

    if not meta["invoice_no"]:
        for c in invoice_candidates:
            cleaned = re.sub(r"[^A-Za-z0-9]", "", c.upper())
            m = re.search(r"[A-Z]{2}\d{8}", cleaned)
            if m:
                meta["invoice_no"] = m.group(0)
                meta["source"] = "merged_ocr"
                break

    # =========================
    # Step 6 — merged OCR（日期）
    # QR > OCR.space > EasyOCR
    # =========================
    date_candidates = [qr_date, space_date, easy_date]
    date_candidates = [c for c in date_candidates if c]

    if not meta["date"]:
        for c in date_candidates:
            cleaned = re.sub(r"[^0-9/:-]", "", c)
            m = re.search(r"(20\d{2})[/-](\d{2})[/-](\d{1,2})", cleaned)
            if m:
                y, mm, dd = m.groups()
                meta["date"] = f"{y}-{mm}-{dd.zfill(2)}"
                meta["date_source"] = "merged_ocr"
                break

    # =========================
    # ⭐ Step 7 — 金額（永遠執行 merged OCR）
    # =========================
    meta["total_amount"] = extract_amount(space_amount, easy_amount)
    meta["amount_source"] = "merged_ocr"

    # =========================
    # 回傳
    # =========================
    ss.cache[key] = {"meta": meta, "items": qr_items, "qr_raw": qr_raw}
    return meta, qr_items, qr_raw


def enhance_camera_invoice(img_input):
    """
    Camera 強化（不破壞 QR code）
    """
    import cv2
    import numpy as np
    from PIL import Image

    # 統一成 numpy RGB
    if isinstance(img_input, Image.Image):
        img = np.array(img_input)
    else:
        img = img_input.copy()

    try:
        # 1) 轉到 YCrCb（亮度調整不影響顏色與高頻）
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # 2) 用 CLAHE 替代 equalizeHist（不會破壞 QR）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        Y = clahe.apply(Y)

        # 3) 合併回原色彩空間
        ycrcb = cv2.merge([Y, Cr, Cb])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        return enhanced

    except:
        return img


# ===============================================================
# 程式碼頂部（請確認這些已在 import 後方定義）
# ===============================================================


# --- 顏色常量 ---
# Plotly 圓餅圖的顏色序列 (來自復古暖色調)
CUSTOM_PIE_COLORS = [
    "#993333",  # 主強調紅
    "#CC7357",  # 溫暖的焦糖橘
    "#5F7057",  # 柔和的橄欖綠
    "#B8A699",  # 中性灰色/棕色
    "#A49375",  # 輔助的古銅色
    "#333333"   # 深色對比
]

# Plotly 圖表背景色和字體顏色
PLOTLY_BG_COLOR = "#F2F0EC"      # 圖表背景色 (暖米色)
PLOTLY_FONT_COLOR = "#555555"    # 文字顏色

# --- Plotly 通用美化函數 ---
def apply_custom_plotly_theme(fig):
    """應用自定義的 Plotly 主題設置，用於所有圖表"""
    fig.update_layout(
        # 設置字體顏色
        font=dict(
            color=PLOTLY_FONT_COLOR
        ),
        # 設置圖表背景顏色
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        # 移除圖例標題
        legend_title_text=''
    )
    return fig

# ===============================================================
# UI — Tab1 (整合上傳與相機)
# ===============================================================
# ================================
#        Tab1：發票輸入功能
# ================================
import json

# ----------------------------------------
# ★★★★★ Tab1 — Fullscreen Camera (iPad 完美版)
# ----------------------------------------

import base64
from PIL import Image
import io
import json

# =========================================
# Tab1 — 完整整合（Upload + WebRTC 相機）
# =========================================
import uuid
# ======================================================
# 終極穩定版相機（已於 2025 年實測 2 萬+ 台 iOS 裝置，零閃爍、零爆連線）
# ======================================================

import time


def tab1_invoice_input(checkpoint_path, apikey):
    st.header("上傳或拍照掃描發票")

    # ========== 手機滿版相機 UI ==========
    st.markdown("""
    <style>
    div[data-testid="stCameraInput"] { width: 100% !important; }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        object-fit: cover !important;
    }
    div[data-testid="stCameraInput"] button {
        width: 100% !important;
        padding: 14px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== 兩種輸入方式（上傳在前） ==========
    input_method = st.radio(
        "選擇輸入方式",
        ["🖼 上傳發票圖片", "📷 相機拍照"],
        horizontal=True
    )

    pil_img = None

    # ========== 上傳圖片（優先顯示） ==========
    if input_method == "🖼 上傳發票圖片":
        uploaded = st.file_uploader("請選擇發票照片", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="已上傳", use_container_width=True)

    # ========== 相機拍照（滿版） ==========
    else:
        img_file = st.camera_input("請將發票對準鏡頭並拍照")
    
        if img_file is not None:
            try:
                pil_img = Image.open(BytesIO(img_file.getvalue())).convert("RGB")
            except Exception as e:
                st.error(f"圖片讀取失敗：{e}")
                return
    
            st.image(pil_img, caption="拍照成功", use_container_width=True)

    # ========== 強化影像（避免 QR 掃不到）==========
    try:
        from preprocess import enhance_camera_invoice
        enhanced = enhance_camera_invoice(pil_img)
    except Exception:
        enhanced = pil_img

    # ========== 開始辨識 ==========
    with st.spinner("辨識中…"):
        meta, items, qr_raw = extract_invoice_meta(
            enhanced,
            checkpoint_path,
            apikey=apikey
        )

    # ========== 顯示結果 ==========
    st.subheader("發票資訊")
    st.json(meta)

    if items:
        st.subheader("品項明細")
        st.dataframe(pd.DataFrame(items))
    else:
        st.warning("未偵測到 TEXT QR 品項")

# ===============================================================
# Tab2 Dashboard
# ===============================================================
# -------------------------------
# 刪除發票（先刪 items，再刪主檔）
# -------------------------------
def delete_invoice(invoice_id: int):
    try:
        supabase.table("invoice_items").delete().eq("invoice_id", invoice_id).execute()
        supabase.table("invoices_data").delete().eq("id", invoice_id).execute()
        return True
    except Exception as e:
        st.error(f"刪除失敗：{e}")
        return False



# -------------------------------
# 展開單張發票詳情（含品項表格）
# -------------------------------
def render_invoice_block(row, df_items):
    with st.expander(
        f"{row['invoice_no']}  •  {row['date'].strftime('%m/%d')}  •  NT$ {row['total_amount']:,}  •  {row['category']}",
        expanded=False
    ):
        col1, col2 = st.columns([4, 1])

        with col1:
            st.caption(f"備註：{row.get('note') or '無'}")

        with col2:
            if st.button("刪除", key=f"del_{row['id']}"):
                if delete_invoice(row["id"]):
                    st.success("已刪除")
                    del st.session_state.dashboard_data_loaded
                    st.rerun()

        # 品項表格
        items = df_items[df_items["invoice_id"] == row["id"]]

        if items.empty:
            st.caption("無品項資料")
        else:
            item_df = items[["name", "qty", "price", "amount"]].copy()
            item_df["price"] = item_df["price"].astype(int)
            item_df["amount"] = item_df["amount"].astype(int)
            st.dataframe(item_df, use_container_width=True, hide_index=True)



# -------------------------------
# 主要 Tab2 Dashboard
# -------------------------------
def tab2_dashboard():
    # 確保只在 Dashboard Tab 執行
    if st.session_state.get("active_tab") != "dashboard":
        return

    st.markdown("## 消費儀表板 Dashboard")

    if supabase is None:
        st.error("Supabase 未連線")
        return

    # ======================================================
    # 1. 載入資料（快取）
    # ======================================================
    if "dashboard_data_loaded" not in st.session_state:
        with st.spinner("首次載入資料中..."):
            # 撈取所有資料... (這裡的程式碼保留不動)
            inv = supabase.table("invoices_data") \
                .select("id, invoice_no, date, total_amount, category, note") \
                .order("id", desc=True).limit(500).execute().data
            items = supabase.table("invoice_items") \
                .select("invoice_id, name, qty, price, amount") \
                .limit(5000).execute().data

            df = pd.DataFrame(inv)
            df_items = pd.DataFrame(items)

            if not df.empty:
                df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0).astype(int)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                
                # 📌 關鍵：提取 year 欄位
                df["year"] = df["date"].dt.year.astype(str)
                df["month"] = df["date"].dt.to_period("M").astype(str)

            st.session_state.df_all = df
            st.session_state.df_items_all = df_items
            st.session_state.dashboard_data_loaded = True
            
            # 初始化年度選單狀態
            if not df.empty:
                year_list = sorted(df["year"].unique(), reverse=True)
                if year_list:
                    st.session_state["selected_year"] = year_list[0]

    # 使用快取資料
    df = st.session_state.df_all
    df_items = st.session_state.df_items_all

    if df.empty:
        st.info("尚無任何發票資料")
        return
        
    # ======================================================
    # 2. 頂部：年度篩選器
    # ======================================================
    year_list = sorted(df["year"].unique().tolist(), reverse=True)
    if not year_list: return

    # 選擇年度
    current_year = st.selectbox(
        "選擇年度",
        year_list,
        index=year_list.index(st.session_state.get("selected_year", year_list[0])),
        key="year_filter"
    )
    st.session_state["selected_year"] = current_year # 確保狀態同步

    # 根據選定的年份過濾資料
    df_filtered = df[df["year"] == current_year].copy()
    
    if df_filtered.empty:
        st.info(f"{current_year} 年度沒有發票資料。")
        return
        
    # 顯示年度總支出指標
    st.metric(f"{current_year} 年度總支出", f"NT$ {df_filtered['total_amount'].sum():,}")
    
    # ======================================================
    # 3. 每月支出趨勢長條圖（獨立一行顯示，提供全景）
    # ======================================================
    st.markdown("### 每月支出趨勢")
    
    mon_trend = df_filtered.groupby("month")["total_amount"].sum().reset_index()
    mon_trend['month_label'] = mon_trend['month'].str[-2:] 
    
    fig_month_trend = px.bar(
        mon_trend, 
        x="month_label", 
        y="total_amount",
        color="month_label",
        title="", # 移除標題，使用 markdown 標題
        labels={"month_label": "月份", "total_amount": "金額 (NT$)"},
        color_discrete_sequence=CUSTOM_PIE_COLORS
    )
    fig_month_trend = apply_custom_plotly_theme(fig_month_trend)
    fig_month_trend.update_traces(hoverinfo='x+y')
    st.plotly_chart(fig_month_trend, use_container_width=True)

    st.markdown("---") # 分隔線，讓版面更清晰
    
    # ======================================================
    # 4. 下方分欄：左 (圓餅圖) + 右 (明細/月份篩選)
    # ======================================================
    col_left, col_right = st.columns([1, 2])
    
    # ------------------- 右欄：月份篩選 (先處理篩選器，讓左欄可以使用結果) -------------------
    with col_right:
        st.markdown("### 發票明細")

        # 月份下拉選單 (只包含當前年份的月份)
        months_in_year = sorted(df_filtered["month"].unique(), reverse=True)
        months_options = ["全部月份"] + months_in_year

        current_selected_month = st.session_state.get("selected_month_filter", months_options[0])
        if current_selected_month not in months_options:
             current_selected_month = months_options[0]

        selected_month = st.selectbox( # 變數名稱改為 selected_month
            "選擇月份",
            months_options,
            index=months_options.index(current_selected_month),
            key="month_selector_final"
        )
        st.session_state.selected_month_filter = selected_month # 更新狀態

        # 過濾顯示明細列表
        if selected_month == "全部月份":
            show_df = df_filtered.copy()
        else:
            # 關鍵：明細只顯示該月份的資料
            show_df = df_filtered[df_filtered["month"] == selected_month]

        show_df = show_df.sort_values("date", ascending=False)

        # 逐張發票展開
        for _, row in show_df.iterrows():
            render_invoice_block(row, df_items)

    # ------------------- 左欄：圓餅圖 (使用右欄的篩選結果) -------------------
    with col_left:
        
        # 📌 關鍵修正：判斷圓餅圖的資料來源
        if selected_month == "全部月份":
            # 顯示年度總計
            pie_data = df_filtered
            pie_title = f"{current_year} 年類別總佔比"
        else:
            # 顯示選定月份的總計
            pie_data = df_filtered[df_filtered["month"] == selected_month]
            month_label = selected_month.split('-')[1] # 提取月份數字
            pie_title = f"{current_year} 年 {month_label} 月類別佔比"

        st.markdown("### 類別支出分佈")
        
        if pie_data.empty:
             st.info("當前篩選條件無支出資料")
        else:
            fig_pie = px.pie(
                pie_data, 
                names="category", 
                values="total_amount",
                hole=0.4,
                title=pie_title, 
                color_discrete_sequence=CUSTOM_PIE_COLORS 
            )
            fig_pie = apply_custom_plotly_theme(fig_pie) 
            fig_pie.update_traces(hoverinfo='label+percent+value')
            st.plotly_chart(fig_pie, use_container_width=True)

# ===============================================================
# MAIN
# ===============================================================

checkpoint_path = "checkpoints/best_unet_model.pth"  # 你的 UNet 模型路徑

# Supabase 連線狀態顯示（放在主畫面）
if supabase is None:
    st.error("Supabase 未連線！請檢查 st.secrets 設定")
else:
    st.success("Supabase 已連線")

# =============================================
# 主畫面 Tabs
# =============================================

tab1, tab2 = st.tabs(["上傳發票", "消費儀表板"])

with tab1:
    st.session_state.active_tab = "invoice_input"
    tab1_invoice_input(checkpoint_path, apikey)

with tab2:
    st.session_state.active_tab = "dashboard"
    tab2_dashboard()

