# ===============================================================
#  app_camera.py — Full Invoice System
#  QR 全圖掃描 + UNet 金額 + OCR.space + EasyOCR + Supabase
# ===============================================================

import streamlit as st
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
    text = meta.get("invoice_no", "") + " " + " ".join(names)
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
def render_invoice_result(pil_img, checkpoint_path, apikey):
    """顯示發票結果（含 UNet Debug、日期條件顯示）"""

    # 1️⃣ 辨識（有快取，不會重跑）
    meta, items, qr_raw = extract_invoice_meta(pil_img, checkpoint_path, apikey)

    # =====================================================
    # 2️⃣ 顯示 UNet Debug（一定在 extract_invoice_meta 裡跑）
    # 👉 這裡不用再寫，extract_invoice_meta 已經畫完
    # =====================================================

    # =====================================================
    # 3️⃣ 顯示主要資訊（✅ 關鍵修正在這）
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                background-color:#1f1f1f;
                padding:16px;
                border-radius:10px;
                border:1px solid #333;
                font-size:18px;
                line-height:1.7;
            ">
            """,
            unsafe_allow_html=True,
        )

        # ✅ 一定顯示
        st.markdown(f"📄 **發票號碼**： {meta.get('invoice_no', '-')}")
        
        # ✅ ✅ 關鍵：只有「真的有日期」才顯示
        if meta.get("date"):
            st.markdown(f"📅 **日期**： {meta['date']}")

        # ✅ 一定顯示
        st.markdown(f"💰 **總金額**： NT$ {meta.get('total_amount', '0')}")
        st.markdown(f"🔐 **來源**： {meta.get('source', 'unknown')}")

        # （可選但強烈建議）
        if meta.get("amount_source"):
            st.caption(f"金額來源：{meta['amount_source']}")
        if meta.get("date_source") and meta.get("date"):
            st.caption(f"日期來源：{meta['date_source']}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if items:
            st.success(f"成功解析 {len(items)} 筆品項")
        else:
            st.warning("無 TEXT QR 品項")

    # =====================================================
    # 4️⃣ 顯示品項表格
    # =====================================================
    if items:
        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("只有總金額，無明細品項")

    # =====================================================
    # 5️⃣ 類別選擇
    # =====================================================
    pred = classify_invoice(meta, items)

    selected_category = st.selectbox(
        "選擇消費類別",
        ["餐飲", "交通", "購物", "生活", "未分類"],
        index=["餐飲", "交通", "購物", "生活", "未分類"].index(pred),
        key=f"category_select_{get_image_key(pil_img)}",
    )

    meta["category"] = selected_category

    # =====================================================
    # 6️⃣ 儲存
    # =====================================================
    save_key = f"save_{get_image_key(pil_img)}"

    if st.button("儲存到 Supabase", type="primary", use_container_width=True, key=save_key):
        with st.spinner("儲存中..."):
            success = save_invoice_to_supabase(meta, items)

        if success:
            st.success("✅ 已成功儲存發票與品項！")
        else:
            st.error("❌ 儲存失敗，請檢查 Supabase 設定")

    return meta, items

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

# ===============================================================
# 主邏輯：extract_invoice_meta v46
# ===============================================================
# ================== 終極版：pyzxing 台灣發票專用解碼器 ==================

reader = BarCodeReader()  # 全域只建一次，速度快
def blank_img(width=300, height=120):
    """
    用於 UNet Debug：當 crop 為 None 時顯示空白圖
    """
    return Image.fromarray(
        np.ones((height, width, 3), dtype=np.uint8) * 30
    )

def decode_invoice_qr(pil_img):
    """
    只做一件事：掃全圖，吐出所有 QR 文字（通常 2 個）
    ✅ 版本：效能優化版
      - 只保留「原圖 / 灰階 / 2x 放大」三種版本
      - 一旦抓到 >= 2 個 QR，立刻停止（台灣電子發票只需要兩顆）
    """
    results = set()

    # 確保是 RGB 3 channel，避免 cv2 出錯
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # 準備少量但高 CP 的變體
    variants = [
        img,                                   # 原圖 RGB
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰階
    ]

    # 2 倍放大（小 QR 救星），限制最大尺寸避免超大圖
    if min(h, w) * 2 <= 5000:
        big2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants.append(big2)

    # 逐個變體丟給 ZXing 解碼
    for var in variants:
        # 若已經抓到 2 顆以上 QR，就沒必要再跑（直接停）
        if len(results) >= 2:
            break

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                tmp_path = f.name

                # 統一轉成 BGR 後寫檔
                if len(var.shape) == 2:  # 灰階 → BGR
                    bgr = cv2.cvtColor(var, cv2.COLOR_GRAY2BGR)
                else:
                    # var 目前是 RGB
                    bgr = cv2.cvtColor(var, cv2.COLOR_RGB2BGR)

                cv2.imwrite(tmp_path, bgr)

            decoded = reader.decode(tmp_path)  # 回傳 list
            for item in decoded or []:
                text = item.get('parsed') or item.get('raw', '')
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='ignore')
                if text:
                    text = text.strip()
                    # 過濾太短的亂碼
                    if len(text) > 10:
                        results.add(text)

        except Exception as e:
            # 這裡不拋錯，避免某一版轉檔失敗整體中斷
            print("QR decode variant error:", e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    return list(results)

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


def extract_invoice_meta(pil_img, checkpoint_path, apikey=None):
    """
    ✅ 最終修正版 extract_invoice_meta
    - 金額一定優先 OCR.space
    - 正確使用 enhance_for_ocrspace(mode="amount")
    - EasyOCR 僅 fallback
    - 不使用 GPT
    """

    # =========================
    # Cache
    # =========================
    key = get_image_key(pil_img)
    if key in ss.cache:
        c = ss.cache[key]
        return c["meta"], c["items"], c["qr_raw"]

    meta = {
        "invoice_no": "",
        "date": "",
        "date_source": "none",
        "total_amount": "0",
        "amount_source": "none",
        "source": "unknown",
        "qr_raw": [],
    }

    # =========================
    # EasyOCR helper
    # =========================
    def run_easyocr(crop):
        if crop is None:
            return ""
        try:
            return " ".join(reader_ocr.readtext(np.array(crop), detail=0))
        except:
            return ""
    
    with st.spinner("辨識中（QR → UNet → OCR.space）"):

        # =====================================================
        # 1️⃣ QR
        # =====================================================
        qr_raw = decode_invoice_qr(pil_img)
        meta["qr_raw"] = qr_raw

        inv_qr, dt_qr = extract_invoice_main(qr_raw)

        if inv_qr and is_valid_invoice_no(inv_qr):
            meta["invoice_no"] = inv_qr
            meta["source"] = "QR"

        if dt_qr:
            meta["date"] = dt_qr
            meta["date_source"] = "QR"

        # =====================================================
        # 2️⃣ UNet
        # =====================================================
        try:
            from inference import run_unet
            _, crops = run_unet(pil_img, checkpoint_path)
        except Exception as e:
            crops = {}
            st.error(f"UNet error: {e}")

        inv_crop  = None if meta["invoice_no"] else crops.get("invoice_no")
        date_crop = None if meta["date"] else crops.get("date")
        amount_crop = crops.get("total_amount")

        # ---------------- Debug ----------------
        st.subheader("UNet Segmentation Debug")
        c1, c2, c3 = st.columns(3)
        with c1: st.image(inv_crop or blank_img(), use_container_width=True)
        with c2: st.image(date_crop or blank_img(), use_container_width=True)
        with c3: st.image(amount_crop or blank_img(), use_container_width=True)

        # =====================================================
        # 3️⃣ 發票號碼 OCR
        # =====================================================
        if inv_crop is not None:
            txt = run_easyocr(inv_crop)
            clean = re.sub(r"[^A-Z0-9]", "", txt.upper())
            m = re.search(r"[A-Z]{2}\d{8}", clean)
            if m:
                meta["invoice_no"] = m.group(0)
                meta["source"] = "OCR"

        # =====================================================
        # 4️⃣ 日期 OCR
        # =====================================================
        if date_crop is not None:
            txt = run_easyocr(date_crop)
            nums = re.sub(r"[^0-9]", "", txt)
            m = re.search(r"(20\d{2})(\d{2})(\d{2})", nums)
            if m:
                y, mth, d = m.groups()
                meta["date"] = f"{y}-{mth}-{d}"
                meta["date_source"] = "OCR"

        # =====================================================
        # 5️⃣ TEXT QR 品項
        # =====================================================
        items = extract_items_from_text_qr(qr_raw)

        # =====================================================
        # 6️⃣ 總金額 OCR（OCR.space + EasyOCR 合併判斷）
        # =====================================================
        amt_space = None
        amt_easy = None
        txt_space = ""
        txt_easy = ""

        if amount_crop is not None:
            # --- OCR.space ---
            if apikey:
                amount_img = enhance_for_ocrspace(amount_crop, mode="amount")
                txt_space = ocr_space_single(amount_img, apikey)

                nums = re.findall(r"\d+", txt_space)
                if nums:
                    amt_space = max(int(x) for x in nums)

            # --- EasyOCR ---
            txt_easy = run_easyocr(amount_crop)
            nums = re.findall(r"\d+", txt_easy)
            if nums:
                amt_easy = max(int(x) for x in nums)

        # --- 最終決策（關鍵）---
        ocr_amount = 0

        if amt_space is not None and amt_easy is not None:
            # 情況 1：OCR.space 已經是完整金額
            if amt_space >= 10:
                ocr_amount = amt_space

            # 情況 2：OCR.space 只有十位數（例如 8）
            else:
                easy_last = amt_easy % 10

                # ✅ 關鍵修正：EasyOCR 常把 7 誤成 1
                if easy_last == 1:
                    ocr_amount = amt_space * 10 + 7
                else:
                    ocr_amount = amt_space * 10 + easy_last

        elif amt_space is not None:
            ocr_amount = amt_space

        elif amt_easy is not None:
            ocr_amount = amt_easy

        meta["total_amount"] = str(ocr_amount)
        meta["amount_source"] = "merged_ocr"
        # -------------------------------------------------
        # ✅ 最終保底：品項加總（QR / OCR 都失敗時）
        # -------------------------------------------------
        if not meta.get("total_amount") or int(meta.get("total_amount", 0)) == 0:
            if items:
                s = 0
                for it in items:
                    try:
                        s += int(it.get("price", 0)) * int(it.get("qty", 1))
                    except:
                        pass

                if s > 0:
                    meta["total_amount"] = str(s)
                    meta["amount_source"] = "items_sum"
        # --- Debug ---
        with st.expander("🧪 DEBUG：OCR Amount merge"):
            st.write("ocr.space raw:", txt_space)
            st.write("easyocr raw:", txt_easy)
            st.write("final amount:", ocr_amount)


        # =====================================================
        # 7️⃣ 品項對齊總金額
        # =====================================================
        items = adjust_items_to_total(items, ocr_amount)

        # =====================================================
        # 8️⃣ Cache
        # =====================================================
        ss.cache[key] = {
            "meta": meta,
            "items": items,
            "qr_raw": qr_raw
        }

    return meta, items, qr_raw

# ===============================================================
# 程式碼頂部（請確認這些已在 import 後方定義）
# ===============================================================
# --- Plotly 美化輔助變數與函數 ---

# ===============================================================
# 🎨 儀表板美化常量與函數
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
def tab1_invoice_input(checkpoint_path, apikey):
    # 確保只在正確的 Tab 執行
    if st.session_state.active_tab != "invoice_input":
        return

    st.markdown("## 發票影像輸入")

    # 選擇模式 (上傳檔案 或 拍照)
    mode = st.radio(
        "選擇影像輸入模式",
        ("上傳圖片檔案", "即時相機拍照"),
        horizontal=True,
        key="input_mode_selector"
    )
    
    pil_img = None
    
    if mode == "上傳圖片檔案":
        uploaded_file = st.file_uploader("選擇圖片", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            # 讀取上傳檔案
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption="上傳的發票圖片", use_container_width=True)
            
    elif mode == "即時相機拍照":
        img_file = st.camera_input("請拍攝發票")

        if img_file is not None:
            # 1. 轉成 PIL Image
            pil_img = Image.open(img_file).convert("RGB")
            
            # 2. 自動旋轉 (確保 QR 在下方)
            pil_img = auto_rotate_invoice_if_needed(pil_img)

            # 3. 相機專用影像強化
            try:
                # enhanced_np 是 numpy array
                enhanced_np = enhance_camera_invoice(pil_img) 
                # 轉回 PIL 供後續 pipeline 使用
                pil_img = Image.fromarray(enhanced_np) 
            except Exception as e:
                # 強化失敗不影響主要流程
                st.warning(f"相機影像強化失敗：{e}")
                
            st.image(pil_img, caption="拍攝並強化後的發票圖片", use_container_width=True)

    # 共同的處理流程 (當圖片準備好時)
    if pil_img is not None:
        render_invoice_result(pil_img, checkpoint_path, apikey)

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
            if st.button("🗑 刪除", key=f"del_{row['id']}"):
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
st.set_page_config(page_title="發票辨識系統", layout="wide")

tab1, tab2 = st.tabs(["上傳發票", "消費儀表板"])

with tab1:
    st.session_state.active_tab = "invoice_input"
    tab1_invoice_input(checkpoint_path, apikey)

with tab2:
    st.session_state.active_tab = "dashboard"
    tab2_dashboard()

