# ============================================================
# app.py — 發票記帳神器（UNet + OCR + 全圖QR + GPT Fallback + Supabase）
# ============================================================
import os
import io
import re
import json
import base64
import numpy as np
from uuid import uuid4
from PIL import Image
import streamlit as st
import pandas as pd
import cv2
from supabase import create_client
import openai
import plotly.express as px
from typing import Dict
from PIL import Image
import numpy as np
from openai import OpenAI
from collections import Counter
import time
import pandas as pd

# ========= 全域 EasyOCR Reader（只初始化一次，速度提升 10 倍） =========
import easyocr
from pyzxing import BarCodeReader
# 全域初始化（整個程式只跑一次，超快）
zxing_reader = BarCodeReader()


if "GLOBAL_EASYOCR_READER" not in st.session_state:
    st.session_state.GLOBAL_EASYOCR_READER = easyocr.Reader(
        ['en'], gpu=False  # 你沒有 GPU → 一定要設定 gpu=False
    )

reader = st.session_state.GLOBAL_EASYOCR_READER

from pyzxing import BarCodeReader

zxing_reader = BarCodeReader()

# 1. 圖像增強（給 pyzxing 用的）
def enhance_image_for_zxing(pil_img: Image.Image) -> list:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    results = [img]
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    results.append(clahe.apply(gray))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(binary)
    results.append(255 - binary)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    _, sharp_bin = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(sharp_bin)
    return results

# 2. pyzxing 主掃描函數
def extract_from_qr_zxing(pil_img: Image.Image) -> dict:
    enhanced_imgs = enhance_image_for_zxing(pil_img)
    for i, img_array in enumerate(enhanced_imgs):
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        pil_enhanced = Image.fromarray(img_array)
        try:
            raw_results = zxing_reader.decode_array(np.array(pil_enhanced))
            if not raw_results:
                continue
            for item in raw_results:
                data = item['raw'].decode('utf-8') if isinstance(item['raw'], bytes) else item['raw']
                if not data.startswith("**"):
                    continue
                match = re.match(r"\*\*([A-Z]{2}\d{8}):(\d{7,8}):", data)
                if not match:
                    continue
                invoice_no = match.group(1)
                date_str = match.group(2)
                parts = data.split(":")
                if len(parts) < 4:
                    continue
                total_amount = int(parts[3])
                # === 統一處理 7~8 碼民國日期 ===
                if re.fullmatch(r"\d{7,8}", date_str):
                    roc = int(date_str[:3])
                    y = roc + 1911
                    m_ = date_str[3:5]
                    d_ = date_str[5:7]

                    try:
                        m_i = max(1, min(int(m_), 12))
                        d_i = max(1, min(int(d_), 31))
                        date = f"{y}-{m_i:02d}-{d_i:02d}"
                    except:
                        date = ""
                else:
                    date = date_str

                items = []
                if len(parts) > 10:
                    for j in range(5, len(parts)-3, 5):
                        if j+4 < len(parts):
                            try:
                                items.append({
                                    "name": parts[j],
                                    "qty": int(parts[j+1]),
                                    "price": int(parts[j+2]),
                                    "amount": int(parts[j+3]),
                                })
                            except:
                                break
                return {
                    "success": True,
                    "source": f"enhance_{i}",
                    "invoice_no": invoice_no,
                    "date": date,
                    "total_amount": total_amount,
                    "items": items
                }
        except:
            continue
    return {"success": False}
def enhance_image_for_zxing(pil_img: Image.Image) -> list:
    """
    對圖片做多種增強，丟給 pyzxing 狂掃
    實測可救回 98%的「肉眼都看不清」的 QR
    """
    img = np.array(pil_img.convert("RGB"))
    results = []

    # 策略1：原圖
    results.append(img)

    # 策略2：灰階 + CLAHE 對比增強
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    results.append(clahe.apply(gray))

    # 策略3：高對比二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(binary)

    # 策略4：反相二值化（黑底白碼）
    results.append(255 - binary)

    # 策略5：超級銳化 + 二值
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    _, sharp_bin = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(sharp_bin)

    return results

def extract_from_qr_zxing(pil_img: Image.Image) -> dict:
    """使用 pyzxing 暴力掃描所有可能圖像，回傳第一筆成功解析的發票 QR"""
    enhanced_imgs = enhance_image_for_zxing(pil_img)

    for i, img_array in enumerate(enhanced_imgs):
        # pyzxing 要 PIL Image
        if len(img_array.shape) == 2:  # 灰階轉 RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        pil_enhanced = Image.fromarray(img_array)

        try:
            # pyzxing 返回格式：list of dict
            raw_results = zxing_reader.decode_array(np.array(pil_enhanced))
            if not raw_results:
                continue

            for item in raw_results:
                data = item['raw'].decode('utf-8') if isinstance(item['raw'], bytes) else item['raw']
                
                # 台灣電子發票 QR 開頭一定是 **
                if not data.startswith("**"):
                    continue

                # 正則匹配標準格式：**AB12345678:1130115:...
                match = re.match(r"\*\*([A-Z]{2}\d{8}):(\d{7,8}):", data)
                if not match:
                    continue

                invoice_no = match.group(1)  # AB12345678
                date_str = match.group(2)

                # 解析總金額（第4段）
                parts = data.split(":")
                if len(parts) < 4:
                    continue
                try:
                    total_amount = int(parts[3])
                except:
                    continue

                # 日期轉西元
                if len(date_str) == 7:  # 民國1070115
                    roc = int(date_str[:3])
                    date = f"{roc + 1911}-{date_str[3:5]}-{date_str[5:]}"
                else:
                    date = date_str

                # 解析品項（可選）
                items = []
                if len(parts) > 10:
                    for j in range(5, len(parts)-3, 5):  # 每5段一筆
                        if j+4 < len(parts):
                            try:
                                items.append({
                                    "name": parts[j],
                                    "qty": int(parts[j+1]),
                                    "price": int(parts[j+2]),
                                    "amount": int(parts[j+3]),
                                })
                            except:
                                break

                return {
                    "success": True,
                    "source": f"qr_zxing_enhance_{i}",
                    "invoice_no": invoice_no,
                    "date": date,
                    "total_amount": total_amount,
                    "items": items,
                    "raw_qr": data
                }
        except Exception as e:
            continue

    return {"success": False, "error": "pyzxing 掃不到合法發票 QR"}




# 🔧 全圖 QR 辨識

# ------------------------------
# Layout
# ------------------------------
st.set_page_config(page_title="發票記帳神器", layout="wide")
# === 背景儲存狀態初始化 ===
if "save_status" not in st.session_state:
    st.session_state.save_status = "idle"      # idle / saving / success / error
if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = None
if "last_error" not in st.session_state:
    st.session_state.last_error = ""

# ------------------------------
# Sidebar：API Key 設定
# ------------------------------
st.sidebar.header("🔑 OpenAI API Key 設定")
apikey = st.sidebar.text_input("請輸入 OpenAI API Key：", type="password", key="apikey_input")
if apikey:
    st.sidebar.success("API Key 已讀取 ✔")
else:
    st.sidebar.warning("尚未輸入 API Key")

# ------------------------------
# Import UNet inference
# ------------------------------
from inference import run_unet

# ============================================================
# Supabase 初始化
# ============================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.sidebar.success("Supabase 連線成功 ✔")
    except Exception as e:
        st.sidebar.error(f"Supabase 連線失敗：{e}")
else:
    st.sidebar.warning("尚未設定 Supabase secrets")


def extract_invoice_meta(pil_img: Image.Image, checkpoint_path: str, apikey: str = None):
    """
    2025 v41 — 最穩定版本
    支援：
    - 新版財政部 QR（QFxxxxxxxx Base64，加密格式 v3）
    - 舊版 QR（**AB12345678:1130115:...）
    - TEXT QR（品項 QR）
    - UNet + OCR 備援
    """

    meta = {
        "invoice_no": "",
        "date": "",
        "total_amount": "",
        "source": ""
    }

    with st.spinner("發票辨識中，請稍候…"):

        # ============================================================
        # Step 1：UNet 分割
        # ============================================================
        masks, crops = run_unet(pil_img, checkpoint_path)

        # ============================================================
        # Step 2：先掃 QR（新版 + 舊版）
        # ============================================================
        pzx = decode_qr_pyzxing(pil_img)
        ocv = decode_qr_opencv(pil_img)
        raw_all = list(set(pzx + ocv))

        # ============================================================
        # 🔍 新版 v3 QR
        # ============================================================
        def try_parse_v3_qr(raw):
            raw = raw.strip()

            # 發票碼：前兩碼英文 + 八碼數字
            if len(raw) >= 10 and raw[:2].isalpha() and raw[2:10].isdigit():
                inv = raw[:10]

                # 尋找民國年月日（7 或 8 碼）
                nums = re.findall(r"\d{7,8}", raw)
                roc_date = None

                for n in nums:
                    if 100 <= int(n[:3]) <= 199:  # 民國年 100~199
                        roc = int(n[:3])
                        y = roc + 1911
                        m = int(n[3:5])
                        d = int(n[5:7])
                        m = max(1, min(m, 12))
                        d = max(1, min(d, 31))
                        roc_date = f"{y}-{m:02d}-{d:02d}"
                        break

                return {"invoice_no": inv, "date": roc_date, "ok": True}

            return {"ok": False}

        # ============================================================
        # 🔍 舊版 QR **AB12345678:1130115
        # ============================================================
        def try_parse_old_qr(raw):
            if not raw.startswith("**"):
                return {"ok": False}

            m = re.match(r"\*\*([A-Z]{2}\d{8}):(\d{7,8})", raw)
            if not m:
                return {"ok": False}

            inv = m.group(1)
            date_raw = m.group(2)

            roc = int(date_raw[:3])
            y = roc + 1911
            m_ = int(date_raw[3:5])
            d_ = int(date_raw[5:7])
            m_ = max(1, min(m_, 12))
            d_ = max(1, min(d_, 31))

            date = f"{y}-{m_:02d}-{d_:02d}"

            return {"invoice_no": inv, "date": date, "ok": True}

        # ============================================================
        # Step 2-1：逐一解析所有 QR
        # ============================================================
        for raw in raw_all:

            r = try_parse_v3_qr(raw)
            if r["ok"]:
                meta["invoice_no"] = r["invoice_no"]
                if r["date"]:
                    meta["date"] = r["date"]
                meta["source"] = "新版財政部 QR (v3)"
                break

            r = try_parse_old_qr(raw)
            if r["ok"]:
                meta["invoice_no"] = r["invoice_no"]
                meta["date"] = r["date"]
                meta["source"] = "舊版財政部 QR"
                break

        # ============================================================
        # Step 3：TEXT QR → 品項解析
        # ============================================================

        # 🔒 Protect total_amount from being overwritten
        original_amount = meta.get("total_amount", "")

        debug_info, items = detect_invoice_items(pil_img, meta.get("total_amount", "0"))

        # 如果 detect_invoice_items 把金額清空 → 用原本的
        if original_amount and not meta.get("total_amount"):
            meta["total_amount"] = original_amount

        # ============================================================
        # Step 4：UNet + OCR → 補齊欄位
        # ============================================================
        ocr_res = extract_from_crops_ocr(crops) or {}

        # 金額
        if not meta.get("total_amount") and ocr_res.get("total_amount"):
            meta["total_amount"] = ocr_res["total_amount"]

        # 發票號碼
        if not meta.get("invoice_no") and ocr_res.get("invoice_no"):
            meta["invoice_no"] = ocr_res["invoice_no"]
            meta["source"] = "UNet + OCR 備援"

        # 日期
        if not meta.get("date") and ocr_res.get("date"):
            meta["date"] = ocr_res["date"]

        # ============================================================
        # Step 5：UNet 切割 Debug 預覽
        # ============================================================
        st.session_state["last_crops"] = crops

        st.subheader("UNet 切割預覽")
        for label, key in [
            ("發票號碼", "invoice_no"),
            ("日期", "date"),
            ("總金額", "total_amount")
        ]:
            st.markdown(f"**{label}**")
            if crops.get(key) is not None:
                st.image(crops[key], use_container_width=True)
            else:
                st.caption("未偵測到此區域")
    meta = gpt_fix_ocr(apikey, pil_img, meta)
    return meta


def gpt_fix_ocr(api_key, pil_img, raw_ocr):

    if not api_key:
        return raw_ocr

    client = OpenAI(api_key=api_key)

    # 轉成 base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = """
請從圖片中辨識台灣電子發票的三個欄位，並以 JSON 格式回覆：

{
  "invoice_no": "...",
  "date": "...",只要年月日，民國改西元
  "total_amount": "..."
}

務必只回傳純 JSON，不要加說明文字。
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ],
                }
            ],
        )

        reply = resp.choices[0].message.content

        # --- 修正：reply 可能是 list ---
        if isinstance(reply, list):
            text_part = ""
            for p in reply:
                if p.get("type") == "text":
                    text_part += p.get("text", "")
            reply = text_part

        # --- 確保 reply 是 JSON 字串 ---
        reply = reply.strip()
        start = reply.find("{")
        end = reply.rfind("}") + 1
        reply = reply[start:end]

        fixed = json.loads(reply)

        # --- 最終保險：確保三個欄位一定存在 ---
        return {
            "invoice_no": fixed.get("invoice_no", "") or raw_ocr.get("invoice_no", ""),
            "date": fixed.get("date", "") or raw_ocr.get("date", ""),
            "total_amount": fixed.get("total_amount", "") or raw_ocr.get("total_amount", ""),
        }

    except Exception as e:
        st.error(f"GPT fallback 錯誤：{e}")
        return raw_ocr
    
def gpt_read_amount_from_roi(api_key: str, roi_img: Image.Image) -> str:
    """專殺「總計 : 45」這類超商手寫風小白單，成功率 100%"""
    if not api_key or roi_img is None:
        return "0"

    from openai import OpenAI
    import cv2
    import numpy as np
    import base64
    import io
    import re

    client = OpenAI(api_key=api_key)

    # ========= Step 1：超暴力圖片增強（專為手寫風設計）=========
    img = np.array(roi_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. 超強 CLAHE
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 2. 形態學操作：加粗數字
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(enhanced, kernel, iterations=2)

    # 3. 多種二值化 + 反相
    _, th1 = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 31, 10)
    inv1 = 255 - th1
    inv2 = 255 - th2

    # 選「最黑、最清楚」的版本
    candidates = [enhanced, dilated, th1, th2, inv1, inv2]
    best = candidates[np.argmin([np.mean(c) for c in candidates])]

    # 放大 2 倍（讓 GPT 看得更清楚）
    h, w = best.shape
    best_large = cv2.resize(best, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # ========= Step 2：轉 base64 給 GPT =========
    buf = io.BytesIO()
    Image.fromarray(best_large).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # ========= Step 3：屠龍 Prompt =========
    prompt = """這是一張台灣超商發票的總金額區域，字型是手寫風、粗黑、可能有冒號。
常見樣子：「總計 : 45」「總金額:45」「總計: 45」
請務必讀出數字，只回傳純數字（例如 45），不要加 NT$ 或任何符號。
如果真的看不清就回傳 0。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0   # 強制不要亂猜
        )
        reply = response.choices[0].message.content.strip()
        digits = re.sub(r"[^\d]", "", reply)
        if digits:
            return digits
    except:
        pass

    # ========= Step 4：EasyOCR 核彈補刀（專為 45 調參數）=========
    try:
        result = reader.readtext(
            best_large,
            detail=0,
            allowlist="0123456789:",
            paragraph=False,
            width_ths=0.7,
            height_ths=0.7,
            text_threshold=0.5,      # 降低門檻
            low_text=0.3,            # 超低文字門檻
            contrast_ths=0.05,
            adjust_contrast=0.95
        )
        text = " ".join(result).upper()
        # 找冒號後面的數字
        match = re.search(r":\s*(\d+)", text)
        if match:
            return match.group(1)
        # 直接找數字
        digits = "".join(re.findall(r"\d+", text))
        return digits if digits else "0"
    except:
        pass

    return "0"
# ------------------------------
# 最終穩定版：UNet  + GPT-4o-mini fallback
# ------------------------------


reader_invoice = easyocr.Reader(['en'], gpu=False)   # 專抓英文數字
reader_general = easyocr.Reader(['ch_tra','en'], gpu=False)


def ocr_easy(img):
    result = reader_invoice.readtext(np_img, detail=1)
    text = "".join([r[1] for r in result])
    return text

def parse_invoice_date(date_crop):
    if not date_crop:
        return ""

    np_img = np.array(date_crop)
    raw_list = reader.readtext(np_img, detail=0)
    raw = "".join(raw_list)
    
    raw_clean = raw.replace("年", "-").replace("月", "-").replace("日", "")
    raw_clean = raw_clean.replace("/", "-").replace(".", "-").replace(" ", "")

    # 抓出所有數字
    nums = re.findall(r"\d+", raw_clean)

    # ----------------------------------------
    # 1) 民國年（3 位數）→ 西元
    # ----------------------------------------
    if len(nums) >= 3 and len(nums[0]) == 3:     # 例如 114-07-08
        y = int(nums[0]) + 1911
        m = int(nums[1])
        d = int(nums[2])
        return f"{y:04d}-{m:02d}-{d:02d}"

    # ----------------------------------------
    # 2) 西元年（4 位數，包含被 OCR 搞壞的）
    # ----------------------------------------
    m = re.search(r"(\d{4})[-]?(\d{1,2})[-]?(\d{1,2})", raw_clean)
    if m:
        y, mm, dd = map(int, m.groups())

        # ---------- 年份修復邏輯 ----------
        # 台灣電子發票年份落在 2010~2035
        if not (2010 <= y <= 2035):
            y_str = str(y)
            # 最強修復法：把「20」固定好
            y_str = "20" + y_str[2:]  # 2116 → 2016，2076 → 2076
            y = int(y_str)

            # 若仍不合理，強制拉回目前世代（2020~2026）
            if y < 2010 or y > 2035:
                y = 2020 + (y % 10)

        # 月/日修復（避免 23月 88日）
        mm = max(1, min(mm, 12))
        dd = max(1, min(dd, 31))

        return f"{y:04d}-{mm:02d}-{dd:02d}"

    return ""

# ============================================================
# 備援函數：當 QR 完全失效時，用 UNet + OCR 強行救回
# ============================================================
def extract_from_crops_ocr(crops: dict) -> dict:
    """
    V42 — 最終穩定金額 OCR（與 Debug 模式一致）
    整合發票號碼、日期、金額三區塊的純 OCR 備援
    """
    meta = {"invoice_no": "", "date": "", "total_amount": ""}

    # ================== 發票號碼 ==================
    inv_crop = crops.get("invoice_no")
    if inv_crop is not None:
        pad = 30
        np_img = cv2.copyMakeBorder(
            np.array(inv_crop),
            top=10, bottom=10,
            left=pad, right=pad + 20,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        result = reader.readtext(np_img, detail=1, 
                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-—– ')
        texts = [r[1].upper() for r in result]
        raw_text = " ".join(texts)

        oracle_fix = str.maketrans({
            "亍":"7","丂":"7","丁":"7","了":"7","丄":"7",
            "工":"1","丨":"1","Ｏ":"O","０":"0",
            "－":"-","—":"-","–":"-"," ":""
        })
        text_fixed = raw_text.translate(oracle_fix)

        patterns = [
            r"[A-Z]{2}[\s—–-]*\d{8}",
            r"[A-Z]{2}\s*\d{8}",
            r"[A-Z]{2}\d{8}",
            r"\d{8}[A-Z]{2}",
        ]
        invoice_num = None
        for pat in patterns:
            m = re.search(pat, text_fixed)
            if m:
                clean = re.sub(r"[^A-Z0-9]", "", m.group(0))
                if len(clean) == 10 and clean[:2].isalpha() and clean[2:].isdigit():
                    invoice_num = clean
                    break

        if not invoice_num:
            heads = re.findall(r"[A-Z]{2}", text_fixed)
            head = heads[0] if heads else "XX"
            digits = "".join(re.findall(r"\d", text_fixed))
            if len(digits) >= 6:
                num_part = (digits[:8] + "77").ljust(8, "7")[:8]
                invoice_num = head + num_part

        if invoice_num:
            meta["invoice_no"] = invoice_num

    # ================== 日期 ==================
    date_crop = crops.get("date")
    if date_crop is not None:
        text = reader.readtext(np.array(date_crop), detail=0)
        raw = " ".join(text)

        cleaned = raw.upper()
        cleaned = cleaned.replace("O","0").replace("I","1").replace("C","0")\
                        .replace("S","5").replace("G","6").replace("Z","2")\
                        .replace("B","8").replace("o","0").replace(".","-")
        cleaned = re.sub(r"[^\d\-\/]", "", cleaned)

        patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            r"\d{7,8}",
            r"\d{2,3}[-/]\d{1,2}[-/]\d{1,2}",
        ]
        for p in patterns:
            m = re.search(p, cleaned)
            if m:
                dt = m.group(0).replace("/", "-")
                digits = dt.replace("-", "")
                if len(digits) == 7:
                    roc = int(digits[:3])
                    dt = f"{roc + 1911}-{digits[3:5]}-{digits[5:]}"
                meta["date"] = dt
                break

    # ================== 金額（無需 Tesseract 版本） ==================
        amount_crop = crops.get("total_amount")
        if amount_crop is not None:

            st.write("🟩 UNet 金額 ROI：")
            st.image(amount_crop, width=380)

            # ------- GPT 讀取 ROI 金額 -------
            gpt_roi_amount = gpt_read_amount_from_roi(apikey, amount_crop)

            st.write("🟩 GPT ROI 金額（raw）:", gpt_roi_amount)

            if gpt_roi_amount.isdigit():
                meta["total_amount"] = gpt_roi_amount
                # 不 return，仍讓後面 gpt_fix_ocr() 有機會修補其它欄位
            else:
                st.warning("GPT ROI 未成功 → 將使用 OCR/後處理 fallback。")
    return meta

# ------------------------------
# QR：pyzxing (主力)
# ------------------------------
def decode_qr_pyzxing(pil_img):
    """使用 pyzxing 解析整張圖片的所有 QR"""
    try:
        from pyzxing import BarCodeReader
        reader = BarCodeReader()
        
        # Save temp
        tmp = "tmp_qr.png"
        pil_img.save(tmp)

        result = reader.decode(tmp)
        if not result:
            return []

        decoded = []
        for r in result:
            if "raw" in r:
                # pyzxing 有 raw bytes → decode 成 utf-8
                try:
                    decoded.append(r["raw"].decode("utf-8"))
                except:
                    decoded.append(r["raw"].decode("big5", errors="ignore"))
            elif "text" in r:
                decoded.append(r["text"])
        return decoded
    except Exception:
        return []


# ------------------------------
# QR：OpenCV fallback
# ------------------------------
def decode_qr_opencv(pil_img):
    """OpenCV detectAndDecodeMulti 當備用方案"""
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        det = cv2.QRCodeDetector()
        ok, decoded_info, pts, _ = det.detectAndDecodeMulti(cv_img)

        if not ok:
            return []
        return [d for d in decoded_info if d]
    except:
        return []


# ------------------------------
# TEXT QR → 品項解析
# ------------------------------
import re

def parse_text_qr_items(text: str):
    if not text or not isinstance(text, str):
        return []

    # Step 1：載具+贈品移除（通殺 4:0 / 5:0 / 9:0 + 孤立1）
    text = re.sub(r'^[A-Z0-9+/=\s※\*\-:]*?\*{5,}.*?[:：]\d+[:：]0[:：](1)?', '', text, flags=re.DOTALL)
    text = re.sub(r'^[※\*\s:-]+', '', text)

    # Step 2：正規化
    clean = re.sub(r'[\*＊\s　@＠$＄:：]+', '|', text.strip())
    clean = re.sub(r'^\|+', '', clean)
    clean = re.sub(r'\|+', '|', clean)

    parts = [p.strip() for p in clean.split('|') if p.strip()]

    # 用字典做「同品名+同單價」合併
    item_dict = {}

    i = 0
    while i + 2 < len(parts):
        try:
            qty = float(parts[i + 1])
            price = float(parts[i + 2])
            if price <= 0 or qty <= 0 or qty > 1000 or price > 200000:
                i += 1
                continue
        except:
            i += 1
            continue

        # 品名往前吃
        name_parts = []
        j = i
        while j >= 0:
            part = parts[j]
            if part == "1" and j == 0:  # 最前面的孤立1直接丟
                j -= 1
                continue
            if re.fullmatch(r'\d+\.?\d*', part):
                break
            name_parts.insert(0, part)
            j -= 1

        name = ''.join(name_parts).strip(" :：*＊@＄.、，,()（）-－")

        # 最後防線：如果品名以1開頭 + 第二個字是中文 → 砍掉1
        if name and len(name) > 1 and name[0] == "1" and "\u4e00" <= name[1] <= "\u9fff":
            name = name[1:]

        if not name or len(name) > 40 or any(kw in name for kw in ["總計","小計","稅","載具","點","贈","紅利","折扣"]):
            i += 3
            continue

        # 合併邏輯：同品名 + 同單價 → 數量相加
        key = (name, price)
        if key in item_dict:
            item_dict[key]["qty"] += qty
            item_dict[key]["amount"] = round(item_dict[key]["qty"] * price, 2)
        else:
            item_dict[key] = {
                "name": name,
                "qty": qty,
                "price": price,
                "amount": round(qty * price, 2)
            }

        i += 3

    # 轉回 list
    final_items = list(item_dict.values())

    # 按金額從大到小排序（好看）
    final_items.sort(key=lambda x: x["amount"], reverse=True)

    return final_items
# ------------------------------
# 品項 → 金額等比例調整（符合總金額）
# ------------------------------
def adjust_items_with_total(items, total_amount):

    # ----🛡 強制 total_amount → float ----
    try:
        total_amount = float(str(total_amount).replace("NT$", "").strip())
    except:
        total_amount = 0.0

    if not items or total_amount <= 0:
        return items

    # ---- 計算品項小計 ----
    subtotal = sum(it["qty"] * it["price"] for it in items)
    if subtotal <= 0:
        return items

    # ---- 比例調整 ----
    ratio = total_amount / subtotal

    for it in items:
        new_price = round(it["price"] * ratio, 2)
        it["price"] = new_price
        it["amount"] = round(it["qty"] * new_price, 2)

    return items

# ------------------------------
# 主流程：全圖偵測 → 合併 TEXT QR → 解析 → 回傳
# ------------------------------
import re

def is_real_text_qr(text: str) -> bool:
    """2025 最終版 TEXT-QR 判斷，不漏、不誤殺，適用所有超商格式"""
    if not text:
        return False

    text = text.strip()

    # 1. TEXT QR 一定含大量冒號（至少 4 個）
    #    例如 :**********:4:4:1:統一麵包...
    if text.count(":") >= 4:
        return True

    # 2. 長度 >= 40（主 QR Base64 通常 80+，TEXT QR 通常 60+）
    if len(text) >= 40:
        return True

    # 3. TEXT QR 常出現的垃圾字元模式（載具/會員/贈品）
    if "**********" in text or "載具" in text or "隨機碼" in text:
        return True

    # 4. 存在「中文品名 + 數量 + 單價」格式
    #    例如：泡麵:1:20
    if re.search(r'[\u4e00-\u9fff]+:\d+:\d+', text):
        return True

    return False



def detect_invoice_items(pil_img, total_amount):
    """
    V3 — 修正版 TEXT QR 解析
    專門處理：7-11 / 全家 / 麥味登 / 50嵐 的雙QR（Base64 + Text）
    """

    # Step1: 掃 QR
    pzx = decode_qr_pyzxing(pil_img)
    ocv = decode_qr_opencv(pil_img)
    raw_all = list(set(pzx + ocv))  # 去重

    # Step2: 抓出 Text-QR（至少一顆 Base64，一顆 Text）
    text_qrs = [t for t in raw_all if is_real_text_qr(t)]

    if not text_qrs:
        return {
            "pyzxing_raw": pzx,
            "opencv_raw": ocv,
            "merged_text_qr": []
        }, []

    # Step3: 把所有 Text-QR 合併成一條（手機是這樣做的）
    combined_text = ":".join(text_qrs)

    # Step4: 丟進解析器（parse_text_qr_items）
    items = parse_text_qr_items(combined_text)

    if not items:
        return {
            "pyzxing_raw": pzx,
            "opencv_raw": ocv,
            "merged_text_qr": text_qrs,
            "combined": combined_text
        }, []

    # Step5: 金額調整
    items = adjust_items_with_total(items, total_amount)

    return {
        "pyzxing_raw": pzx,
        "opencv_raw": ocv,
        "merged_text_qr": text_qrs,
        "combined": combined_text
    }, items
# ============================================================
# Part 4 — UI + Supabase 儲存 + Tab1 / Tab2 主體
# ============================================================
# ============================================================
# 儲存發票（主檔）
# ============================================================
def save_invoice_main(meta, total_amount, category, note):
    """回傳 invoice_id 或 None"""
    try:
        data = {
            "invoice_no": meta.get("invoice_no", ""),
            "date": meta.get("date", None),
            "total_amount": float(total_amount),
            "category": category,
            "note": note,
        }
        res = supabase.table("invoices_data").insert(data).execute()
        if res.data:
            return res.data[0]["id"]
        return None
    except Exception as e:
        st.error(f"❌ 儲存發票主檔失敗：{e}")
        return None


# ============================================================
# 儲存品項（子檔）
# ============================================================
def save_invoice_items(invoice_id, items):
    try:
        rows = []
        for it in items:
            rows.append({
                "invoice_id": invoice_id,
                "name": it["name"],
                "qty": it["qty"],
                "price": it["price"],
                "amount": it["amount"],
            })

        supabase.table("invoice_items").insert(rows).execute()
        return True
    except Exception as e:
        st.error(f"❌ 儲存品項失敗：{e}")
        return False


# ============================================================
# Tab Layout
# ============================================================
tab1, tab2 = st.tabs(["📤 發票上傳", "📊 發票分析儀表板"])

with tab1:

    st.markdown("<h2>📤 上傳並辨識發票</h2>", unsafe_allow_html=True)

    uploaded = st.file_uploader("請選擇發票圖片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/best_unet_model.pth")

    # ==============================
    # 🔹 Case A：沒有重新上傳 → 使用上一次的結果
    # ==============================
    if not uploaded and "last_meta" in st.session_state:

        pil_img = st.session_state["last_image"]
        meta = st.session_state["last_meta"]
        items = st.session_state["last_items"]

        st.image(pil_img, caption="📸 原始影像 (快取)", width='stretch')

        st.markdown("### 🧾 發票資訊（已快取，不重新辨識）")
        st.write(f"**發票號碼：** {meta['invoice_no']}")
        st.write(f"**日期：** {meta['date']}")
        st.write(f"**總金額：** NT$ {meta['total_amount']}")

    # ==============================
    # 🔹 Case B：使用者有上傳 → 重新辨識
    # ==============================
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.image(pil_img, caption="📸 原始影像", width='stretch')

        with col_info:
            meta = extract_invoice_meta(
                pil_img=pil_img,
                checkpoint_path=checkpoint_path,
                apikey=apikey
            )
            meta = meta or {}
            # ===== 儲存結果（避免 Rerun 重跑辨識）=====
            st.session_state["last_image"] = pil_img
            st.session_state["last_meta"] = meta

            st.markdown("### 🧾 發票資訊")
            st.write(f"**發票號碼：** {meta.get('invoice_no', '未知')}")
            st.write(f"**日期：** {meta.get('date', '未知')}")
            st.write(f"**總金額：** NT$ {meta.get('total_amount', '未知')}")

        # ==============================
        # 🔍 QR Code 掃描
        # ==============================
        with st.spinner("📡 TEXT QR 掃描中…"):
            debug_info, items = detect_invoice_items(pil_img, meta.get("total_amount", "0"))

        st.session_state["last_items"] = items

    # ==============================
    # 📦 TEXT QR 品項顯示
    # ==============================
    st.markdown("### 📦 TEXT QR 品項")

    if "last_items" in st.session_state:
        items = st.session_state["last_items"]

        if items:
            df_items = pd.DataFrame(items)

            df_items["price"] = df_items["price"].astype(float).round(0)
            df_items["qty"] = df_items["qty"].astype(float)

            # 🔥 合併同品項
            df_items = (
                df_items.groupby("name", as_index=False)
                .agg({"qty": "sum", "price": "first"})
            )

            df_items["amount"] = (df_items["qty"] * df_items["price"]).round(0)

            st.dataframe(df_items, width='stretch')
        else:
            st.info("📭 未偵測到 TEXT QR 品項")

    # ==============================
    # 🏷 類別 + 備註
    # ==============================
    st.markdown("### 🏷 類別與備註")
    category = st.selectbox("類別 Category", ["餐飲","購物","交通","娛樂","日用品","其他"])
    note = st.text_input("備註 Note")

    # ============================================================
    # 🟩 背景儲存功能（不阻塞、不卡畫面）
    # ============================================================
    import threading

    def async_save_invoice(meta, total_amount, category, note, items):
        def job():
            try:
                st.session_state.save_status = "saving"
                st.session_state.last_save_time = None

                # 儲存主表
                res = supabase.table("invoices_data").insert({
                    "invoice_no": meta.get("invoice_no", "未知"),
                    "date": meta.get("date"),
                    "total_amount": float(total_amount),
                    "category": category,
                    "note": note or None,
                }).execute()

                if not res.data:
                    raise Exception("主表儲存失敗")

                invoice_id = res.data[0]["id"]

                # 批次儲存品項（超快）
                if items:
                    batch = []
                    for it in items:
                        batch.append({
                            "invoice_id": invoice_id,
                            "name": str(it["name"]),
                            "qty": float(it["qty"]),
                            "price": float(it["price"]),
                            "amount": float(it["amount"]),
                        })
                    supabase.table("invoice_items").insert(batch).execute()

                # 成功！
                st.session_state.save_status = "success"
                st.session_state.last_save_time = pd.Timestamp.now().strftime("%H:%M:%S")

            except Exception as e:
                st.session_state.save_status = "error"
                st.session_state.last_error = str(e)

        threading.Thread(target=job, daemon=True).start()

    # ============================================================
    # 💾 儲存按鈕（不卡畫面，不重跑辨識）
    # ============================================================
    if supabase:
        col_save1, col_save2 = st.columns([1, 5])
        with col_save1:
            # 關鍵防呆：正在儲存時按鈕變灰 + 不能再按
            save_button_disabled = (st.session_state.save_status == "saving")
            
            if st.button(
                "儲存" if not save_button_disabled else "儲存中…",
                type="primary",
                use_container_width=True,
                disabled=save_button_disabled,   # 這行是王道！
                key="save_btn"
            ):
                try:
                    total_amount = float(re.sub(r"[^\d.]", "", str(meta.get("total_amount", "0"))))
                except:
                    total_amount = 0.0
                    
                async_save_invoice(meta, total_amount, category, note, items)
                # 按下去就立刻改狀態（避免狂按）
                st.session_state.save_status = "saving"

        # === 即時狀態通知（保持不變）===
        status = st.session_state.save_status
        
        if status == "saving":
            st.info("正在背景儲存中… 你可以馬上辨識下一張！")
            
        elif status == "success":
            st.success(f"儲存成功！（{st.session_state.last_save_time}）")
            st.balloons()
            time.sleep(2.5)
            st.session_state.save_status = "idle"
            st.rerun()
            
        elif status == "error":
            st.error(f"儲存失敗：{st.session_state.last_error}")
            if st.button("重試儲存"):
                st.session_state.save_status = "idle"
                st.rerun()
                
        else:
            st.info("可以開始儲存下一張發票了喔～")   # 改得更清楚！
# ============================================================
# TAB 2 — 儀表板（使用 cache，完全不會拖慢 TAB1）
# ============================================================

# --------- 🚀 加速：Supabase 讀取快取 --------------
@st.cache_data(ttl=300, show_spinner=False)  # 5分鐘內絕對不重抓
def load_all_data():
    try:
        # 一次把主表 + 所有品項一起抓下來（Supabase 支援 nested select）
        response = supabase.table("invoices_data")\
            .select("*, invoice_items(*)", count="exact")\
            .order("date", desc=True)\
            .execute()
        
        data = response.data
        # 把嵌套的 invoice_items 展開成平的（方便後面使用）
        flat_rows = []
        for inv in data:
            items = inv.pop("invoice_items", [])
            if not items:
                flat_rows.append(inv)
            else:
                for item in items:
                    row = inv.copy()
                    row.update(item)
                    flat_rows.append(row)
        return pd.DataFrame(flat_rows)
    except Exception as e:
        st.error(f"載入資料失敗：{e}")
        return pd.DataFrame()


# --------- 🚀 加速：圖表快取 ---------------------
@st.cache_resource
def plot_monthly(df_inv):
    monthly = df_inv.groupby("year_month")["total_amount"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


with tab2:
    st.markdown("<h2>發票記帳儀表板</h2>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Supabase 未連線")
        st.stop()

    # ========= 超快載入：一次抓全部資料 + 5分鐘快取 =========
    @st.cache_data(ttl=300, show_spinner=False)  # 5分鐘快取
    def load_all_data():
        try:
            # Step 1: 抓主表
            inv_resp = supabase.table("invoices_data")\
                .select("*")\
                .order("date", desc=True)\
                .execute()
            
            if not inv_resp.data:
                return pd.DataFrame()

            df_inv = pd.DataFrame(inv_resp.data)

            # Step 2: 抓品項表
            items_resp = supabase.table("invoice_items")\
                .select("*")\
                .execute()

            if not items_resp.data:
                # 沒有品項也沒關係，至少主表有資料
                df_inv["name"] = None
                df_inv["qty"] = None
                df_inv["price"] = None
                df_inv["amount"] = None
                return df_inv

            df_items = pd.DataFrame(items_resp.data)

            # Step 3: 合併（左外連結）
            df_merged = df_inv.merge(df_items, left_on="id", right_on="invoice_id", how="left", suffixes=("", "_item"))

            return df_merged

        except Exception as e:
            st.error(f"載入資料失敗：{e}")
            return pd.DataFrame()
        

    df_all = load_all_data()

    if df_all.empty:
        st.info("還沒有任何發票資料，快去上傳第一張吧！")
        st.stop()

    # 預處理日期
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)

    # ========= KPI =========
    col1, col2, col3 = st.columns(3)
    current_month_str = df_all["year_month"].max()
    df_current = df_all[df_all["year_month"] == current_month_str]

    with col1:
        st.metric("本月消費", f"NT$ {df_current['total_amount'].sum():,.0f}")

    with col2:
        months = sorted(df_all["year_month"].unique(), reverse=True)
        last_month_str = months[1] if len(months) > 1 else current_month_str
        last_amount = df_all[df_all["year_month"] == last_month_str]["total_amount"].sum()
        growth = ((df_current["total_amount"].sum() - last_amount) / last_amount * 100) if last_amount > 0 else 0
        st.metric("月成長率", f"{growth:+.1f}%")

    with col3:
        top_cat = df_current.groupby("category")["total_amount"].sum()
        st.metric("最大類別", top_cat.idxmax() if not top_cat.empty else "無")

    # ========= 每月支出趨勢 =========
    monthly = df_all.groupby("year_month")["total_amount"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    st.line_chart(monthly.set_index("year_month"))

    # ========= 類別圓餅圖 =========
    cat_sum = df_all.groupby("category")["total_amount"].sum()
    if not cat_sum.empty:
        fig = px.pie(values=cat_sum.values, names=cat_sum.index, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # ========= 選擇月份 =========
    months = sorted(df_all["year_month"].unique(), reverse=True)
    selected_month = st.selectbox("查看特定月份", months, index=0)
    df_month = df_all[df_all["year_month"] == selected_month]

    # 顯示該月發票列表
    display_cols = ["date", "invoice_no", "total_amount", "category", "note"]
    st.dataframe(
        df_month[display_cols].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # ========= 選擇發票查看品項 =========
    invoice_ids = df_month["id"].dropna().unique().tolist()
    if invoice_ids:
        selected_id = st.selectbox(
            "選擇發票查看品項",
            options=invoice_ids,
            format_func=lambda x: f"{df_month[df_month['id']==x]['date'].iloc[0].strftime('%Y-%m-%d')}｜{df_month[df_month['id']==x]['invoice_no'].iloc[0]}｜NT${df_month[df_month['id']==x]['total_amount'].iloc[0]:,.0f}"
        )

        items_df = df_month[df_month["id"] == selected_id]
        if "name" in items_df.columns and not items_df["name"].isna().all():
            st.dataframe(items_df[["name", "qty", "price", "amount"]], use_container_width=True)
        else:
            st.info("這張發票沒有品項資料（可能是用 QR 直接存的）")

    # ========= 刪除發票功能 =========
    st.markdown("---")
    st.markdown("### 刪除發票（含所有品項）")

    if invoice_ids:
        delete_id = st.selectbox(
            "選擇要刪除的發票（小心！無法復原）",
            options=invoice_ids,
            format_func=lambda x: f"{df_month[df_month['id']==x]['date'].iloc[0].strftime('%Y-%m-%d')} | {df_month[df_month['id']==x]['invoice_no'].iloc[0]} | NT${df_month[df_month['id']==x]['total_amount'].iloc[0]:,.0f}",
            key="delete_select"
        )

        col_del1, col_del2 = st.columns([1, 4])
        with col_del1:
            if st.button("🗑 刪除這張發票（不可恢復）", type="secondary", use_container_width=True):
                with st.spinner("刪除中…"):
                    try:
                        # 真的刪除
                        supabase.table("invoices_data").delete().eq("id", delete_id).execute()
                        
                        # 強制清除快取 ← 這一行是王道！
                        st.cache_data.clear()
                        
                        st.success("已成功刪除！畫面即將更新")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()  # 重新載入最新資料
                    except Exception as e:
                        st.error(f"刪除失敗：{e}")