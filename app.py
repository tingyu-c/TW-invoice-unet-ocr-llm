# ============================================================
# app.py v42 — 發票記帳神器（UNet + OCR + 全圖QR + GPT Fallback + Supabase）
# ============================================================

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
import pytesseract
from supabase import create_client
import openai
import plotly.express as px


# 🔧 全圖 QR 辨識
from pyzxing import BarCodeReader

# ------------------------------
# Tesseract for Windows
# ------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ------------------------------
# Layout
# ------------------------------
st.set_page_config(page_title="發票記帳神器 v42", layout="wide")

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
from inference import run_unet_inference

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


# ============================================================
# Part 2 — UNet → OCR → GPT fallback 修正
# ============================================================

# ------------------------------
# OCR：Tesseract
# ------------------------------
def ocr_text(pil_img):
    """使用 Tesseract OCR 讀取裁切影像"""
    try:
        text = pytesseract.image_to_string(pil_img, lang="eng")
        return text.strip()
    except:
        return ""


# ------------------------------
# GPT fallback：修正 OCR 錯誤
# ------------------------------
from openai import OpenAI

from openai import OpenAI

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
  "date": "...",
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

        reply = resp.choices[0].message["content"]

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


# ------------------------------
# UNet Segmentation + OCR master
# ------------------------------
def extract_invoice_meta(pil_img, checkpoint_path, apikey):
    """
    直接用 GPT-4o-mini 救場，UNet 現在沒用
    """
    meta = {"invoice_no": "", "date": "", "total_amount": ""}
    
    if not apikey:
        st.error("請輸入 OpenAI API Key")
        return meta
        
    # GPT 直接看整張圖
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    prompt = """
請從圖片中辨識台灣電子發票的三個欄位，並以 JSON 格式回覆：

{
  "invoice_no": "...",
  "date": "...",只要年月日，自動轉西元
  "total_amount": "..."前方會有"總計:"幾個字，只要後面的數字
}

只回傳純 JSON，什麼都別多說。
"""
    
    try:
        client = OpenAI(api_key=apikey)
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
        
        reply = resp.choices[0].message.content.strip()
        start = reply.find("{")
        end = reply.rfind("}") + 1
        reply = reply[start:end]
        meta = json.loads(reply)
        
        # 保險：確保欄位存在
        meta = {
            "invoice_no": meta.get("invoice_no", ""),
            "date": meta.get("date", ""),
            "total_amount": meta.get("total_amount", ""),
        }
        
    except Exception as e:
        st.error(f"GPT 辨識失敗：{e}")
    
    return meta

# ============================================================
# Part 3 — QR 全圖偵測（pyzxing + OpenCV fallback）+ TEXT QR 品項解析
# ============================================================

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
    if not items or total_amount <= 0:
        return items
        
    subtotal = sum(it["qty"] * it["price"] for it in items)
    if subtotal <= 0:
        return items

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
    """超寬鬆版 TEXT QR 判斷，永遠不會漏掉任何一顆（包含載具贈品那顆）"""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    
    # 只要包含這些關鍵字，就一定是 TEXT QR（不管多亂）
    keywords = ["**:", "※※", "隨貨發票", "載具", "*********", "加鹽黑松", "點數", "贈送"]
    if any(kw in text for kw in keywords):
        return True
        
    # 或者符合標準格式：有品名:數量:單價結構
    if re.search(r'[^\d\s]{2,}.*?\d+:\d+$', text):
        return True
        
    # 或者長度超過 50（載具碼那顆一定很長）
    if len(text) > 50:
        return True
        
    return False


def detect_invoice_items(pil_img, total_amount):

    # Step1: 掃描 QR
    pzx = decode_qr_pyzxing(pil_img)
    ocv = decode_qr_opencv(pil_img)

    raw_all = pzx + ocv

    # Step2: 過濾出真正 TEXT QR
    text_qrs = [t for t in raw_all if is_real_text_qr(t)]

    text_qrs = list(set(text_qrs))  # 去除重複

    # DEBUG
    # st.write("FILTERED TEXT QRs:", text_qrs)

    final_items = []

    # Step3: 逐段解析
    for t in text_qrs:
        items = parse_text_qr_items(t)
        final_items.extend(items)

    if not final_items:
        return {
            "pyzxing_raw": pzx,
            "opencv_raw": ocv,
            "merged_text_qr": text_qrs
        }, []

    # Step4: 金額調整
    final_items = adjust_items_with_total(final_items, total_amount)

    return {
        "pyzxing_raw": pzx,
        "opencv_raw": ocv,
        "merged_text_qr": text_qrs
    }, final_items


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


# ============================================================
# TAB 1 — 上傳發票
# ============================================================
# ============================================================
# TAB 1 — 深色版 發票上傳頁
# ============================================================
with tab1:
    st.markdown("<h2>📤 上傳並辨識發票</h2>", unsafe_allow_html=True)

    uploaded = st.file_uploader("請選擇發票圖片 (JPG / PNG)", type=["jpg","jpeg","png"])

    checkpoint_path = "unet_epoch30.pth"

    if uploaded:
        col_img, col_info = st.columns([1,1])

        pil_img = Image.open(uploaded).convert("RGB")

        with col_img:
            st.image(pil_img, caption="📸 原始影像", use_container_width=True)

        with col_info:
            with st.spinner("🔍 UNet Segmentation + OCR 辨識中…"):
                meta = extract_invoice_meta(pil_img, checkpoint_path, apikey)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🧾 發票資訊")
            st.write(f"**發票號碼：** <span class='highlight'>{meta.get('invoice_no','')}</span>", unsafe_allow_html=True)
            st.write(f"**日期：** <span class='highlight'>{meta.get('date','')}</span>", unsafe_allow_html=True)
            st.write(f"**總金額：** <span class='highlight'>NT$ {meta.get('total_amount','')}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 整理金額
        try:
            total_amount = float(re.sub(r"[^0-9.]", "", meta.get("total_amount", "0")))
        except:
            total_amount = 0

        # 🔍 QR 全圖掃描
        with st.spinner("📡 QR Code 掃描中…"):
            debug_qr, items = detect_invoice_items(pil_img, total_amount)

        st.markdown("### 📦 TEXT QR 品項")
        if items:
            df_items = pd.DataFrame(items)
            st.dataframe(df_items, use_container_width=True)
        else:
            st.info("📭 未偵測到 TEXT QR 品項")

        # 類別 + 備註
        st.markdown("### 🏷 類別與備註")
        category = st.selectbox("類別 Category", ["餐飲","購物","交通","娛樂","日用品","其他"])
        note = st.text_input("備註 Note")

        # 儲存
        if supabase:
            if st.button("💾 儲存到資料庫", type="primary"):
                invoice_id = save_invoice_main(meta, total_amount, category, note)
                if invoice_id:
                    ok = save_invoice_items(invoice_id, items)
                    if ok:
                        st.success("🎉 發票與品項成功儲存！")
                    else:
                        st.error("❌ 品項儲存失敗")
        else:
            st.warning("❗ Supabase 未連線，無法儲存資料")


# ============================================================
# TAB 2 — 深色專業版 財務儀表板
# ============================================================
with tab2:
    st.markdown("<h2>📊 發票記帳儀表板</h2>", unsafe_allow_html=True)

    if not supabase:
        st.warning("尚未連接 Supabase")
    else:
        with st.spinner("讀取資料中…"):
            invoices = supabase.table("invoices_data").select("*").order("date", desc=True).execute().data
            items = supabase.table("invoice_items").select("*").execute().data

        if not invoices:
            st.info("📭 目前沒有資料")
        else:
            df_inv = pd.DataFrame(invoices)
            df_items = pd.DataFrame(items)

            df_inv["date"] = pd.to_datetime(df_inv["date"], errors="coerce")
            df_inv["year_month"] = df_inv["date"].dt.to_period("M")

            # ========= 顯示 KPI 區塊 =========
            st.markdown("### 💎 本月概要")
            colA, colB, colC = st.columns(3)

            this_month = df_inv["year_month"].astype(str).max()
            df_this_month = df_inv[df_inv["year_month"].astype(str) == this_month]

            with colA:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("📅 本月消費")
                st.markdown(f"<h3 class='highlight'>NT$ {df_this_month['total_amount'].sum():,.0f}</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with colB:
                last_month = sorted(df_inv["year_month"].astype(str).unique())[-2] if len(df_inv) > 1 else this_month
                df_last_month = df_inv[df_inv["year_month"].astype(str) == last_month]

                growth = 0
                if df_last_month["total_amount"].sum() > 0:
                    growth = ((df_this_month["total_amount"].sum() - df_last_month["total_amount"].sum())
                            / df_last_month["total_amount"].sum()) * 100

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("📈 月成長率")
                st.markdown(f"<h3 class='highlight'>{growth:.1f}%</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with colC:
                top_cat = df_this_month.groupby("category")["total_amount"].sum().reset_index()
                top_cat = top_cat.sort_values("total_amount", ascending=False)
                top_name = top_cat.iloc[0]["category"] if len(top_cat) > 0 else "無資料"

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("🏷 本月最大支出類別")
                st.markdown(f"<h3 class='highlight'>{top_name}</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ========= 每月折線圖 =========
            st.markdown("### 📉 每月支出趨勢")
            monthly = df_inv.groupby("year_month")["total_amount"].sum().reset_index()
            monthly["year_month"] = monthly["year_month"].astype(str)

            st.line_chart(monthly, x="year_month", y="total_amount")

            # ========= 圓餅圖 =========
            st.markdown("### 🥧 類別支出比例")
            cat_sum = df_inv.groupby("category")["total_amount"].sum().reset_index()
            fig = px.pie(cat_sum, names="category", values="total_amount", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

            # ========= 月份選擇 =========
            st.markdown("### 🔍 查看特定月份")
            month_selected = st.selectbox("選擇月份", monthly["year_month"].unique())

            df_month = df_inv[df_inv["year_month"] == month_selected]
            st.dataframe(df_month, use_container_width=True)

            # ========= 發票選擇 =========
            st.markdown("### 📄 選擇發票查看品項")
            invoice_id_selected = st.selectbox("選擇發票 ID", df_month["id"])

            df_selected_items = df_items[df_items["invoice_id"] == invoice_id_selected]
            st.dataframe(df_selected_items, use_container_width=True)

            # ========= 刪除發票 =========
            st.markdown("### 🗑 刪除此發票")
            if st.button("❗ 刪除（含所有品項）"):
                supabase.table("invoice_items").delete().eq("invoice_id", invoice_id_selected).execute()
                supabase.table("invoices_data").delete().eq("id", invoice_id_selected).execute()
                st.success("已刪除成功！請重新整理頁面")
