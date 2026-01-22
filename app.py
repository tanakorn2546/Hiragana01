import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import gdown
import cv2 

# --- 1. Page Configuration ---
st.set_page_config(page_title="Hiragana Sensei AI", page_icon="🌸", layout="centered", initial_sidebar_state="collapsed")

# --- 2. CSS Styling ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Zen+Maru+Gothic:wght@700&display=swap');
        :root { --japan-red: #D72638; }
        html, body, [class*="css"] { font-family: 'Prompt', sans-serif !important; }
        .stApp { background: linear-gradient(180deg, #d4fcff 0%, #fff 60%, #fff 100%); background-attachment: fixed; }
        /* Elements */
        .glass-card { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(15px); border-radius: 20px; border: 2px solid white; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .result-card { background: white; border-radius: 15px; padding: 20px; text-align: center; border-top: 5px solid var(--japan-red); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        .big-char { font-size: 5rem; color: var(--japan-red); font-weight: bold; line-height: 1; }
        .stButton button { border-radius: 12px !important; font-weight: 600 !important; border: none !important; }
        div[data-testid="stVerticalBlock"] .stButton button { background: var(--japan-red) !important; color: white !important; }
        div[data-testid="stHorizontalBlock"] .stButton button { background: white !important; color: var(--japan-red) !important; border: 2px solid var(--japan-red) !important; }
        .hero-title { font-family: 'Zen Maru Gothic', sans-serif; font-size: 3.5rem; color: var(--japan-red); text-align: center; margin-bottom: 0; }
        .hero-subtitle { text-align: center; color: #555; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)
local_css()

# --- 3. Database Functions ---
def init_connection():
    return mysql.connector.connect(host="www.cedubru.com", user="cedubruc_hiragana_app", password="7gZ8gDJyufzJyzELZkdg", database="cedubruc_hiragana_app")

def get_work_list(filter_mode):
    try:
        conn = init_connection(); cursor = conn.cursor()
        base_sql = "SELECT id, char_code, ai_result FROM progress WHERE image_data IS NOT NULL"
        sql = f"{base_sql} AND ai_result IS NULL ORDER BY id ASC" if "ยังไม่ตรวจ" in filter_mode else (f"{base_sql} AND ai_result IS NOT NULL ORDER BY id DESC" if "ตรวจแล้ว" in filter_mode else f"{base_sql} ORDER BY id DESC")
        cursor.execute(sql); data = cursor.fetchall(); conn.close(); return data
    except: return []

def get_work_data(work_id):
    try:
        conn = init_connection(); cursor = conn.cursor()
        cursor.execute("SELECT image_data, ai_result, ai_confidence, char_code FROM progress WHERE id = %s", (work_id,))
        data = cursor.fetchone(); conn.close(); return data 
    except: return None

def update_database(work_id, result, confidence):
    try:
        conn = init_connection(); cursor = conn.cursor()
        cursor.execute("UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s", (result, float(confidence), work_id))
        conn.commit(); conn.close(); return True
    except: return False

def get_stats():
    try:
        conn = init_connection(); cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        total, checked = cursor.fetchone(); conn.close(); return total, checked
    except: return 0, 0

# --- 4. Model Loading (Updated with GDrive ID) ---
@st.cache_resource
def load_model():
    # ชื่อไฟล์ที่ต้องการ (เวอร์ชั่นใหม่)
    model_name = 'hiragana_cnn_v3.h5'
    
    # ------------------------------------------------------------------
    # 👇 ใส่ ID จาก Google Drive ของไฟล์ v3 ตรงนี้ครับ
    # ------------------------------------------------------------------
    file_id = '11YqKURFNuUZH0h1lnkn4C8Cc0MFcMgJh'  # <--- แก้ไขตรงนี้
    # ------------------------------------------------------------------

    # กำหนดตำแหน่งไฟล์ (Local)
    local_path = os.path.join('saved_models', model_name)
    
    # ตรวจสอบว่ามีไฟล์หรือไม่ ถ้าไม่มีให้โหลด
    if not os.path.exists(model_name) and not os.path.exists(local_path):
        # สร้างโฟลเดอร์ saved_models ถ้ายังไม่มี
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner(f'กำลังดาวน์โหลดโมเดล {model_name} ...'):
                gdown.download(url, local_path, quiet=False)
        except Exception as e:
            st.error(f"❌ ดาวน์โหลดไม่สำเร็จ: {e}")
            return None

    # เลือก path ที่ถูกต้องเพื่อโหลด
    final_path = local_path if os.path.exists(local_path) else model_name
    
    try: 
        return tf.keras.models.load_model(final_path, compile=False)
    except Exception as e: 
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return None

def load_class_names():
    return ['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho', 'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo', 'n']

# --- 🟢 Preprocessing Logic (Corrected) ---
def import_and_predict(image_data, model):
    # 1. Convert to Grayscale
    image = ImageOps.exif_transpose(image_data)
    img_gray = np.array(image.convert("L"))

    # 2. Smart Invert (บังคับให้เป็น ตัวขาว พื้นดำ)
    # ถ้าพื้นหลังสว่าง (ค่าเฉลี่ย > 127) -> กลับสีให้พื้นหลังดำ
    if np.mean(img_gray) > 127:
        img_gray = cv2.bitwise_not(img_gray)
    
    # 3. Thresholding (ตัด Noise ให้เหลือแค่ ขาวสุด กับ ดำสุด)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Dilation (ทำเส้นให้หนาขึ้น) - สำคัญสำหรับปากกาลูกลื่น
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)

    # 5. Crop & Center (ตัดขอบ)
    coords = cv2.findNonZero(img_dilated)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 10
        img_cropped = img_dilated[max(0, y-pad):min(img_dilated.shape[0], y+h+pad), 
                                  max(0, x-pad):min(img_dilated.shape[1], x+pad)]
    else:
        img_cropped = img_dilated

    # 6. Resize to 64x64
    final_pil = Image.fromarray(img_cropped)
    final_img = ImageOps.fit(final_pil, (64, 64), Image.Resampling.LANCZOS)
    
    # --- Debug: โชว์สิ่งที่ AI เห็น (สำคัญมาก) ---
    with st.expander("👁️ สิ่งที่ AI เห็น (ต้องเป็นตัวขาว พื้นดำ)", expanded=True):
        st.image(final_img, width=150, caption="Input sent to Model")
    
    # 7. Normalize (0-1)
    img_array = np.asarray(final_img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    return model.predict(img_array)

# --- UI Logic ---
model = load_model()
class_names = load_class_names()

with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือภาษาญี่ปุ่นอัจฉริยะ (v3 Fix)</div>', unsafe_allow_html=True)

query_params = st.query_params
target_work_id = query_params.get("work_id", None)
c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    filter_option = "ทั้งหมด (All)" if target_work_id else st.selectbox("เลือกโหมด", ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"], label_visibility="collapsed")

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state: st.session_state.current_index = 0
    
    if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0
    elif st.session_state.current_index < 0: st.session_state.current_index = len(id_list) - 1

    current_id = id_list[st.session_state.current_index]
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.caption(f"ID: {current_id} | {st.session_state.current_index + 1}/{len(id_list)}")

    data_row = get_work_data(current_id)
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        try: image = Image.open(io.BytesIO(blob_data))
        except: image = None

        if image:
            col_img, col_res = st.columns([1, 1.2], gap="large")
            with col_img:
                st.markdown(f"**โจทย์:** `{true_label}`")
                st.image(image, use_container_width=True)
            with col_res:
                st.markdown("**ผลการตรวจ**")
                if saved_result:
                    parts = saved_result.split(' '); char_part = parts[0]; romaji_part = parts[1] if len(parts) > 1 else ''
                    st.markdown(f"""<div class="result-card"><div style="font-size:1.2rem; color:#555;">{romaji_part}</div><div class="big-char">{char_part}</div><div style="color:green; font-weight:bold;">มั่นใจ: {saved_conf:.1f}%</div></div>""", unsafe_allow_html=True)
                    st.write("")
                    if st.button("🔄 ตรวจใหม่", key=f"re_{current_id}", use_container_width=True): update_database(current_id, None, 0); st.rerun()
                else:
                    st.markdown("""<div class="result-card" style="border: 2px dashed #ffcdd2; background:#fffaf0;"><h1 style="color:#ef5350; opacity:0.5;">⏳</h1><p style="color:#888;">รอการตรวจ...</p></div>""", unsafe_allow_html=True)
                    st.write("")
                    if st.button("✨ วิเคราะห์ทันที", key=f"an_{current_id}", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("กำลังวิเคราะห์..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds); conf = np.max(preds) * 100
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    hiragana_map = {'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)', 'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)', 'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)', 'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)', 'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)', 'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)', 'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)', 'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)', 'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)', 'wa': 'わ (wa)', 'wo': 'を (wo)', 'n': 'ん (n)'}
                                    final_res = hiragana_map.get(res_code, res_code)
                                    update_database(current_id, final_res, conf)
                                    time.sleep(0.3); st.rerun()
                                except Exception as e: st.error(f"Error: {e}")
                        else: st.error("Model Error")
    st.markdown('</div>', unsafe_allow_html=True)
    c_prev, c_space, c_next = st.columns([1, 0.2, 1])
    with c_prev:
        if st.button("⬅️ ก่อนหน้า", use_container_width=True): st.session_state.current_index -= 1; st.rerun()
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ➡️", use_container_width=True): st.session_state.current_index += 1; st.rerun()
        else:
            if st.button("⏮ เริ่มใหม่", use_container_width=True): st.session_state.current_index = 0; st.rerun()
else: st.info("ไม่พบข้อมูล")

st.markdown("""<div style="text-align: center; margin-top: 50px; position:relative; z-index:20;"><a href="https://www.cedubru.com/hiragana/teacher.php?view_student=7" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">🏠 กลับสู่หน้าหลัก</a></div>""", unsafe_allow_html=True)