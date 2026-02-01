import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import cv2
import json
import gdown

# --- 1. Config & Setup ---
st.set_page_config(
    page_title="Hiragana Sensei AI",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# ⚠️⚠️⚠️ จุดสำคัญที่ต้องแก้ไข ⚠️⚠️⚠️
# 1. เอา ID ของไฟล์ .h5 ตัวใหม่ (v6) จาก Google Drive มาใส่ตรงนี้
GOOGLE_DRIVE_FILE_ID = '1sM-ZFhECDFU-hculh2FKbfQszMpRE2_D' 

# 2. ชื่อไฟล์โมเดล (เปลี่ยนเป็น v6 เพื่อให้ระบบรู้ว่าเป็นตัวใหม่)
MODEL_FILENAME = 'hiragana_mobilenet_v2_final_v6.h5'

# 3. ชื่อไฟล์ JSON (ต้องตรงกับที่เทรนมา)
JSON_FILENAME = 'class_indices.json' 
# ---------------------------------------------------------

CONFIDENCE_THRESHOLD = 40.0                 # เกณฑ์ความมั่นใจ

# --- เพิ่มเมนู Sidebar สำหรับล้างค่า (แก้ปัญหาโมเดลค้าง) ---
with st.sidebar:
    st.header("🔧 เครื่องมือแก้ปัญหา")
    if st.button("🗑️ ล้าง Cache โมเดล (Force Reset)"):
        st.cache_resource.clear()
        if os.path.exists(MODEL_FILENAME):
            try:
                os.remove(MODEL_FILENAME) # ลบไฟล์ทิ้งเพื่อโหลดใหม่
            except:
                pass
        st.success("✅ ล้าง Cache แล้ว! กรุณากด Rerun หรือรีเฟรชหน้าเว็บ")
        time.sleep(1)
        st.rerun()

# --- 2. CSS Styling ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Zen+Maru+Gothic:wght@700&display=swap');
        :root { --japan-red: #D72638; --quiz-purple: #7c3aed; }
        html, body, [class*="css"] { font-family: 'Prompt', sans-serif !important; }
        .stApp {
            background: linear-gradient(180deg, #d4fcff 0%, #fff 60%, #fff 100%);
            background-attachment: fixed;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px);
            border-radius: 20px; border: 2px solid white; padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        .result-card {
            background: white; border-radius: 15px; padding: 20px; text-align: center;
            border-top: 5px solid var(--japan-red); box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .big-char { font-size: 5rem; color: var(--japan-red); font-weight: bold; line-height: 1; }
        .hero-title {
            font-family: 'Zen Maru Gothic', sans-serif; font-size: 3.5rem; color: var(--japan-red);
            text-align: center; text-shadow: 2px 2px 0px white; margin-bottom: 0;
        }
        .hero-subtitle { text-align: center; color: #555; margin-bottom: 30px; }
        .stButton button { border-radius: 12px !important; font-weight: 600 !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Configuration ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

def update_database(target_id, table_name, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        col_res = "ai_result"
        col_conf = "ai_confidence"
        
        sql = f"UPDATE {table_name} SET {col_res} = %s, {col_conf} = %s WHERE id = %s"
        # กรณีล้างค่า (Re-check) ให้ส่ง NULL หรือค่าว่างไป
        if result is None:
            cursor.execute(sql, (None, 0, target_id))
        else:
            cursor.execute(sql, (result, float(confidence), target_id))
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Database Update Error: {e}")
        return False

def get_work_data(target_id, table_name="progress"):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        if table_name == "progress":
            sql = "SELECT image_data, ai_result, ai_confidence, char_code FROM progress WHERE id = %s"
        else:
            sql = "SELECT image_data, ai_result, ai_confidence, char_label FROM quiz_submissions WHERE id = %s"
            
        cursor.execute(sql, (target_id,))
        data = cursor.fetchone()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ Data Fetch Error: {e}")
        return None

# --- 4. Model Loading & Preprocessing ---

# Custom Layer เพื่อแก้ปัญหา Version Compatibility (เผื่อไว้)
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_ai_model():
    # 1. Download from Google Drive if needed
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    
    # ถ้ายังไม่มีไฟล์ หรือไฟล์มีขนาด 0 (เสีย) ให้โหลดใหม่
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) == 0:
        try:
            with st.spinner(f"☁️ Downloading Model... ({MODEL_FILENAME})"):
                gdown.download(url, MODEL_FILENAME, quiet=False)
                st.success("✅ Download Success!")
        except Exception as e:
            st.error(f"❌ Download Error: {e}")
            return None

    # 2. Load Model
    try:
        return tf.keras.models.load_model(
            MODEL_FILENAME, 
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
            compile=False 
        )
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        # ถ้าโหลดไม่ได้ ให้ลองลบไฟล์แล้วโหลดใหม่ (เผื่อไฟล์เสีย)
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)
        return None

@st.cache_data
def load_class_mapping():
    if not os.path.exists(JSON_FILENAME):
        st.warning(f"⚠️ ไม่พบไฟล์ {JSON_FILENAME} ระบบจะใช้ Default Mapping")
        return None
    
    try:
        with open(JSON_FILENAME, 'r') as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"❌ JSON Load Error: {e}")
        return None

def preprocess_image(image_data):
    # 1. Resize & Grayscale
    img = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    if img.mode != "L":
        img = img.convert("L")
    img_array = np.array(img)

    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 19, 5)
    
    # 3. Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 4. Convert to RGB & Preprocess
    img_back = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_back.astype(np.float32))
    
    return np.expand_dims(img_preprocessed, axis=0)

def get_display_text(class_name):
    mapping = {
        'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)',
        'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)',
        'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)',
        'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)',
        'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)',
        'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)',
        'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)',
        'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)',
        'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)',
        'wa': 'わ (wa)', 'wo': 'を (wo)', 'n': 'ん (n)'
    }
    return mapping.get(class_name, class_name)

# --- 5. Main Application Logic ---
model = load_ai_model()
idx_to_label = load_class_mapping()

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">Model Version: {MODEL_FILENAME}</div>', unsafe_allow_html=True)

query_params = st.query_params
req_work_id = query_params.get("work_id", None)
req_quiz_id = query_params.get("quiz_id", None)

target_id = req_quiz_id if req_quiz_id else req_work_id
active_table = "quiz_submissions" if req_quiz_id else "progress"
mode_color = "#7c3aed" if req_quiz_id else "#D72638"
mode_text = "แบบทดสอบ (Quiz)" if req_quiz_id else "แบบฝึกหัด (Practice)"

if target_id:
    # Header Bar
    st.markdown(f"""
    <div style="background:{mode_color}15; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold;">
        📝 กำลังตรวจ: {mode_text} (ID: {target_id})
    </div>
    """, unsafe_allow_html=True)
    if req_quiz_id:
        st.markdown("<style>.stApp { background: linear-gradient(180deg, #f3e8ff 0%, #fff 60%, #fff 100%) !important; }</style>", unsafe_allow_html=True)

    # Fetch Data
    data_row = get_work_data(target_id, active_table)
    
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        try: image = Image.open(io.BytesIO(blob_data))
        except: image = None

        if image:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_img, col_res = st.columns([1, 1.2], gap="large")
            
            with col_img:
                st.markdown(f"**โจทย์:** `{true_label}`")
                st.image(image, use_container_width=True)
            
            with col_res:
                st.markdown("**ผลการตรวจ**")
                
                # Check Logic: ถ้ามีค่า saved_result แสดงว่าตรวจแล้ว ให้โชว์ของเดิม
                if saved_result and saved_result != "None": 
                    parts = saved_result.split(' ')
                    char_part = parts[0]
                    romaji_part = parts[1] if len(parts) > 1 else ''
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-top-color:{mode_color};">
                        <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                        <div class="big-char" style="color:{mode_color};">{char_part}</div>
                        <div style="color:green; font-weight:bold;">{saved_conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    st.write("")
                    
                    # ปุ่มตรวจใหม่: สำคัญมาก! ต้องกดปุ่มนี้ถึงจะล้างค่าเก่าแล้วตรวจด้วยโมเดลใหม่
                    if st.button("🔄 ตรวจใหม่ (Re-Check)", use_container_width=True):
                        update_database(target_id, active_table, None, 0) # ล้างค่าใน DB เป็น NULL
                        st.cache_data.clear() # เคลียร์ cache ข้อมูล
                        st.rerun() # รีโหลดหน้าเว็บ
                else:
                    # กรณีรอยังไม่ตรวจ หรือกด Re-Check มาแล้ว
                    st.markdown(f"""
                    <div class="result-card" style="border: 2px dashed #ddd; background:#fffaf0;">
                        <h1 style="color:{mode_color}; opacity:0.5;">⏳</h1>
                    </div>""", unsafe_allow_html=True)
                    st.write("")
                    
                    if st.button("✨ วิเคราะห์", type="primary", use_container_width=True):
                        if model and idx_to_label:
                            with st.spinner("AI กำลังวิเคราะห์ลายเส้น..."):
                                try:
                                    # 1. Preprocess
                                    input_tensor = preprocess_image(image)
                                    
                                    # 2. Predict
                                    preds = model.predict(input_tensor)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
                                    # 3. Map Result
                                    pred_code = idx_to_label.get(idx, "Unknown")
                                    final_res = get_display_text(pred_code)
                                    
                                    # 4. Confidence Check
                                    if conf < CONFIDENCE_THRESHOLD:
                                        final_res = "❓ Unknown (เขียนใหม่)"
                                    
                                    # 5. Save & Refresh
                                    if update_database(target_id, active_table, final_res, conf):
                                        time.sleep(0.3)
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Prediction Error: {e}")
                        else:
                            st.error("⚠️ ไม่พบ Model หรือไฟล์ Mapping JSON กรุณาตรวจสอบ")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ ไม่พบข้อมูลภาพในฐานข้อมูล")
else:
    st.info("👋 กรุณาเลือกแบบฝึกหัดจากหน้า Teacher Dashboard")
    st.markdown("""<div style="text-align: center; margin-top: 50px;"><a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">🏠 กลับสู่หน้าหลัก</a></div>""", unsafe_allow_html=True)