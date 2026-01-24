import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import gdown

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Hiragana Sensei AI",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded" # เปิด Sidebar ไว้ให้เห็นสถานะ
)

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

# --- 3. Database Functions ---
# ⚠️ สำคัญ: ตรวจสอบ Host ว่าตรงกับที่ PHP คุณใช้หรือไม่
DB_CONFIG = {
    "host": "www.cedubru.com",  # ถ้าเทสในเครื่องแล้วมี XAMPP ให้แก้เป็น "localhost"
    "user": "cedubruc_hiragana_app",
    "password": "7gZ8gDJyufzJyzELZkdg",
    "database": "cedubruc_hiragana_app"
}

def init_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_work_data(target_id, table_name="progress"):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ใช้ Dynamic Query (ระวัง SQL Injection แต่ในที่นี้เราคุมตัวแปร table_name เอง)
        sql = f"SELECT image_data, ai_result, ai_confidence, char_code FROM {table_name} WHERE id = %s"
        cursor.execute(sql, (target_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        return None

def update_database(target_id, table_name, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        sql = f"UPDATE {table_name} SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), target_id))
        conn.commit()
        conn.close()
        return True
    except: return False

def get_stats():
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        return cursor.fetchone()
    except: return 0, 0

# --- 4. Model Loading ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    model_name = 'hiragana_model_best.h5'
    file_id = '1iPYeqEv8uYBvbcgb90pjHX-0cJBakOrI' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path): model_name = local_path
        else:
            try:
                gdown.download(url, model_name, quiet=False)
            except Exception as e:
                st.error(f"❌ Load Error: {e}")
                return None
    
    try:
        return tf.keras.models.load_model(
            model_name, compile=False,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D} 
        )
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
        return None

def load_class_names():
    return [
        'a', 'chi', 'e', 'fu', 'ha', 'he', 'hi', 'ho', 'i', 'ka', 'ke', 'ki', 'ko', 'ku', 
        'ma', 'me', 'mi', 'mo', 'mu', 'n', 'na', 'ne', 'ni', 'no', 'nu', 'o', 
        'ra', 're', 'ri', 'ro', 'ru', 'sa', 'se', 'shi', 'so', 'su', 
        'ta', 'te', 'to', 'tsu', 'u', 'wa', 'wo', 'ya', 'yo', 'yu'
    ]

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    if image.mode != "L": image = image.convert("L")
    image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# --- 6. Main Logic ---
model = load_model()
class_names = load_class_names()

# Sidebar Debug Info
with st.sidebar:
    st.markdown("### 🔌 Connection Status")
    st.code(f"Host: {DB_CONFIG['host']}\nDB: {DB_CONFIG['database']}")
    st.markdown("---")
    st.markdown("### 🌸 สรุปข้อมูล")
    total_w, checked_w = get_stats()
    st.info(f"ภาพ (Practice): {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือด้วย MobileNetV2</div>', unsafe_allow_html=True)

# รับค่า URL Params
query_params = st.query_params
req_work_id = query_params.get("work_id", None)
req_quiz_id = query_params.get("quiz_id", None)

current_id = None
active_table = "progress"
mode_color = "#D72638"
mode_name = "Practice"

if req_quiz_id:
    current_id = req_quiz_id
    active_table = "quiz_submissions"
    mode_color = "#7c3aed"
    mode_name = "Quiz"
    st.markdown(f"""
    <div style="background:#f3e8ff; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold;">
        📝 Mode: ตรวจแบบทดสอบ (ID: {current_id})
    </div>
    <style>.stApp {{ background: linear-gradient(180deg, #f3e8ff 0%, #fff 60%, #fff 100%) !important; }}</style>
    """, unsafe_allow_html=True)

elif req_work_id:
    current_id = req_work_id
    active_table = "progress"
    st.markdown(f"""
    <div style="background:#ffebee; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold;">
        ✍️ Mode: ตรวจแบบฝึกหัด (ID: {current_id})
    </div>""", unsafe_allow_html=True)

if current_id:
    # ดึงข้อมูลตาม ID และ ตารางที่ระบุ
    data_row = get_work_data(current_id, active_table)
    
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
                if saved_result:
                    parts = saved_result.split(' ')
                    char_part = parts[0]
                    romaji_part = parts[1] if len(parts) > 1 else ''
                    st.markdown(f"""
                    <div class="result-card" style="border-top-color: {mode_color};">
                        <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                        <div class="big-char" style="color:{mode_color};">{char_part}</div>
                        <div style="color:green; font-weight:bold;">{saved_conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 ตรวจใหม่", type="secondary", use_container_width=True):
                        update_database(current_id, active_table, None, 0)
                        st.rerun()
                else:
                    st.markdown(f"""
                    <div class="result-card" style="border: 2px dashed #ddd; background:#fffaf0; border-top-color:{mode_color};">
                        <h1 style="color:{mode_color}; opacity:0.5;">⏳</h1><p style="color:#888;">รอผล...</p>
                    </div>""", unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("✨ วิเคราะห์", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("AI กำลังคิด..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    
                                    hiragana_map = {
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
                                    final_res = hiragana_map.get(res_code, res_code)
                                    update_database(current_id, active_table, final_res, conf)
                                    time.sleep(0.3)
                                    st.rerun()
                                except Exception as e: st.error(f"Error: {e}")
                        else: st.error("ไม่พบโมเดล")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # กรณีไม่พบข้อมูล ให้แสดงข้อความแจ้งเตือนที่ชัดเจน
        st.error(f"""
        ❌ **ไม่พบข้อมูล ID: {current_id} ในตาราง {active_table}**
        
        **สาเหตุที่เป็นไปได้:**
        1. ฐานข้อมูลที่คุณบันทึก (Localhost) ไม่ตรงกับที่ AI อ่าน (Server: {DB_CONFIG['host']})
        2. คุณลบข้อมูลนี้ไปแล้ว
        3. มีปัญหาการเชื่อมต่อ
        """)
        st.info("💡 คำแนะนำ: หากคุณทดสอบในเครื่อง ให้รันไฟล์นี้ด้วยคำสั่ง `streamlit run app.py` ในเครื่องของคุณเอง")

else:
    st.info("👋 ยินดีต้อนรับ! กรุณากดปุ่ม 'ตรวจด้วย AI' จากหน้าเว็บหลัก")

# Footer
st.markdown("""<div style="text-align: center; margin-top: 50px; position:relative; z-index:20;">
<a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">🏠 กลับสู่หน้าหลัก</a></div>""", unsafe_allow_html=True)