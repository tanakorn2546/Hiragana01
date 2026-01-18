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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS Styling (Refined for Clarity & Alignment) ---
def local_css():
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600;700&family=Sawarabi+Mincho&display=swap');

        /* --- Global Font Settings --- */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
            color: #1a1a2e; /* Dark Blue-Black for high contrast */
        }

        /* --- Background --- */
        .stApp {
            background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fad0c4, #a18cd1);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- Glass Card --- */
        .glass-card {
            background: rgba(255, 255, 255, 0.85); /* เพิ่มความทึบแสงให้อ่านตัวหนังสือชัดขึ้น */
            backdrop-filter: blur(25px);
            border-radius: 30px;
            border: 2px solid rgba(255, 255, 255, 1);
            padding: 40px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        /* --- Headers --- */
        .hero-title {
            font-family: 'Sawarabi Mincho', serif;
            font-size: 4rem;
            font-weight: 800; /* หนาพิเศษ */
            background: linear-gradient(45deg, #FF416C, #FF4B2B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 0px 5px 15px rgba(255, 65, 108, 0.3);
            margin-bottom: 0px;
            letter-spacing: -1px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: #444;
            font-weight: 500;
            margin-bottom: 40px;
            letter-spacing: 0.5px;
        }

        /* --- Buttons (จัดระเบียบปุ่ม) --- */
        .stButton button {
            border-radius: 12px !important;
            font-family: 'Prompt', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important; /* เพิ่มขนาดตัวอักษรในปุ่ม */
            padding: 0.75rem 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            border: none !important;
        }

        /* ปุ่ม Primary (Analyze) */
        div[data-testid="stVerticalBlock"] .stButton button[kind="primary"] {
            background: linear-gradient(90deg, #FF416C, #FF4B2B) !important;
            color: white !important;
            font-size: 1.2rem !important; /* ปุ่มหลักตัวใหญ่กว่า */
        }
        div[data-testid="stVerticalBlock"] .stButton button[kind="primary"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(255, 75, 43, 0.4) !important;
        }

        /* ปุ่ม Secondary (Navigation) */
        div[data-testid="stVerticalBlock"] .stButton button[kind="secondary"] {
            background: #ffffff !important;
            color: #333 !important;
            border: 2px solid #eee !important;
        }
        div[data-testid="stVerticalBlock"] .stButton button[kind="secondary"]:hover {
            border-color: #FF4B2B !important;
            color: #FF4B2B !important;
            background: #fff5f5 !important;
        }

        /* --- Result Styling --- */
        .result-box {
            background: white;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            border: 2px solid #ffebee;
            box-shadow: inset 0 0 20px rgba(255, 235, 238, 0.5);
        }
        .big-char {
            font-size: 5rem;
            font-weight: 700;
            color: #d32f2f;
            margin: 0;
            line-height: 1.2;
        }
        .label-text {
            font-size: 1rem;
            color: #666;
            font-weight: 500;
            margin-bottom: 5px;
        }

        /* --- Image Styling --- */
        div[data-testid="stImage"] img {
            border-radius: 15px;
            border: 4px solid white;
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database & Model Functions ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app" 
    )

def get_work_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        base_sql = "SELECT id, char_code, ai_result FROM progress WHERE image_data IS NOT NULL"
        if "ยังไม่ตรวจ" in filter_mode:
            sql = f"{base_sql} AND ai_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = f"{base_sql} AND ai_result IS NOT NULL ORDER BY id DESC"
        else:
            sql = f"{base_sql} ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        return []

def get_work_data(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT image_data, ai_result, ai_confidence, char_code FROM progress WHERE id = %s", (work_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

def update_database(work_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), work_id))
        conn.commit()
        conn.close()
        return True
    except: return False

def get_stats():
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        total, checked = cursor.fetchone()
        conn.close()
        return total, checked
    except: return 0, 0

@st.cache_resource
def load_model():
    file_id = '1Yw1YCu35oxQT5jpB0xqouZMD-MH2EGZO' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path): model_name = local_path
        else:
            try: gdown.download(url, model_name, quiet=False)
            except: return None
    try: return tf.keras.models.load_model(model_name, compile=False)
    except: return None

def load_class_names():
    return [
        'a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko',
        'sa', 'shi', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to',
        'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
        'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo',
        'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo', 'n'
    ]

def import_and_predict(image_data, model):
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    if image.mode != "RGB": image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. UI Layout & Logic ---
model = load_model()
class_names = load_class_names()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    total_w, checked_w = get_stats()
    st.markdown(f"""
    <div style="background:white; padding:20px; border-radius:15px; box-shadow:0 4px 10px rgba(0,0,0,0.05);">
        <h2 style="margin:0; color:#FF4B2B;">{total_w}</h2>
        <p style="margin:0; color:#555; font-size:0.9rem;">Total Images</p>
    </div>
    <div style="height:15px;"></div>
    <div style="background:white; padding:20px; border-radius:15px; box-shadow:0 4px 10px rgba(0,0,0,0.05);">
        <h2 style="margin:0; color:#23a6d5;">{checked_w}</h2>
        <p style="margin:0; color:#555; font-size:0.9rem;">Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

# Main Header
st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent Handwriting Recognition System</div>', unsafe_allow_html=True)

# Filter Bar (Center Aligned)
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([1, 6, 1]) # จัดให้ Radio อยู่ตรงกลางสวยๆ
with c2:
    if target_work_id:
        st.info(f"🔍 Focused on ID: {target_work_id}")
        filter_option = "ทั้งหมด (All)"
    else:
        filter_option = st.radio(
            "Select View Mode",
            ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"],
            horizontal=True,
            label_visibility="collapsed"
        )

# Logic
work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    # Progress Bar
    progress = (st.session_state.current_index + 1) / len(id_list)
    st.progress(progress)
    
    # --- Glass Card Layout ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Card Header
    st.markdown(f"<div style='text-align:center; font-weight:600; color:#555; margin-bottom:20px; font-size:1.1rem;'>Image ID: {current_id} ({st.session_state.current_index + 1}/{len(id_list)})</div>", unsafe_allow_html=True)

    data_row = get_work_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        try: image = Image.open(io.BytesIO(blob_data))
        except: image = None

        if image:
            col_img, col_res = st.columns([1, 1], gap="large")
            
            # Left: Image
            with col_img:
                st.markdown(f"<div class='label-text'>📝 โจทย์ตัวอักษร: <b style='color:#1a1a2e; font-size:1.2rem;'>{true_label}</b></div>", unsafe_allow_html=True)
                st.image(image, use_column_width=True)
            
            # Right: Result & Actions
            with col_res:
                st.markdown("<div class='label-text'>🤖 ผลการวิเคราะห์ (AI Analysis)</div>", unsafe_allow_html=True)
                
                if saved_result:
                    # Result Box
                    char_part = saved_result.split(' ')[0]
                    romaji_part = saved_result.split(' ')[1] if len(saved_result.split(' ')) > 1 else ''
                    
                    st.markdown(f"""
                        <div class="result-box">
                            <div class="big-char">{char_part}</div>
                            <div style="font-size:1.5rem; font-weight:600; color:#333; margin-top:5px;">{romaji_part}</div>
                            <div style="margin-top:15px; background:#e8f5e9; color:#2e7d32; padding:5px 15px; border-radius:20px; display:inline-block; font-weight:600;">
                                Confidence: {saved_conf:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Spacer
                    if st.button("🔄 ตรวจสอบใหม่ (Re-Check)", type="secondary", use_container_width=True):
                        update_database(current_id, None, 0)
                        st.rerun()
                        
                else:
                    # Pending Box
                    st.markdown("""
                        <div class="result-box" style="border: 2px dashed #ccc; background:#f9f9f9; padding: 40px;">
                            <h1 style="color:#ccc; font-size:4rem; margin:0;">⏳</h1>
                            <p style="color:#888; margin-top:10px;">รอการประมวลผล...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Spacer
                    # ใช้ use_container_width=True เพื่อให้ปุ่มเต็มความกว้าง
                    if st.button("✨ วิเคราะห์ผล (Analyze Now)", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("AI กำลังคิด..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    # Mapping Logic
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
                                    update_database(current_id, final_res, conf)
                                    st.success(f"เสร็จสิ้น! อ่านว่า: {final_res}")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Model not found")

    st.markdown('</div>', unsafe_allow_html=True) # End Glass Card

    # --- Navigation Bar (Aligned Beautifully) ---
    # ใช้ Columns แบบ 5 ช่องเพื่อจัดปุ่มให้อยู่กึ่งกลางและมีระยะห่างที่พอดี
    c_nav1, c_nav2, c_nav3, c_nav4, c_nav5 = st.columns([1, 1, 0.2, 1, 1])
    
    with c_nav2: # ปุ่มซ้าย (Previous)
        if st.session_state.current_index > 0:
            if st.button("⬅️ ก่อนหน้า (Prev)", use_container_width=True):
                st.session_state.current_index -= 1
                st.rerun()
        else:
            # ใส่ปุ่มหลอกๆ เพื่อรักษา layout ไม่ให้โล่ง (optional)
            st.write("") 

    with c_nav4: # ปุ่มขวา (Next)
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป (Next) ➡️", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()
        else:
             if st.button("⏮ เริ่มต้นใหม่", use_container_width=True):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.markdown("""
        <div class="glass-card" style="text-align:center; padding:60px;">
            <h1 style="font-size:80px; margin:0;">📭</h1>
            <h3 style="color:#555;">ไม่พบข้อมูล</h3>
            <p style="color:#888;">กรุณาเลือกหมวดหมู่ใหม่ หรือรอข้อมูลจากนักเรียน</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
teacher_dashboard_url = "https://www.cedubru.com/hiragana/teacher.php?view_student=7" 
st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; padding-bottom: 30px;">
        <a href="{teacher_dashboard_url}" target="_self" class="home-btn">
            🏠 กลับสู่หน้าหลัก (Dashboard)
        </a>
        <p style="margin-top:20px; color:#1a1a2e; font-size:0.8rem; opacity:0.6;">
            Hiragana Image Classification System V.3.1 Ultimate | Design by Hiragana Sensei Team
        </p>
    </div>
""", unsafe_allow_html=True)