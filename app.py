import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import gdown

# --- 1. Page Configuration (ตั้งค่าหน้าเว็บ) ---
st.set_page_config(
    page_title="Hiragana Sensei AI",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Ultra-Modern CSS (ส่วนตกแต่งระดับสูง) ---
def local_css():
    st.markdown("""
    <style>
        /* Import Font: Prompt (Thai) & Sawarabi Mincho (Japanese) */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&family=Sawarabi+Mincho&display=swap');

        /* --- GLOBAL --- */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
            color: #1a1a2e;
        }

        /* --- BACKGROUND: Aurora Gradient --- */
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- GLASSMORPHISM CARD (การ์ดกระจก) --- */
        div.block-container {
            max-width: 1000px;
            padding-top: 2rem;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            padding: 40px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            margin-bottom: 30px;
            transition: transform 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.2);
        }

        /* --- HEADER & TITLES --- */
        .hero-title {
            font-family: 'Sawarabi Mincho', serif;
            font-size: 4rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #C62828, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            text-shadow: 0px 4px 15px rgba(198, 40, 40, 0.2);
            letter-spacing: -2px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 40px;
            font-weight: 300;
            letter-spacing: 1px;
        }

        /* --- IMAGE FRAME (กรอบรูป) --- */
        div[data-testid="stImage"] {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            border: 5px solid #fff;
            transition: all 0.5s ease;
        }
        div[data-testid="stImage"]:hover {
            transform: scale(1.03) rotate(1deg);
            box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        }
        div[data-testid="stImage"] img {
            border-radius: 15px;
        }

        /* --- RESULT BOX (กล่องผลลัพธ์) --- */
        .result-container {
            background: white;
            border-radius: 25px;
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
            border: 2px solid #f0f0f0;
        }
        .result-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 10px;
            background: linear-gradient(90deg, #FF416C, #FF4B2B);
        }
        .big-char {
            font-size: 6rem;
            line-height: 1;
            font-weight: bold;
            color: #1a1a2e;
            margin: 10px 0;
            text-shadow: 4px 4px 0px #eee;
        }
        .confidence-badge {
            background: #e0ffe0;
            color: #00b894;
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-block;
            box-shadow: 0 4px 10px rgba(0, 184, 148, 0.2);
        }

        /* --- BUTTONS (ปุ่มกด) --- */
        .stButton button {
            border: none !important;
            border-radius: 15px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
            width: 100%;
        }
        
        /* ปุ่ม Primary (Analyze) - ไล่สีแดงทอง */
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
            color: white !important;
        }
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button[kind="primary"]:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 15px 30px rgba(255, 75, 43, 0.4) !important;
        }

        /* ปุ่ม Secondary (Next/Prev) */
        .stButton button[kind="secondary"] {
            background: white !important;
            color: #333 !important;
        }
        .stButton button[kind="secondary"]:hover {
            background: #f8f9fa !important;
            color: #FF4B2B !important;
            transform: translateY(-2px) !important;
        }

        /* --- RADIO BUTTONS --- */
        div[role="radiogroup"] {
            background: rgba(255,255,255,0.6);
            padding: 15px;
            border-radius: 20px;
            justify-content: center;
        }

        /* --- HOME BUTTON --- */
        .home-btn {
            background: #1a1a2e;
            color: #fff !important;
            padding: 15px 40px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            transition: all 0.3s;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .home-btn:hover {
            background: #0f3460;
            box-shadow: 0 0 20px rgba(15, 52, 96, 0.5);
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database & Model Logic (คงเดิม) ---
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

# --- 4. Main UI Layout ---
model = load_model()
class_names = load_class_names()

# Sidebar Stats
with st.sidebar:
    st.markdown("## 📊 Status")
    total_w, checked_w = get_stats()
    st.markdown(f"""
    <div style="background:white; padding:15px; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.05);">
        <h3 style="margin:0; color:#FF416C;">{total_w}</h3>
        <p style="margin:0; color:#888; font-size:0.8rem;">Total Images</p>
    </div>
    <div style="height:10px;"></div>
    <div style="background:white; padding:15px; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.05);">
        <h3 style="margin:0; color:#23a6d5;">{checked_w}</h3>
        <p style="margin:0; color:#888; font-size:0.8rem;">Analyzed</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**User:** Toey (Admin)")

# Header
st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent Handwriting Recognition System</div>', unsafe_allow_html=True)

# Filter
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c_fil_1, c_fil_2, c_fil_3 = st.columns([1, 4, 1])
with c_fil_2:
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

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    # Navigation Logic
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
    
    # --- MAIN GLASS CARD ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Top Control Bar (Prev/Next inside card)
    c_p, c_title, c_n = st.columns([1, 4, 1])
    with c_title:
        st.markdown(f"<div style='text-align:center; font-weight:bold; color:#555;'>Image ID: {current_id} ({st.session_state.current_index + 1}/{len(id_list)})</div>", unsafe_allow_html=True)

    data_row = get_work_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        try: image = Image.open(io.BytesIO(blob_data))
        except: image = None

        if image:
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.markdown(f"##### 🎯 Target: `{true_label}`")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("##### 🤖 AI Analysis")
                st.write("")
                
                if saved_result:
                    # Result Display
                    char_part = saved_result.split(' ')[0]
                    romaji_part = saved_result.split(' ')[1] if len(saved_result.split(' ')) > 1 else ''
                    
                    st.markdown(f"""
                        <div class="result-container">
                            <p style="color:#aaa; font-size:0.8rem; letter-spacing:1px;">PREDICTION</p>
                            <div class="big-char">{char_part}</div>
                            <p style="font-size:1.2rem; color:#555; margin-bottom:15px;">{romaji_part}</p>
                            <div class="confidence-badge">✨ {saved_conf:.1f}% Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 Re-Analyze", type="secondary"):
                        update_database(current_id, None, 0)
                        st.rerun()
                else:
                    # Pending Display
                    st.markdown("""
                        <div class="result-container" style="border: 2px dashed #ddd; background: #fafafa;">
                            <div style="padding: 40px 0; color: #ccc;">
                                <h1>⏳</h1>
                                <p>Waiting for Analysis</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("⚡ ANALYZE NOW", type="primary"):
                        if model:
                            with st.spinner("AI is thinking..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    # Mapping
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
                                    st.success(f"Result: {final_res}")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("⚠️ Model not loaded")

    st.markdown('</div>', unsafe_allow_html=True) # End Glass Card

    # Navigation Buttons (Bottom)
    col_prev, col_space, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.session_state.current_index > 0:
            if st.button("⬅️ PREVIOUS", key="nav_prev"):
                st.session_state.current_index -= 1
                st.rerun()
    with col_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("NEXT ➡️", key="nav_next"):
                st.session_state.current_index += 1
                st.rerun()
        else:
            if st.button("⏮ RESTART", key="nav_reset"):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.markdown("""
        <div class="glass-card" style="text-align:center; padding:60px;">
            <h1 style="font-size:80px; margin:0;">📭</h1>
            <h3 style="color:#555;">No Data Found</h3>
            <p style="color:#999;">Please select a different category or upload data.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
teacher_dashboard_url = "https://www.cedubru.com/hiragana/teacher.php?view_student=7" 
st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; padding-bottom: 30px;">
        <a href="{teacher_dashboard_url}" target="_self" class="home-btn">
            🏠 Return to Dashboard
        </a>
        <p style="margin-top:20px; color:#1a1a2e; font-size:0.8rem; opacity:0.6;">
            Hiragana Image Classification System V.3.0 Ultimate | Design by Hiragana Sensei Team
        </p>
    </div>
""", unsafe_allow_html=True)