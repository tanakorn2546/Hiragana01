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
    layout="centered", # เปลี่ยนเป็น Centered เพื่อให้โฟกัสที่เนื้อหาตรงกลางเหมือนแอปมือถือ
    initial_sidebar_state="collapsed"
)

# --- 2. CSS Styling (Modern Zen Tech Design) ---
def local_css():
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Zen+Maru+Gothic:wght@500;700&display=swap');

        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #4D96FF;
            --bg-color: #Fdfbf7;
            --card-bg: rgba(255, 255, 255, 0.65);
            --text-color: #2D3436;
        }

        /* --- Global Settings --- */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
            color: var(--text-color);
            background-color: var(--bg-color);
        }

        /* --- Background Animation (Soft & Flowing) --- */
        .stApp {
            background: radial-gradient(circle at 0% 0%, #ffe6fa 0%, transparent 50%), 
                        radial-gradient(circle at 100% 100%, #e0f7fa 0%, transparent 50%);
            background-attachment: fixed;
        }

        /* --- Modern Glass Card --- */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            padding: 40px 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }
        
        /* --- Hero Section --- */
        .hero-title {
            font-family: 'Zen Maru Gothic', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #FF6B6B, #556270);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1rem;
            color: #636e72;
            font-weight: 300;
            margin-bottom: 30px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        /* --- Buttons Styling --- */
        .stButton button {
            border-radius: 50px !important;
            font-family: 'Prompt', sans-serif !important;
            font-weight: 500 !important;
            padding: 0.6rem 1.5rem !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }

        /* Primary Button (Analyze) */
        div[data-testid="stVerticalBlock"] .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
            color: white !important;
            font-size: 1.1rem !important;
        }
        div[data-testid="stVerticalBlock"] .stButton button[kind="primary"]:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4) !important;
        }

        /* Secondary Button (Nav) */
        div[data-testid="stVerticalBlock"] .stButton button[kind="secondary"] {
            background: white !important;
            color: #555 !important;
            border: 1px solid #eee !important;
        }
        div[data-testid="stVerticalBlock"] .stButton button[kind="secondary"]:hover {
            border-color: #FF6B6B !important;
            color: #FF6B6B !important;
            background: #fff0f0 !important;
        }

        /* --- Result Display --- */
        .result-card {
            background: white;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        }
        .result-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 6px;
            background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        }
        .big-char {
            font-family: 'Zen Maru Gothic', sans-serif;
            font-size: 5.5rem;
            color: #2d3436;
            line-height: 1;
            margin: 10px 0;
        }
        .romaji-tag {
            background: #f1f2f6;
            color: #57606f;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }
        .confidence-pill {
            margin-top: 10px;
            font-size: 0.85rem;
            color: #27ae60;
            background: rgba(39, 174, 96, 0.1);
            padding: 4px 12px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        /* --- Image Frame --- */
        div[data-testid="stImage"] img {
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        div[data-testid="stImage"] img:hover {
            transform: scale(1.02);
        }

        /* --- Custom Progress Bar --- */
        div[data-testid="stProgress"] > div > div > div {
            background-color: #FF6B6B;
            background-image: linear-gradient(315deg, #FF6B6B 0%, #FF8E53 74%);
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database & Model Functions (คงเดิม) ---
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

# --- 4. UI Logic ---
model = load_model()
class_names = load_class_names()

# Sidebar (Stats Only - Cleaner)
with st.sidebar:
    st.markdown("### 📊 Dataset Status")
    total_w, checked_w = get_stats()
    
    st.markdown(f"""
    <div style="background:white; padding:15px; border-radius:12px; margin-bottom:10px; border-left: 4px solid #FF6B6B;">
        <span style="font-size:0.8rem; color:#888;">All Images</span>
        <h3 style="margin:0; color:#2d3436;">{total_w}</h3>
    </div>
    <div style="background:white; padding:15px; border-radius:12px; border-left: 4px solid #4D96FF;">
        <span style="font-size:0.8rem; color:#888;">Processed</span>
        <h3 style="margin:0; color:#2d3436;">{checked_w}</h3>
    </div>
    """, unsafe_allow_html=True)

# Main Header
st.markdown('<div class="hero-title">Hiragana AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent Recognition System</div>', unsafe_allow_html=True)

# Filter Logic
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    if target_work_id:
        st.info(f"🔍 Viewing Specific ID: {target_work_id}")
        filter_option = "ทั้งหมด (All)"
    else:
        filter_option = st.selectbox(
            "Filter Data",
            ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"],
            label_visibility="collapsed"
        )

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    # State Management
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Boundary Check
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0
    elif st.session_state.current_index < 0:
        st.session_state.current_index = len(id_list) - 1

    current_id = id_list[st.session_state.current_index]
    
    # Progress
    st.progress((st.session_state.current_index + 1) / len(id_list))
    
    # --- Modern Card UI ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Card Toolbar
    c_meta, c_nav = st.columns([2, 1])
    with c_meta:
        st.caption(f"🆔 Image ID: {current_id} | Sequence: {st.session_state.current_index + 1}/{len(id_list)}")

    data_row = get_work_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        try: image = Image.open(io.BytesIO(blob_data))
        except: image = None

        if image:
            col_img, col_res = st.columns([1, 1.2], gap="large")
            
            # Left: Image
            with col_img:
                st.markdown(f"**Target:** `{true_label}`")
                st.image(image, use_container_width=True)
            
            # Right: Action / Result
            with col_res:
                st.markdown("**AI Analysis Result**")
                
                if saved_result:
                    # Parse Result
                    parts = saved_result.split(' ')
                    char_part = parts[0]
                    romaji_part = parts[1] if len(parts) > 1 else ''
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="romaji-tag">{romaji_part}</div>
                        <div class="big-char">{char_part}</div>
                        <div class="confidence-pill">
                            ⚡ Confidence: {saved_conf:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 Re-Analyze", type="secondary", use_container_width=True):
                        update_database(current_id, None, 0)
                        st.rerun()
                        
                else:
                    # Pending State
                    st.markdown("""
                    <div class="result-card" style="border: 2px dashed #eee; box-shadow:none; padding:40px;">
                        <div style="font-size:3rem; opacity:0.3; margin-bottom:10px;">🧠</div>
                        <div style="color:#aaa; font-size:0.9rem;">Waiting for analysis...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("✨ Analyze Now", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("Processing..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    # Dictionary
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
                                    time.sleep(0.3)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Model not loaded.")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation Buttons (Pill Shape)
    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
         if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.current_index -= 1
            st.rerun()
    with col_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()
        else:
            if st.button("⏮ Restart", use_container_width=True):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.markdown("""
        <div class="glass-card" style="text-align:center; padding:60px;">
            <div style="font-size:4rem; margin-bottom:20px;">📭</div>
            <h3 style="color:#555;">No Data Found</h3>
            <p style="color:#888;">Please check the database or filters.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding-bottom: 20px; border-top:1px solid rgba(0,0,0,0.05); padding-top:20px;">
        <a href="https://www.cedubru.com/hiragana/teacher.php?view_student=7" style="text-decoration:none; color:#FF6B6B; font-weight:600;">
            ← Return to Dashboard
        </a>
        <p style="margin-top:10px; font-size:0.75rem; color:#aaa;">
            Hiragana Character Classification System • Developed for Research
        </p>
    </div>
""", unsafe_allow_html=True)