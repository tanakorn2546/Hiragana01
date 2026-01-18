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
    initial_sidebar_state="collapsed"
)

# --- 2. CSS Styling (Fuji & Waves Animation) ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Zen+Maru+Gothic:wght@700&display=swap');

        /* --- Theme Variables --- */
        :root {
            --japan-red: #D72638; /* สีแดงแบบธงชาติ/เสาโทริอิ */
            --bg-sky: linear-gradient(to top, #a18cd1 0%, #fbc2eb 100%);
            --wave-color: rgba(255, 255, 255, 0.4);
        }

        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
        }

        /* --- Background Animation Container --- */
        .stApp {
            background: linear-gradient(180deg, #d4fcff 0%, #fff 60%, #fff 100%);
            background-attachment: fixed;
            overflow-x: hidden;
        }

        /* สร้างภูเขาไฟฟูจิด้วย CSS (อยู่ด้านหลังสุด) */
        .stApp::before {
            content: "";
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 300px solid transparent;
            border-right: 300px solid transparent;
            border-bottom: 250px solid #a2d2ff; /* ตัวภูเขา */
            z-index: 0;
            filter: drop-shadow(0 -10px 20px rgba(0,0,0,0.1));
        }
        /* หิมะบนยอดฟูจิ */
        .stApp::after {
            content: "";
            position: fixed;
            bottom: 160px; /* ปรับความสูงหิมะ */
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 90px solid transparent;
            border-right: 90px solid transparent;
            border-bottom: 90px solid white;
            z-index: 0;
        }
        
        /* ดวงอาทิตย์สีแดง */
        div[data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: 10%;
            right: 15%;
            width: 100px;
            height: 100px;
            background: #FF4E50;
            border-radius: 50%;
            box-shadow: 0 0 40px rgba(255, 78, 80, 0.4);
            z-index: 0;
            animation: sunPulse 5s infinite alternate;
        }
        @keyframes sunPulse {
            0% { transform: scale(1); opacity: 0.9; }
            100% { transform: scale(1.1); opacity: 1; }
        }

        /* --- Moving Waves (คลื่นขยับด้านล่าง) --- */
        /* เราจะใช้ CSS Masking หรือ Background Image ซ้อนกันเพื่อทำคลื่น */
        /* เพื่อความง่ายและสวยงาม ใช้ CSS Gradient ทำลาย Seigaiha (คลื่นญี่ปุ่น) */
        
        .wave-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 150px;
            background: url('https://www.transparenttextures.com/patterns/seigaiha.png');
            background-color: #4facfe;
            opacity: 0.8;
            z-index: 1;
            mask-image: linear-gradient(to top, black 20%, transparent 100%);
            -webkit-mask-image: linear-gradient(to top, black 20%, transparent 100%);
            animation: waveMove 60s linear infinite;
        }
        @keyframes waveMove {
            0% { background-position: 0 0; }
            100% { background-position: 500px 0; }
        }

        /* --- Glass Card --- */
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 2px solid white;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: relative;
            z-index: 10; /* ให้อยู่เหนือภูเขา */
        }

        /* --- Buttons (Red Theme) --- */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        /* ปุ่มหลัก & ปุ่มรอง เปลี่ยนเป็นโทนแดงทั้งหมดตามขอ */
        div[data-testid="stVerticalBlock"] .stButton button {
            background: var(--japan-red) !important;
            color: white !important;
        }
        div[data-testid="stVerticalBlock"] .stButton button:hover {
            background: #b71c1c !important; /* แดงเข้มขึ้นเมื่อเอาเมาส์ชี้ */
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(215, 38, 56, 0.4) !important;
        }
        
        /* ปุ่ม Navigation ให้เป็นสีขาวขอบแดง เพื่อไม่ให้แย่งซีน */
        div[data-testid="stHorizontalBlock"] .stButton button {
            background: white !important;
            color: var(--japan-red) !important;
            border: 2px solid var(--japan-red) !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton button:hover {
            background: var(--japan-red) !important;
            color: white !important;
        }

        /* --- Typography --- */
        .hero-title {
            font-family: 'Zen Maru Gothic', sans-serif;
            font-size: 3.5rem;
            color: var(--japan-red);
            text-align: center;
            text-shadow: 2px 2px 0px white;
            margin-bottom: 0;
            position: relative;
            z-index: 10;
        }
        .hero-subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 30px;
            position: relative;
            z-index: 10;
        }
        
        /* --- Result Card --- */
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border-top: 5px solid var(--japan-red);
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .big-char {
            font-size: 5rem;
            color: var(--japan-red);
            font-weight: bold;
            line-height: 1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Inject Wave Container
    st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

local_css()

# --- 3. Database & Model Functions (เหมือนเดิม) ---
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
    file_id = '1uT5Pg7vnf-Gbl7w6i8FGQyZv8QHYmmFH' 
    model_name = 'efficientnetv2_hiragana_final.h5'
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

# Sidebar
with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")

# Header
st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือภาษาญี่ปุ่นอัจฉริยะ</div>', unsafe_allow_html=True)

# Filter
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    if target_work_id:
        st.info(f"🔍 Viewing ID: {target_work_id}")
        filter_option = "ทั้งหมด (All)"
    else:
        filter_option = st.selectbox("เลือกโหมดการแสดงผล", ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"], label_visibility="collapsed")

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0
    elif st.session_state.current_index < 0: st.session_state.current_index = len(id_list) - 1

    current_id = id_list[st.session_state.current_index]
    
    # --- ลบ st.progress ออกตามที่ขอ --- 
    # (พื้นที่นี้จะว่างลง ทำให้ดูสะอาดตาขึ้น)

    # --- Glass Card ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.caption(f"Image ID: {current_id} | {st.session_state.current_index + 1}/{len(id_list)}")

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
                st.markdown("**ผลการตรวจ (AI Result)**")
                
                if saved_result:
                    parts = saved_result.split(' ')
                    char_part = parts[0]
                    romaji_part = parts[1] if len(parts) > 1 else ''
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                        <div class="big-char">{char_part}</div>
                        <div style="color:green; font-weight:bold;">ความมั่นใจ: {saved_conf:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 ตรวจใหม่", type="secondary", use_container_width=True):
                        update_database(current_id, None, 0)
                        st.rerun()
                else:
                    st.markdown("""
                    <div class="result-card" style="border: 2px dashed #ffcdd2; background:#fffaf0;">
                        <h1 style="color:#ef5350; opacity:0.5;">⏳</h1>
                        <p style="color:#888;">รอการตรวจ...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("✨ วิเคราะห์ทันที", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("AI กำลังเพ่งจิต..."):
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
                                    update_database(current_id, final_res, conf)
                                    time.sleep(0.3)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Model Error")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    c_prev, c_space, c_next = st.columns([1, 0.2, 1])
    with c_prev:
        if st.button("⬅️ รูปก่อนหน้า", use_container_width=True):
            st.session_state.current_index -= 1
            st.rerun()
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("รูปถัดไป ➡️", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()
        else:
            if st.button("⏮ เริ่มใหม่", use_container_width=True):
                st.session_state.current_index = 0
                st.rerun()
else:
    st.info("ไม่พบข้อมูล")

# Footer Link
st.markdown("""
    <div style="text-align: center; margin-top: 50px; position:relative; z-index:20;">
        <a href="https://www.cedubru.com/hiragana/teacher.php?view_student=7" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">
            🏠 กลับสู่หน้าหลัก
        </a>
    </div>
""", unsafe_allow_html=True)