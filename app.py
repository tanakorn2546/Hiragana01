import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os
import mysql.connector
import io
import gdown

# 1. แก้ไข: นำเข้า preprocess_input ของ ResNet50 ให้ตรงกับโมเดลที่เทรนมา
from tensorflow.keras.applications.resnet50 import preprocess_input 

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Hiragana Sensei AI",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
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
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(15px);
            border-radius: 20px; border: 2px solid white; padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; position: relative; z-index: 10;
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
TABLE_CONFIG = {
    "progress": {
        "label_col": "char_code",       
        "image_col": "image_data",      
        "result_col": "ai_result",      
        "conf_col": "ai_confidence"     
    },
    "quiz_submissions": {
        "label_col": "char_label",      
        "image_col": "image_data",
        "result_col": "ai_result",      
        "conf_col": "ai_confidence"     
    }
}

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
        if "ยังไม่ตรวจ" in filter_mode: sql = f"{base_sql} AND ai_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode: sql = f"{base_sql} AND ai_result IS NOT NULL ORDER BY id DESC"
        else: sql = f"{base_sql} ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ Database List Error: {e}")
        return []

def get_work_data(target_id, table_name="progress"):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        config = TABLE_CONFIG.get(table_name)
        if not config: return None

        sql = f"""
            SELECT {config['image_col']}, {config['result_col']}, {config['conf_col']}, {config['label_col']} 
            FROM {table_name} WHERE id = %s
        """
        cursor.execute(sql, (target_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except Exception as e:
        st.error(f"❌ Data Fetch Error: {e}")
        return None

def update_database(target_id, table_name, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        config = TABLE_CONFIG.get(table_name)
        sql = f"UPDATE {table_name} SET {config['result_col']} = %s, {config['conf_col']} = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), target_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Update Error: {e}")
        return False

def get_stats():
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        return cursor.fetchone()
    except: return 0, 0

# --- 4. Model Loading ---
@st.cache_resource
def load_model():
    GOOGLE_DRIVE_FILE_ID = '1y6m5PCGmDFxYF8yz4FyiUVYGOiffdtD2' # อัปเดต ID ไฟล์ใหม่หากอัปขึ้น Drive
    # เปลี่ยนชื่อไฟล์ให้สอดคล้องกับ ResNet50 (ถ้าคุณอัปโหลดไฟล์ใหม่ทับไปแล้วใช้ชื่อเดิม ก็ไม่ต้องเปลี่ยนครับ)
    model_filename = 'resnet50_hiragana_model_research.h5' 
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    
    if os.path.exists(model_filename):
        final_path = model_filename
    elif os.path.exists(os.path.join('saved_models', model_filename)):
        final_path = os.path.join('saved_models', model_filename)
    else:
        try:
            st.info(f"☁️ Downloading Model...")
            gdown.download(url, model_filename, quiet=False)
            final_path = model_filename
            st.success("✅ Download Success!")
        except Exception as e:
            st.warning(f"⚠️ หาไฟล์โมเดลไม่เจอ: {e}")
            return None

    try:
        return tf.keras.models.load_model(final_path, compile=False)
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None

def load_class_names():
    return [
        'a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko',
        'sa', 'shi', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to',
        'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
        'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo',
        'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo', 'nn'
    ]

# --- 5. Cleaned Preprocessing ---
def import_and_predict(image, model):
    # จัดการพื้นหลังโปร่งใสให้เป็นสีขาว
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) 
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # ย่อขนาดและแปลงเป็น Array
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    
    # เพิ่มมิติภาพ (Batch) ก่อน
    img_batch = np.expand_dims(img_array, axis=0)
    
    # ใช้ preprocess_input ของ ResNet50 
    processed_img = preprocess_input(img_batch)
    
    return model.predict(processed_img)

# --- 6. Scoring Logic ---
def calculate_score(conf):
    if conf >= 85.0: return 1.0
    elif conf >= 60.0: return 0.5
    else: return 0.0

# --- 7. Main Application Logic ---
model = load_model()
class_names = load_class_names()

with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล (Practice)")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")
    st.markdown("---")
    st.markdown("### 📊 เกณฑ์การให้คะแนน")
    st.markdown("""
    - **85% - 100%**: ได้ 1 คะแนน
    - **60% - 84.9%**: ได้ 0.5 คะแนน
    - **ต่ำกว่า 60%**: ได้ 0 คะแนน
    """)

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือ (Ultimate)</div>', unsafe_allow_html=True)

query_params = st.query_params
req_work_id = query_params.get("work_id", None)
req_quiz_id = query_params.get("quiz_id", None)

current_id = None
active_table = "progress"
is_single_view = False
mode_color = "#D72638"

if req_quiz_id:
    current_id = req_quiz_id
    active_table = "quiz_submissions"
    is_single_view = True
    mode_color = "#7c3aed"
    st.markdown(f"""
    <div style="background:#f3e8ff; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold;">
        📝 กำลังตรวจ: แบบทดสอบ (Quiz ID: {current_id})
    </div>
    <style>.stApp {{ background: linear-gradient(180deg, #f3e8ff 0%, #fff 60%, #fff 100%) !important; }}</style>
    """, unsafe_allow_html=True)
elif req_work_id:
    current_id = req_work_id
    active_table = "progress"
    is_single_view = True
    st.markdown(f"""
    <div style="background:#ffebee; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold;">
        ✍️ กำลังตรวจ: แบบฝึกหัด (Work ID: {current_id})
    </div>
    """, unsafe_allow_html=True)

if is_single_view:
    if current_id:
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
                        score = calculate_score(saved_conf) 
                        
                        st.markdown(f"""
                        <div class="result-card" style="border-top-color:{mode_color};">
                            <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                            <div class="big-char" style="color:{mode_color};">{char_part}</div>
                            <div style="color:green; font-weight:bold; margin-top:10px;">
                                ความมั่นใจ: {saved_conf:.1f}%<br>
                                <span style="font-size:1.2rem; color:#FF9800;">ได้คะแนน: {score}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        st.write("")
                        if st.button("🔄 ตรวจใหม่", use_container_width=True):
                            update_database(current_id, active_table, None, 0)
                            st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="result-card" style="border: 2px dashed #ddd; background:#fffaf0;">
                            <h1 style="color:{mode_color}; opacity:0.5;">⏳</h1>
                        </div>""", unsafe_allow_html=True)
                        st.write("")
                        if st.button("✨ วิเคราะห์", type="primary", use_container_width=True):
                            if model:
                                with st.spinner("AI กำลังคิด..."):
                                    try:
                                        preds = import_and_predict(image, model)
                                        idx = np.argmax(preds)
                                        conf = np.max(preds) * 100
                                        
                                        if conf < 60.0: 
                                            final_res = "❓ Unknown (เขียนใหม่)"
                                            res_code = "Unknown"
                                        else:
                                            res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                            hiragana_map = {
                                                'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)',
                                                'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)',
                                                'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)',
                                                'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)',
                                                'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)', # แก้บั๊ก 'ของ' เป็น 'の'
                                                'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)',
                                                'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)',
                                                'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)',
                                                'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)',
                                                'wa': 'わ (wa)', 'wo': 'を (wo)', 'nn': 'ん (nn)'
                                            }
                                            final_res = hiragana_map.get(res_code, res_code)
                                        
                                        if update_database(current_id, active_table, final_res, conf):
                                            time.sleep(0.3); st.rerun()
                                    except Exception as e: st.error(f"Error: {e}")
                            else: st.error("ไม่พบโมเดล")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2: filter_option = st.selectbox("โหมด", ["ทั้งหมด", "ยังไม่ตรวจ", "ตรวจแล้ว"], label_visibility="collapsed")
    work_list = get_work_list(filter_option)

    if len(work_list) > 0:
        if 'current_index' not in st.session_state: st.session_state.current_index = 0
        if st.session_state.current_index >= len(work_list): st.session_state.current_index = 0
        elif st.session_state.current_index < 0: st.session_state.current_index = len(work_list) - 1
        
        browse_id = work_list[st.session_state.current_index][0]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption(f"ID: {browse_id} | {st.session_state.current_index + 1}/{len(work_list)}")

        data_row = get_work_data(browse_id, "progress")
        if data_row:
            blob_data, saved_result, saved_conf, true_label = data_row
            try: image = Image.open(io.BytesIO(blob_data))
            except: image = None
            if image:
                col_img, col_res = st.columns([1, 1.2], gap="large")
                with col_img: st.markdown(f"**โจทย์:** `{true_label}`"); st.image(image, use_container_width=True)
                with col_res:
                    if saved_result:
                        score = calculate_score(saved_conf) 
                        st.success(f"{saved_result}\n\nความมั่นใจ: {saved_conf:.1f}%\n\n**ได้คะแนน: {score}**")
                        if st.button("🔄 ตรวจใหม่"): update_database(browse_id, "progress", None, 0); st.rerun()
                    else:
                        if st.button("✨ วิเคราะห์"):
                            if model:
                                with st.spinner("AI Thinking..."):
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds); conf = np.max(preds) * 100
                                    
                                    if conf < 60.0: 
                                        final_res = "❓ Unknown (เขียนใหม่)"
                                    else:
                                        res_code = class_names[idx]
                                        hiragana_map = {
                                            'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)',
                                            'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)',
                                            'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)',
                                            'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)',
                                            'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)', # แก้บั๊ก
                                            'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)',
                                            'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)',
                                            'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)',
                                            'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)',
                                            'wa': 'わ (wa)', 'wo': 'を (wo)', 'nn': 'ん (nn)'
                                        }
                                        final_res = hiragana_map.get(res_code, res_code)
                                    
                                    # แก้บั๊ก: เปลี่ยน final_final_res เป็น final_res 
                                    update_database(browse_id, "progress", final_res, conf)
                                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        c_prev, c_space, c_next = st.columns([1, 0.2, 1])
        with c_prev: 
            if st.button("⬅️ ก่อนหน้า", use_container_width=True): st.session_state.current_index -= 1; st.rerun()
        with c_next: 
            if st.button("ถัดไป ➡️", use_container_width=True): st.session_state.current_index += 1; st.rerun()
    else: st.info("ไม่พบข้อมูล")

st.markdown("""<div style="text-align: center; margin-top: 50px;"><a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">🏠 กลับสู่หน้าหลัก</a></div>""", unsafe_allow_html=True)