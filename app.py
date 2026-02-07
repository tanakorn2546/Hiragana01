import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import cv2
import gdown

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

# --- 4. Model Loading with FIX ---

class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    # ⚠️⚠️⚠️ ใส่ ID ของไฟล์โมเดลตัวใหม่ใน Google Drive ตรงนี้ ⚠️⚠️⚠️
    # ถ้ายังไม่ได้อัปโหลดใหม่ ให้ใช้วิธีเอาไฟล์วางไว้ข้างๆ app.py แทนชั่วคราว
    GOOGLE_DRIVE_FILE_ID = '1BkIL8jF1XERR3jPXKAOmSb11Qd-MbYQg' 
    # -------------------------------------------------------------
    
    # ✅ เปลี่ยนชื่อไฟล์ให้ตรงกับที่เทรนมาใหม่
    model_filename = 'hiragana_mobilenet_v2_smart_crop.h5'
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    
    if not os.path.exists(model_filename):
        local_path = os.path.join('saved_models', model_filename)
        if os.path.exists(local_path):
            final_path = local_path
        else:
            # ถ้าหาไฟล์ไม่เจอ จะลองโหลด (แต่ถ้า ID ผิดจะโหลดไฟล์เก่ามา ให้ระวัง)
            try:
                st.info(f"☁️ Downloading Model... (ID: {GOOGLE_DRIVE_FILE_ID})")
                gdown.download(url, model_filename, quiet=False)
                final_path = model_filename
                st.success("✅ Download Success!")
            except Exception as e:
                # ถ้าโหลดไม่ได้ ให้ลองหาไฟล์ชื่อเก่าเผื่อไว้
                old_filename = 'hiragana_mobilenet_v2_enhancedv2.h5'
                if os.path.exists(old_filename):
                    final_path = old_filename
                    st.warning("⚠️ ใช้โมเดลรุ่นเก่า (ไม่พบไฟล์ใหม่)")
                else:
                    st.error(f"❌ Model not found: {e}")
                    return None
    else:
        final_path = model_filename

    try:
        return tf.keras.models.load_model(
            final_path, 
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
            compile=False 
        )
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None

def load_class_names():
    return [
        'a', 'i', 'u', 'e', 'o',
        'ka', 'ki', 'ku', 'ke', 'ko',
        'sa', 'shi', 'su', 'se', 'so',
        'ta', 'chi', 'tsu', 'te', 'to',
        'na', 'ni', 'nu', 'ne', 'no',
        'ha', 'hi', 'fu', 'he', 'ho',
        'ma', 'mi', 'mu', 'me', 'mo',
        'ya', 'yu', 'yo',
        'ra', 'ri', 'ru', 're', 'ro',
        'wa', 'wo', 'nn'
    ]

# --- 5. New Smart Preprocessing (ทีเด็ด!) ---
IMG_SIZE = 128 # ต้องตรงกับที่เทรน

def smart_process_image(pil_image):
    """
    ฟังก์ชันเตรียมภาพแบบฉลาด: หา Contours > Crop > Resize > Pad
    """
    try:
        # 1. แปลง PIL เป็น OpenCV Format (RGB)
        img_array = np.array(pil_image.convert('RGB'))
        
        # 2. แปลงเป็น Grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 3. Thresholding (กลับสี: พื้นดำ ตัวหนังสือขาว เพื่อหา Contours)
        # ใช้ Otsu เพื่อหาค่าที่ดีที่สุด
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. หา Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # หา Bounding Box ที่ใหญ่ที่สุด (สมมติว่าเป็นตัวอักษร)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # กรอง Noise จุดเล็กๆ
            if w > 10 and h > 10:
                # Crop เอาเฉพาะตัวอักษร
                roi = gray[y:y+h, x:x+w]
                
                # คำนวณขนาดที่จะ Resize ลงไปโดยรักษา Aspect Ratio
                target_h, target_w = IMG_SIZE, IMG_SIZE
                final_img = np.ones((target_h, target_w), dtype=np.uint8) * 255 # พื้นขาว
                
                aspect = w / h
                if aspect > 1: # กว้างมากกว่าสูง
                    new_w = target_w - 20 # เผื่อขอบ
                    new_h = int(new_w / aspect)
                else: # สูงมากกว่ากว้าง
                    new_h = target_h - 20
                    new_w = int(new_h * aspect)
                
                # Resize
                roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # แปะลงกลางภาพ
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized
                
                # แปลงกลับเป็น RGB 3 Channels
                final_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
                return tf.keras.applications.mobilenet_v2.preprocess_input(final_rgb.astype(np.float32))

        # ถ้าหา Contour ไม่เจอ (เช่น ภาพขาวล้วน) หรือ Error ให้ใช้ภาพเดิม Resize เอา
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32))

    except Exception as e:
        st.error(f"Error in processing: {e}")
        # Fallback กรณี Error จริงๆ
        img_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array.astype(np.float32))

def import_and_predict(image_data, model):
    # ส่ง PIL Image เข้าไปที่ฟังก์ชัน smart_process_image โดยตรง
    processed_img = smart_process_image(image_data)
    img_batch = np.expand_dims(processed_img, axis=0)
    return model.predict(img_batch)

# --- 6. Main Application Logic ---
model = load_model()
class_names = load_class_names()

with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล (Practice)")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือด้วย MobileNetV2 (Final+SmartCrop)</div>', unsafe_allow_html=True)

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
                        st.markdown(f"""
                        <div class="result-card" style="border-top-color:{mode_color};">
                            <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                            <div class="big-char" style="color:{mode_color};">{char_part}</div>
                            <div style="color:green; font-weight:bold;">{saved_conf:.1f}%</div>
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
                                        
                                        # 🔥🔥🔥 Unknown Logic (Smart Crop จะแม่นยำขึ้น อาจปรับ % ขึ้นได้) 🔥🔥🔥
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
                                                'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)',
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
                        st.success(f"{saved_result}\n\nConf: {saved_conf:.1f}%")
                        if st.button("🔄 ตรวจใหม่"): update_database(browse_id, "progress", None, 0); st.rerun()
                    else:
                        if st.button("✨ วิเคราะห์"):
                            if model:
                                with st.spinner("AI Thinking..."):
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds); conf = np.max(preds) * 100
                                    
                                    # 🔥🔥🔥 Unknown Logic 🔥🔥🔥
                                    if conf < 65.0: # ปรับระดับตรงนี้
                                        final_res = "❓ Unknown (เขียนใหม่)"
                                    else:
                                        res_code = class_names[idx]
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
                                            'wa': 'わ (wa)', 'wo': 'を (wo)', 'nn': 'ん (nn)'
                                        }
                                        final_res = hiragana_map.get(res_code, res_code)
                                    
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