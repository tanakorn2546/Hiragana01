import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import json
import gdown 
import requests 

# --- [Config] ธีมญี่ปุ่น (ขาว-แดง-ชมพู) ---
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

with open(config_path, "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#D32F2F"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF0F5"\ntextColor="#333333"\n')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Hiragana Sensei AI",
    page_icon="🇯🇵",
    layout="centered"
)

# --- 2. CSS ตกแต่ง ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Prompt', sans-serif !important;
            color: #333333 !important;
        }
        .stApp {
            background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }
        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 30px !important;
            padding: 2rem 2rem 4rem 2rem !important; 
            margin-top: 2rem !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.1) !important;
            border-top: 5px solid #D32F2F;
        }
        .app-header-icon {
            font-size: 80px !important;
            background: radial-gradient(circle, #ffcdd2 0%, #ffffff 100%) !important;
            width: 140px !important;
            height: 140px !important;
            border-radius: 50% !important;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px auto !important;
            box-shadow: 0 10px 25px rgba(211, 47, 47, 0.2) !important;
            border: 5px solid #ffffff !important;
        }
        h1 { 
            text-align: center; color: #D32F2F !important; 
            font-weight: 800 !important; font-size: 2.2rem !important;
        }
        .result-box {
            background-color: #FFEBEE; 
            padding: 20px; 
            border-radius: 15px; 
            border: 2px solid #D32F2F; 
            margin-bottom: 20px; 
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Functions ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

def get_student_work(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT image_path, ai_result, ai_confidence FROM progress WHERE id = %s"
        cursor.execute(sql, (work_id,))
        data = cursor.fetchone()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error (Student): {e}")
        return None

def update_student_progress(work_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), work_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Update Error: {e}")
        return False

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        table_name = "culantro_images" 
        if "ยังไม่ตรวจ" in filter_mode:
            sql = f"SELECT id, image_name, prediction_result FROM {table_name} WHERE prediction_result IS NULL ORDER BY id ASC"
        else:
            sql = f"SELECT id, image_name, prediction_result FROM {table_name} ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except: return []

def get_image_data_gallery(img_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT image_data, prediction_result, confidence FROM culantro_images WHERE id = %s", (img_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

# --- 4. Model Loader (รองรับทุกเวอร์ชัน) ---
# [แก้จุดที่ 1] เลือกใช้ Decorator ให้ตรงกับเวอร์ชัน
if hasattr(st, 'cache_resource'):
    cache_decorator = st.cache_resource
else:
    # สำหรับ Streamlit เวอร์ชันเก่า
    cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    file_id = '1ezDUsDxeabZX06ArdjtcWPk0uradYWDD' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        try:
            gdown.download(url, model_name, quiet=False)
        except: return None
    
    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
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
        'wa', 'wo', 'n'
    ]

def get_display_text(romaji):
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
    return mapping.get(romaji, romaji)

def predict_image(image, model, class_names):
    size = (224, 224) 
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    if image.mode != "RGB": image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    
    preds = model.predict(data)
    idx = np.argmax(preds)
    conf = np.max(preds) * 100
    label = class_names[idx] if idx < len(class_names) else "Unknown"
    return label, conf

# [แก้จุดที่ 2] ฟังก์ชันช่วยดึงค่าจาก URL รองรับทุกเวอร์ชัน
def get_query_param(key):
    # ลองใช้แบบใหม่ (v1.30+)
    if hasattr(st, 'query_params'):
        return st.query_params.get(key)
    
    # ถ้าไม่มีให้ใช้แบบเก่า (v1.29-)
    try:
        params = st.experimental_get_query_params()
        if key in params:
            return params[key][0] # แบบเก่าคืนค่าเป็น list ['val']
        return None
    except:
        return None

# --- 5. Main Application Flow ---
model = load_model()
class_names = load_class_names()

st.markdown("""
    <div class='app-header-icon'>🇯🇵</div>
    <h1>Hiragana Sensei AI</h1>
""", unsafe_allow_html=True)

# ใช้ฟังก์ชันช่วยดึงค่า work_id
target_work_id = get_query_param("work_id")

# ==========================================
# 🅰️ MODE 1: Teacher Direct Link (มี ID)
# ==========================================
if target_work_id:
    st.info(f"📋 โหมดตรวจสอบงานนักเรียน (Work ID: {target_work_id})")
    
    work_data = get_student_work(target_work_id)
    
    if work_data and work_data['image_path']:
        image_url = f"https://www.cedubru.com/{work_data['image_path']}"
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    st.image(image, caption="รูปจากนักเรียน", use_column_width=True)
                    
                    if st.button("🔍 ตรวจสอบด้วย AI", type="primary"):
                        if model:
                            with st.spinner("AI กำลังวิเคราะห์..."):
                                label_romaji, conf = predict_image(image, model, class_names)
                                final_text = get_display_text(label_romaji)
                                
                                if update_student_progress(target_work_id, final_text, conf):
                                    st.success("✅ บันทึกผลเรียบร้อย!")
                                    st.session_state['teacher_result'] = (final_text, conf)
                                else:
                                    st.error("บันทึกข้อมูลล้มเหลว")
                        else:
                            st.error("Model Loading Failed")
                else:
                    st.error(f"ไม่สามารถโหลดรูปภาพได้ (HTTP {response.status_code})")
                    st.write(f"URL: {image_url}")
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            if 'teacher_result' in st.session_state:
                res_text, res_conf = st.session_state['teacher_result']
                st.markdown(f"""
                    <div class="result-box">
                        <h3>ผลลัพธ์ AI</h3>
                        <h1 style="color: #D32F2F; font-size: 3rem; margin: 0;">{res_text}</h1>
                        <p>ความมั่นใจ: <strong>{res_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("❌ ปิดการแสดงผล"):
                    del st.session_state['teacher_result']
                    st.experimental_rerun()
            elif work_data.get('ai_result'):
                st.markdown(f"""
                    <div class="result-box" style="background:#f9f9f9; border-color:#ccc;">
                        <h4>ผลการตรวจเดิม</h4>
                        <h2 style="color: #555;">{work_data['ai_result']}</h2>
                        <p>ความมั่นใจ: {work_data['ai_confidence']}%</p>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("❌ ไม่พบข้อมูลงาน หรือไฟล์รูปภาพในฐานข้อมูล")

# ==========================================
# 🅱️ MODE 2: Gallery Browser (ไม่มี ID)
# ==========================================
else:
    st.write("---")
    c1, c2, c3 = st.columns([0.1, 3, 0.1])
    with c2:
        filter_option = st.radio("📂 เลือกดูข้อมูล (Gallery Mode):", ["ทั้งหมด", "ยังไม่ตรวจ"])

    image_list = get_image_list(filter_option)

    if len(image_list) > 0:
        id_list = [row[0] for row in image_list]
        if 'idx' not in st.session_state: st.session_state.idx = 0
        if st.session_state.idx >= len(id_list): st.session_state.idx = 0
        
        current_id = id_list[st.session_state.idx]
        data_row = get_image_data_gallery(current_id)
        
        if data_row:
            blob_data, saved_result, saved_conf = data_row
            image = Image.open(io.BytesIO(blob_data))
            
            col_img, col_act = st.columns([1, 1])
            with col_img: st.image(image, use_column_width=True)
            with col_act:
                if saved_result:
                    st.info(f"ผล: {saved_result} ({saved_conf}%)")
                else:
                    if st.button("ทำนายรูปนี้"):
                        label, conf = predict_image(image, model, class_names)
                        final_res = get_display_text(label)
                        st.success(f"ผลลัพธ์: {final_res} ({conf:.2f}%)")
        
        col_p, col_n = st.columns(2)
        with col_p: 
            if st.button("◀️ ย้อนกลับ"): 
                st.session_state.idx -= 1
                st.experimental_rerun()
        with col_n: 
            if st.button("ถัดไป ▶️"): 
                st.session_state.idx += 1
                st.experimental_rerun()
    else:
        st.info("ไม่มีข้อมูลรูปภาพใน Gallery")

st.markdown("<div style='text-align: center; margin-top: 50px; color: #aaa; font-size: 0.8rem;'>Hiragana Sensei AI System</div>", unsafe_allow_html=True)