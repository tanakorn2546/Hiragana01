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
import requests # 📦 [เพิ่ม] ต้องติดตั้ง pip install requests

# --- [Config] ธีมญี่ปุ่น ---
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_dir): os.makedirs(config_dir)
with open(config_path, "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#D32F2F"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF0F5"\ntextColor="#333333"\n')

st.set_page_config(page_title="Hiragana Sensei AI", page_icon="🇯🇵", layout="centered")

# --- CSS ตกแต่ง ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
    html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'Prompt', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 100%) !important; background-attachment: fixed !important; }
    div.block-container { background-color: rgba(255, 255, 255, 0.95) !important; border-radius: 30px; padding: 2rem; border-top: 5px solid #D32F2F; box-shadow: 0 15px 50px rgba(0,0,0,0.1); }
    h1 { color: #D32F2F !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- Database Connection ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

# --- [ใหม่] ฟังก์ชันอัปเดตตาราง Progress (สำหรับโหมดครู) ---
def update_student_progress(work_id, ai_result, ai_confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # อัปเดตลงตาราง progress แทน culantro_images
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (ai_result, float(ai_confidence), work_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB Error: {e}")
        return False

# --- ฟังก์ชันเดิม (สำหรับโหมดทั่วไป) ---
def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        if "ยังไม่ตรวจ" in filter_mode:
            sql = "SELECT id, image_name, prediction_result FROM culantro_images WHERE prediction_result IS NULL ORDER BY id ASC"
        else:
            sql = "SELECT id, image_name, prediction_result FROM culantro_images ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except: return []

def get_image_data(img_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT image_data, prediction_result, confidence FROM culantro_images WHERE id = %s", (img_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

# --- Model Loader ---
@st.cache_resource
def load_model():
    # File ID เดิมของคุณ
    file_id = '1ezDUsDxeabZX06ArdjtcWPk0uradYWDD' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path):
            model_name = local_path
        else:
            with st.spinner("📥 กำลังดาวน์โหลด Model..."):
                try:
                    gdown.download(url, model_name, quiet=False)
                except: return None
    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except: return None

def load_class_names():
    # Default Classes (ถ้าไม่มีไฟล์ json)
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

def get_hiragana_char(romaji):
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

# ==========================================
# 🔥 MAIN APPLICATION LOGIC
# ==========================================
model = load_model()
class_names = load_class_names()

st.markdown("<h1>Hiragana Sensei AI</h1>", unsafe_allow_html=True)

# 1. เช็คว่ามี Parameter ส่งมาจาก Teacher Dashboard หรือไม่
query_params = st.query_params
target_work_id = query_params.get("work_id", None)
target_image_url = query_params.get("image", None)

# ==========================================
# 🅰️ MODE 1: Teacher Direct Link (ตรวจงานจากลิงก์ครู)
# ==========================================
if target_work_id and target_image_url:
    st.info(f"📋 โหมดตรวจสอบงานนักเรียน (Work ID: {target_work_id})")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            # โหลดรูปจาก URL ที่ PHP ส่งมา
            response = requests.get(target_image_url, timeout=10)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                st.image(image, caption="รูปจากนักเรียน", use_column_width=True)
                
                # ปุ่มกดทำนาย
                if st.button("🔍 ตรวจสอบด้วย AI เดี๋ยวนี้", type="primary"):
                    if model:
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            preds = import_and_predict(image, model)
                            idx = np.argmax(preds)
                            conf = np.max(preds) * 100
                            
                            res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                            final_res = get_hiragana_char(res_code)
                            
                            # อัปเดตลง DB (ตาราง progress)
                            success = update_student_progress(target_work_id, final_res, conf)
                            
                            st.session_state['ai_result'] = final_res
                            st.session_state['ai_conf'] = conf
                            st.session_state['db_updated'] = success
                            st.experimental_rerun()
                    else:
                        st.error("ไม่พบ Model")
            else:
                st.error(f"ไม่สามารถโหลดรูปภาพได้ (Status: {response.status_code})")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดรูป: {e}")

    with col2:
        if 'ai_result' in st.session_state:
            res = st.session_state['ai_result']
            conf = st.session_state['ai_conf']
            
            st.markdown(f"""
            <div style="background:#e3f2fd; padding:20px; border-radius:15px; text-align:center; border:2px dashed #2196f3;">
                <h3>ผลลัพธ์ AI</h3>
                <h1 style="color:#1565c0; font-size:3rem; margin:0;">{res}</h1>
                <p>ความมั่นใจ: <b>{conf:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('db_updated', False):
                st.success("✅ บันทึกผลลง Database เรียบร้อยแล้ว!")
            else:
                st.error("❌ บันทึกผลล้มเหลว")
                
            if st.button("❌ ปิดการแสดงผล"):
                del st.session_state['ai_result']
                st.experimental_rerun()

# ==========================================
# 🅱️ MODE 2: Normal Batch View (ดูจากตาราง culantro_images)
# ==========================================
else:
    st.markdown("---")
    filter_option = st.radio("📂 เลือกดูข้อมูล (Database: culantro_images):", ["ทั้งหมด", "ยังไม่ตรวจ"])
    image_list = get_image_list(filter_option)

    if len(image_list) > 0:
        id_list = [row[0] for row in image_list]
        if 'idx' not in st.session_state: st.session_state.idx = 0
        
        # วนลูป Index
        if st.session_state.idx >= len(id_list): st.session_state.idx = 0
        current_id = id_list[st.session_state.idx]
        
        data_row = get_image_data(current_id)
        if data_row:
            blob_data, saved_result, saved_conf = data_row
            image = Image.open(io.BytesIO(blob_data))
            
            c1, c2 = st.columns([1, 1])
            with c1: st.image(image, width=300)
            with c2: 
                st.write(f"**ID:** {current_id}")
                if saved_result:
                    st.success(f"ผล: {saved_result} ({saved_conf}%)")
                else:
                    if st.button("ทำนายรูปนี้"):
                        preds = import_and_predict(image, model)
                        # ... (Logic ทำนายแบบเดิมสำหรับตาราง culantro_images) ...
                        st.write("ผลการทำนายจะแสดงที่นี่ (Logic เดิม)")
        
        # ปุ่ม Next/Prev
        col_p, col_n = st.columns(2)
        with col_p: 
            if st.button("Previous"): 
                st.session_state.idx -= 1
                st.experimental_rerun()
        with col_n: 
            if st.button("Next"): 
                st.session_state.idx += 1
                st.experimental_rerun()
    else:
        st.info("ไม่มีข้อมูลรูปภาพในระบบ")