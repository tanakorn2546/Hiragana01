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

# --- [Config] ธีมญี่ปุ่น ---
config_dir = ".streamlit"
if not os.path.exists(config_dir): os.makedirs(config_dir)
with open(os.path.join(config_dir, "config.toml"), "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#D32F2F"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF0F5"\ntextColor="#333333"\n')

st.set_page_config(page_title="Hiragana Sensei AI", page_icon="🇯🇵", layout="centered")

# --- CSS ตกแต่ง ---
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
        }
        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 30px !important;
            padding: 2rem;
            box-shadow: 0 15px 50px rgba(0,0,0,0.1) !important;
            border-top: 5px solid #D32F2F;
        }
        .app-header-icon {
            font-size: 80px; background: radial-gradient(circle, #ffcdd2 0%, #ffffff 100%);
            width: 140px; height: 140px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; margin: 0 auto 15px auto;
            box-shadow: 0 10px 25px rgba(211, 47, 47, 0.2); border: 5px solid #ffffff;
        }
        .result-teacher-box {
            background: #FFEBEE; padding: 20px; border-radius: 15px; 
            border: 2px solid #D32F2F; text-align: center; margin-top: 20px;
        }
        .custom-home-btn {
            background: linear-gradient(135deg, #424242 0%, #212121 100%); color: #ffffff !important;
            text-decoration: none; padding: 0.8rem 2rem; border-radius: 15px; display: inline-block;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: all 0.3s ease; width: 100%; text-align: center;
        }
        h1 { text-align: center; color: #D32F2F; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Database ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

# ฟังก์ชันอัปเดตของ Teacher Mode (ตาราง progress)
def update_student_progress(work_id, ai_result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (ai_result, float(confidence), work_id))
        conn.commit()
        conn.close()
        return True
    except: return False

# ฟังก์ชันอัปเดตของโหมดเดิม (ตาราง culantro_images)
def update_database(img_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        sql = "UPDATE culantro_images SET prediction_result = %s, confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), img_id))
        conn.commit()
        conn.close()
        return True
    except: return False

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        table = "culantro_images"
        if "ยังไม่ตรวจ" in filter_mode: sql = f"SELECT id, image_name, prediction_result FROM {table} WHERE prediction_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode: sql = f"SELECT id, image_name, prediction_result FROM {table} WHERE prediction_result IS NOT NULL ORDER BY id DESC"
        else: sql = f"SELECT id, image_name, prediction_result FROM {table} ORDER BY id DESC"
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

# --- 🛠️ FIX MODEL: แก้ปัญหา Version เก่า ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None) # ลบคำสั่งที่ Server ไม่รู้จักทิ้ง
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    file_id = '1ezDUsDxeabZX06ArdjtcWPk0uradYWDD' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        try: gdown.download(url, model_name, quiet=False)
        except: return None

    try:
        # ยัดไส้ Class แก้บั๊กเข้าไป
        return tf.keras.models.load_model(model_name, compile=False, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
    except: return None

def import_and_predict(image_data, model):
    class_names = ['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho', 'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo', 'n']
    hiragana_map = {'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)', 'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)', 'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)', 'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)', 'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)', 'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)', 'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)', 'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)', 'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)', 'wa': 'わ (wa)', 'wo': 'を (wo)', 'n': 'ん (n)'}
    
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS).convert("RGB")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(image, dtype=np.float32))
    data = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(data)
    idx = np.argmax(preds)
    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
    return hiragana_map.get(res_code, res_code), np.max(preds) * 100

# --- MAIN UI ---
model = load_model()
st.markdown("<div class='app-header-icon'>🇯🇵</div><h1>Hiragana Sensei AI</h1>", unsafe_allow_html=True)

# 1. เช็คว่าเปิดมาจาก Teacher หรือไม่?
try:
    qp = st.query_params
    work_id = qp.get("work_id")
    img_url = qp.get("image_url")
except:
    # Fallback สำหรับ Streamlit รุ่นเก่า
    qp = st.experimental_get_query_params()
    work_id = qp.get("work_id", [None])[0]
    img_url = qp.get("image_url", [None])[0]

if work_id and img_url:
    # ================================
    # 🎯 TEACHER MODE (โหมดตรวจงาน)
    # ================================
    st.markdown(f"<h3 style='text-align:center; color:#555;'>📋 กำลังตรวจงาน ID: {work_id}</h3>", unsafe_allow_html=True)
    try:
        # หลอก Server ว่าเป็น Chrome
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'Referer': 'https://www.cedubru.com/'}
        response = requests.get(img_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            col1, col2 = st.columns([1, 1.2])
            with col1: st.image(img, caption="รูปจากนักเรียน", use_column_width=True)
            with col2:
                if model:
                    with st.spinner("AI กำลังวิเคราะห์..."):
                        result, conf = import_and_predict(img, model)
                        st.markdown(f"""<div class="result-teacher-box"><h1 style="color:#D32F2F;margin:0;font-size:3rem;">{result}</h1><p style="color:#555;">ความมั่นใจ: <strong>{conf:.2f}%</strong></p></div>""", unsafe_allow_html=True)
                        if st.button("💾 บันทึกผลการตรวจลงระบบ", type="primary", use_container_width=True):
                            if update_student_progress(work_id, result, conf):
                                st.success("✅ บันทึกเรียบร้อย!"); st.balloons()
                            else: st.error("บันทึกผิดพลาด")
                else: st.error("Model Error")
        else: st.error(f"❌ โหลดรูปไม่ได้ (Code: {response.status_code})")
    except Exception as e: st.error(f"Error: {e}")

else:
    # ================================
    # 📂 STANDARD MODE (โหมดดู Database เดิม)
    # ================================
    st.markdown("<p style='text-align: center; color: #555;'>โหมด: คลังข้อมูลรูปภาพ (culantro_images)</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.1, 3, 0.1])
    with c2: filter_option = st.radio("📂 เลือกดูข้อมูล:", ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"])
    
    image_list = get_image_list(filter_option)
    if len(image_list) > 0:
        id_list = [row[0] for row in image_list]
        if 'current_index' not in st.session_state: st.session_state.current_index = 0
        if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0
        current_id = id_list[st.session_state.current_index]
        
        st.markdown("---")
        st.markdown(f"<div style='text-align: center; background: #FFEBEE; padding: 10px; border-radius: 10px;'>📝 รูปที่ {st.session_state.current_index + 1} / {len(id_list)} (ID: {current_id})</div>", unsafe_allow_html=True)
        
        data_row = get_image_data(current_id)
        if data_row:
            blob_data, saved_result, saved_conf = data_row
            image = Image.open(io.BytesIO(blob_data))
            col_img, col_act = st.columns([1, 1])
            with col_img: st.image(image, use_column_width=True)
            with col_act:
                if saved_result:
                    st.markdown(f"""<div class="result-teacher-box"><h1 style="color:#D32F2F;margin:0;font-size:3rem;">{saved_result}</h1><p>ความมั่นใจ: {saved_conf:.2f}%</p></div>""", unsafe_allow_html=True)
                    if st.button("🔄 ตรวจสอบใหม่"):
                        update_database(current_id, None, 0); st.experimental_rerun()
                else:
                    st.info("⚠️ ยังไม่ระบุ")
                    if st.button("🇯🇵 อ่านตัวอักษรนี้"):
                        if model:
                            with st.spinner("Thinking..."):
                                result, conf = import_and_predict(image, model)
                                update_database(current_id, result, conf)
                                st.success(f"อ่านได้ว่า: {result}"); time.sleep(0.5); st.experimental_rerun()
                        else: st.error("Model Error")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c_prev, c_empty, c_next = st.columns([1, 0.2, 1])
        if c_prev.button("◀️ ย้อนกลับ") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1; st.experimental_rerun()
        if c_next.button("ถัดไป ▶️") and st.session_state.current_index < len(id_list) - 1:
            st.session_state.current_index += 1; st.experimental_rerun()
            
    else: st.warning("ไม่มีข้อมูลรูปภาพ")

    st.markdown(f"""<div style="text-align: center; margin-top: 30px;"><a href="http://www.your-school-website.com/" target="_blank" class="custom-home-btn">🏠 กลับสู่หน้าบทเรียน</a></div>""", unsafe_allow_html=True)