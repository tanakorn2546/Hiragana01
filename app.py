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
    page_title="Hiragana Sensei AI (MobileNetV2)",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS Styling ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Zen+Maru+Gothic:wght@700&display=swap');
        :root { --japan-red: #D72638; }
        html, body, [class*="css"] { font-family: 'Prompt', sans-serif !important; }
        .stApp {
            background: linear-gradient(180deg, #d4fcff 0%, #fff 60%, #fff 100%);
            background-attachment: fixed;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(15px);
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
        .stButton button { border-radius: 12px !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Functions ---

def get_connection():
    # ⚠️ ใช้ Credentials เดียวกับ PHP
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg", 
        database="cedubruc_hiragana_app",
        connect_timeout=10
    )

def get_work_list(filter_mode):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # ดึงรายการที่มีรูปภาพ (image_data ไม่เป็น NULL)
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
    except mysql.connector.Error as err:
        st.error(f"❌ Database Error: {err}")
        return []

def get_work_data(work_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # ดึง BLOB data ออกมา
        cursor.execute("SELECT image_data, ai_result, ai_confidence, char_code FROM progress WHERE id = %s", (work_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

def update_database(work_id, result, confidence):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        conf_val = float(confidence) if confidence else 0.0
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, conf_val, work_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

def get_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        data = cursor.fetchone()
        conn.close()
        return data
    except: return 0, 0

# --- Fix MobileNetV2 & Model Loading ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    model_name = 'best_hiragana_smart_model.h5'
    file_id = '1gcqpDUAdgGTeh1dW9s3ODsvSwNvmQLXs' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        try:
            with st.spinner("⬇️ กำลังดาวน์โหลดโมเดลจาก Server..."):
                gdown.download(url, model_name, quiet=False)
        except:
            st.error("ไม่สามารถดาวน์โหลดโมเดลได้")
            return None
    
    try:
        return tf.keras.models.load_model(
            model_name, 
            compile=False,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D} 
        )
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

def load_class_names():
    return [
        'a', 'chi', 'e', 'fu', 'ha', 'he', 'hi', 'ho', 'i', 'ka', 'ke', 'ki', 'ko', 'ku', 
        'ma', 'me', 'mi', 'mo', 'mu', 'n', 'na', 'ne', 'ni', 'no', 'nu', 'o', 
        'ra', 're', 'ri', 'ro', 'ru', 'sa', 'se', 'shi', 'so', 'su', 'ta', 'te', 'to', 'tsu', 
        'u', 'wa', 'wo', 'ya', 'yo', 'yu'
    ]

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    if image.mode != "L": image = image.convert("L")
    image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# --- 4. Main UI ---
model = load_model()
class_names = load_class_names()

with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")
    if st.button("รีเฟรชข้อมูล"):
        st.rerun()

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือด้วย MobileNetV2</div>', unsafe_allow_html=True)

query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    if target_work_id:
        st.info(f"🔍 กำลังดูงาน ID: {target_work_id}")
        filter_option = "ทั้งหมด (All)"
    else:
        filter_option = st.selectbox("โหมดการทำงาน", ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"], label_visibility="collapsed")

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0
    
    current_id = id_list[st.session_state.current_index]
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.caption(f"Work ID: {current_id} | ลำดับที่ {st.session_state.current_index + 1}/{len(id_list)}")

    # --- ดึงข้อมูล BLOB มาแปลง ---
    data_row = get_work_data(current_id)
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        
        # แปลง BLOB -> Image
        try: 
            image = Image.open(io.BytesIO(blob_data))
        except Exception as e: 
            image = None
            st.error(f"Image Error: {e}")

        if image:
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
                    <div class="result-card">
                        <div style="font-size:1.2rem; color:#555;">{romaji_part}</div>
                        <div class="big-char">{char_part}</div>
                        <div style="color:green; font-weight:bold;">มั่นใจ: {saved_conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 ตรวจใหม่", type="secondary", key=f"rechk_{current_id}"):
                        update_database(current_id, None, 0)
                        st.rerun()
                else:
                    st.markdown("""
                    <div class="result-card" style="border: 2px dashed #ffcdd2; background:#fffaf0;">
                        <h1 style="color:#ef5350; opacity:0.5;">⏳</h1><p style="color:#888;">รอผลการวิเคราะห์...</p>
                    </div>""", unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("✨ วิเคราะห์ด้วย AI", type="primary", key=f"ai_{current_id}"):
                        if model:
                            with st.spinner("AI กำลังคิด..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    
                                    # Hiragana Map
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
                                    
                                    if update_database(current_id, final_res, conf):
                                        st.success("บันทึกแล้ว!")
                                        time.sleep(0.5)
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Prediction Error: {e}")
                        else:
                            st.error("Model not loaded properly.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    c_prev, c_space, c_next = st.columns([1, 0.2, 1])
    with c_prev:
        if st.button("⬅️ ก่อนหน้า", use_container_width=True):
            st.session_state.current_index -= 1
            if st.session_state.current_index < 0:
                st.session_state.current_index = len(id_list) - 1
            st.rerun()
            
    with c_next:
        if st.button("ถัดไป ➡️", use_container_width=True):
            st.session_state.current_index += 1
            if st.session_state.current_index >= len(id_list):
                st.session_state.current_index = 0
            st.rerun()

else:
    st.info("🎉 ไม่พบข้อมูลงานในโหมดนี้")

st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" 
       style="color:#D72638; text-decoration:none; font-weight:bold; background:white; padding:10px 20px; border-radius:30px; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
       🏠 กลับสู่หน้าอัปโหลด
    </a>
</div>
""", unsafe_allow_html=True)