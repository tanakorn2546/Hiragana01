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

# --- [Config] ---
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_dir): os.makedirs(config_dir)
with open(config_path, "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#D32F2F"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF0F5"\ntextColor="#333333"\n')

st.set_page_config(page_title="Hiragana Sensei AI", page_icon="🇯🇵", layout="centered")

def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
        html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'Prompt', sans-serif !important; color: #333333 !important; }
        .stApp { background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 100%) !important; background-attachment: fixed !important; background-size: cover !important; }
        div.block-container { background-color: rgba(255, 255, 255, 0.95) !important; border-radius: 30px !important; padding: 2rem !important; margin-top: 2rem !important; box-shadow: 0 15px 50px rgba(0,0,0,0.1) !important; border-top: 5px solid #D32F2F; }
        .app-header-icon { font-size: 80px !important; background: radial-gradient(circle, #ffcdd2 0%, #ffffff 100%) !important; width: 140px; height: 140px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px auto; box-shadow: 0 10px 25px rgba(211, 47, 47, 0.2); border: 5px solid #ffffff; }
        h1 { text-align: center; color: #D32F2F !important; font-weight: 800 !important; font-size: 2.2rem !important; }
        .custom-home-btn { background: linear-gradient(135deg, #424242 0%, #212121 100%); color: #ffffff !important; text-decoration: none; padding: 0.8rem 2rem; border-radius: 15px; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: all 0.3s ease; text-align: center; width: 100%; }
    </style>
    """, unsafe_allow_html=True)
local_css()

# --- Database ---
def init_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="cedubruc_hiragana_app"
    )

def get_work_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ใช้ตาราง progress แทน culantro_images
        # ดึงเฉพาะที่มีรูป (image_data IS NOT NULL)
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
        st.error(f"❌ DB Error: {e}")
        return []

def get_work_data(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ดึง image_data (BLOB)
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

# --- Model ---
if hasattr(st, 'cache_resource'): cache_decorator = st.cache_resource
else: cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    file_id = '1UmI9gbQZ80sBh3Yj78quqKlQ6SZGkBUe' 
    model_name = 'best_hiragana_mobilenetv2.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path): model_name = local_path
        else:
            st.warning("📥 Downloading Model...")
            try: gdown.download(url, model_name, quiet=False)
            except: return None
    try: return tf.keras.models.load_model(model_name, compile=False)
    except: return None

def load_class_names():
    # เรียงตาม Index ที่โมเดล Train มา (ต้องตรงกับ class_indices.json)
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

def import_and_predict(image_data, model):
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    if image.mode != "RGB": image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- UI Logic ---
model = load_model()
class_names = load_class_names()

st.markdown("""<div class='app-header-icon'>🇯🇵</div><h1>Hiragana Sensei AI</h1>""", unsafe_allow_html=True)

# รับ Query Param (work_id) จาก Teacher Dashboard
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    if not target_work_id:
        filter_option = st.radio("📂 เลือกดูข้อมูล:", ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"])
    else:
        filter_option = "ทั้งหมด (All)"
        st.info(f"🔍 กำลังตรวจสอบงาน ID: {target_work_id}")

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    # ถ้ามี target_work_id ให้กระโดดไปที่ index นั้น
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        
    if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    st.markdown(f"<div style='text-align: center; margin-bottom: 15px; font-size: 1.1rem;'>📝 งานที่ {st.session_state.current_index + 1} / {len(id_list)} (ID: {current_id})</div>", unsafe_allow_html=True)

    data_row = get_work_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf, char_code = data_row
        # แปลง Blob เป็น Image
        try:
            image = Image.open(io.BytesIO(blob_data))
        except:
            st.error("ไม่สามารถอ่านไฟล์รูปภาพได้")
            image = None

        if image:
            col_img, col_act = st.columns([1, 1])
            with col_img:
                st.image(image, caption=f"โจทย์: {char_code}", use_column_width=True)
            
            with col_act:
                st.markdown("### ผลลัพธ์ AI")
                if saved_result:
                    st.markdown(f"""
                        <div style="background-color: #FFEBEE; padding: 20px; border-radius: 15px; border: 2px solid #D32F2F; margin-bottom: 20px; text-align: center;">
                            <h1 style="color: #D32F2F !important; margin: 0; font-size: 3rem; font-weight: 800;">{saved_result}</h1>
                            <p>Confidence: <strong>{saved_conf:.2f}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("🔄 ตรวจสอบใหม่"):
                        update_database(current_id, None, 0)
                        st.rerun()
                else:
                    st.info("⚠️ ยังไม่ได้ระบุตัวอักษร")
                    if st.button("🇯🇵 อ่านลายมือ"):
                        if model:
                            with st.spinner("AI กำลังวิเคราะห์..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
                                    
                                    # Map Romaji -> Hiragana + Romaji
                                    # (ใส่ Map เดิมของคุณตรงนี้ได้เลย)
                                    hiragana_map = {'a':'あ (a)', 'i':'い (i)', 'u':'う (u)', 'ka':'か (ka)', 'n':'ん (n)'} # (ย่อเพื่อประหยัดที่)
                                    # ... ใส่ Map ให้ครบตามเดิม ...
                                    
                                    # Fallback ง่ายๆ ถ้าไม่มีใน map
                                    final_res = hiragana_map.get(res_code, res_code)
                                    
                                    update_database(current_id, final_res, conf)
                                    st.success(f"อ่านได้ว่า: {final_res}")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e: st.error(f"Error: {e}")
                        else: st.error("Model Error")

    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    with c_prev:
        if st.session_state.current_index > 0:
            if st.button("◀️ ก่อนหน้า"):
                st.session_state.current_index -= 1
                st.rerun()
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                st.rerun()
else:
    st.warning("ไม่มีงานค้างอยู่ในระบบ")

# ลิงก์กลับ Teacher Dashboard
teacher_url = "http://localhost/teacher.php" # แก้เป็น URL จริงของคุณ
st.markdown(f"""<div style="text-align: center; margin-top: 30px;"><a href="{teacher_url}" class="custom-home-btn">🏠 กลับห้องพักครู</a></div>""", unsafe_allow_html=True)