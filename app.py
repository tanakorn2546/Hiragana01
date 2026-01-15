import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import json
import gdown # 📦 ต้องติดตั้ง pip install gdown ก่อนใช้งาน

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
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #e57373 0%, #D32F2F 100%) !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 25px !important;
            color: #ffffff !important; 
        }
        div[role="radiogroup"] label:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 10px rgba(211, 47, 47, 0.3) !important;
        }
        .stRadio > label {
            color: #D32F2F !important;
            font-weight: 800 !important;
            font-size: 1.3rem !important;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #ef5350 0%, #c62828 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 15px rgba(198, 40, 40, 0.4) !important;
        }
        div[data-testid="stImage"] > img {
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            border: 2px solid #ffcdd2;
        }
        h1 { 
            text-align: center; color: #D32F2F !important; 
            font-weight: 800 !important; font-size: 2.2rem !important;
        }
        .custom-home-btn {
            background: linear-gradient(135deg, #424242 0%, #212121 100%);
            color: #ffffff !important;
            text-decoration: none;
            padding: 0.8rem 2rem;
            border-radius: 15px;
            display: inline-block;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            text-align: center;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database ---
def init_connection():
    return mysql.connector.connect(
        host="localhost",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        table_name = "culantro_images" 
        
        if "ยังไม่ตรวจ" in filter_mode:
            sql = f"SELECT id, image_name, prediction_result FROM {table_name} WHERE prediction_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = f"SELECT id, image_name, prediction_result FROM {table_name} WHERE prediction_result IS NOT NULL ORDER BY id DESC"
        else:
            sql = f"SELECT id, image_name, prediction_result FROM {table_name} ORDER BY id DESC"
        
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return []

def get_image_data(img_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        table_name = "culantro_images"
        cursor.execute(f"SELECT image_data, prediction_result, confidence FROM {table_name} WHERE id = %s", (img_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

def update_database(img_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        table_name = "culantro_images"
        sql = f"UPDATE {table_name} SET prediction_result = %s, confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), img_id))
        conn.commit()
        conn.close()
        return True
    except: return False

# --- 4. Smart Model Loader (แก้ไข: รองรับ Google Drive) ---
if hasattr(st, 'cache_resource'): cache_decorator = st.cache_resource
else: cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    # -------------------------------------------------------------
    # 🔥 [แก้ไข] ใส่ Google Drive File ID ของคุณตรงนี้ (สำคัญ!) 🔥
    file_id = '1ezDUsDxeabZX06ArdjtcWPk0uradYWDD' 
    # -------------------------------------------------------------
    
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # 1. เช็คว่ามีไฟล์อยู่ในโฟลเดอร์ปัจจุบันหรือไม่
    if not os.path.exists(model_name):
        # 2. ถ้าไม่มี ลองเช็คในโฟลเดอร์ saved_models
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path):
            model_name = local_path
        else:
            # 3. ถ้าไม่มีเลย ให้ดาวน์โหลดจาก Google Drive
            st.warning("📥 กำลังดาวน์โหลด Model จาก Google Drive... (ใช้เวลาสักครู่)")
            try:
                gdown.download(url, model_name, quiet=False)
                st.success("✅ ดาวน์โหลด Model สำเร็จ!")
            except Exception as e:
                st.error(f"❌ ดาวน์โหลดไม่สำเร็จ: {e} (ตรวจสอบ File ID และ Permission 'Anyone with the link')")
                return None

    # โหลด Model
    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except Exception as e:
        st.error(f"❌ ไฟล์โมเดลเสียหาย: {e}")
        return None

# --- Smart Class Loader ---
def load_class_names():
    json_name = 'class_indices.json'
    
    possible_paths = [
        json_name,
        os.path.join('saved_models', json_name)
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if found_path:
        with open(found_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        sorted_classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        return sorted_classes
    else:
        st.warning("⚠️ ไม่พบไฟล์ class_indices.json ใช้ค่า Default")
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

# ฟังก์ชันทำนายผล
def import_and_predict(image_data, model):
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 5. Main UI ---
model = load_model()
class_names = load_class_names()

st.markdown("""
    <div class='app-header-icon'>🇯🇵</div>
    <h1>Hiragana Sensei AI</h1>
    <p style='text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1rem;'>
        ระบบตรวจจับและจำแนกตัวอักษรฮิรางานะด้วย AI (MobileNetV2)
    </p>
""", unsafe_allow_html=True)

# --- ตัวกรอง ---
c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    filter_option = st.radio(
        "📂 เลือกดูข้อมูล:", 
        ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"], 
    )

image_list = get_image_list(filter_option)

if len(image_list) > 0:
    id_list = [row[0] for row in image_list]
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #333; margin-bottom: 15px; font-weight: normal; font-size: 1.1rem; background: #FFEBEE; padding: 10px; border-radius: 10px;'>📝 รูปที่ {st.session_state.current_index + 1} / {len(id_list)} (ID: {current_id})</div>", unsafe_allow_html=True)

    data_row = get_image_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf = data_row
        image = Image.open(io.BytesIO(blob_data))
        
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            st.image(image, use_column_width=True)
        
        with col_act:
            st.markdown("### ผลลัพธ์ AI")
            
            if saved_result:
                st.markdown(f"""
                    <div style="background-color: #FFEBEE; padding: 20px; border-radius: 15px; border: 2px solid #D32F2F; margin-bottom: 20px; text-align: center;">
                        <h1 style="color: #D32F2F !important; margin: 0; font-size: 3rem; font-weight: 800;">{saved_result}</h1>
                        <p style="margin-top: 10px; font-size: 1rem; color: #555;">ความมั่นใจ: <strong>{saved_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 ตรวจสอบใหม่"):
                    update_database(current_id, None, 0)
                    st.experimental_rerun()
            
            else:
                st.info("⚠️ ยังไม่ได้ระบุตัวอักษร")
                if st.button("🇯🇵 อ่านตัวอักษรนี้"):
                    if model:
                        with st.spinner("AI กำลังอ่านลายมือ..."):
                            try:
                                preds = import_and_predict(image, model)
                                idx = np.argmax(preds)
                                conf = np.max(preds) * 100
                                
                                if idx < len(class_names):
                                    res_code = class_names[idx]
                                else:
                                    res_code = "Unknown"

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
                                st.success(f"อ่านได้ว่า: {final_res}")
                                time.sleep(0.5)
                                st.experimental_rerun()

                            except Exception as e:
                                st.error(f"💥 เกิดข้อผิดพลาด: {e}")
                    else:
                        st.error("ไม่พบโมเดล")

                # --- Batch Process ---
                if "ยังไม่ตรวจ" in filter_option:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    if st.button(f"⚡ อ่านลายมือทั้งหมด ({len(image_list)} รูป)"):
                         pass 

    # --- ปุ่มนำทาง ---
    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    
    with c_prev:
        if st.session_state.current_index > 0:
            if st.button("◀️ ย้อนกลับ"):
                st.session_state.current_index -= 1
                st.experimental_rerun()
            
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                st.experimental_rerun()
        else:
             if st.button("🔄 กลับไปรูปแรก"):
                st.session_state.current_index = 0
                st.experimental_rerun()

else:
    st.warning("ยังไม่มีข้อมูลรูปภาพในระบบ")

# --- Link กลับเว็บหลัก ---
base_url = "http://www.your-school-website.com/" 
full_url = base_url

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <a href="{full_url}" target="_blank" class="custom-home-btn">
            🏠 กลับสู่หน้าบทเรียน
        </a>
    </div>
    <div class="footer-credit">
        <strong>Hiragana Image Classification System V.2.0</strong>
    </div>
""", unsafe_allow_html=True)