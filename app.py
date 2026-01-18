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

# --- 3. Database (แก้ไขให้ตรงกับ db.php) ---
def init_connection():
    return mysql.connector.connect(
        host="localhost",          # แก้ไขให้ตรงกับ Server ของคุณ
        user="root",               # แก้ไขให้ตรงกับ User ของคุณ
        password="",               # แก้ไขรหัสผ่าน
        database="cedubruc_hiragana_app"
    )

# ดึงรายการงานจากตาราง progress
def get_work_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        # เลือกเฉพาะที่มีข้อมูลภาพ (image_data)
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

# ดึงข้อมูลภาพ (BLOB) ของงานชิ้นนั้น
def get_work_data(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ดึง image_data, ai_result, ai_confidence, char_code
        cursor.execute("SELECT image_data, ai_result, ai_confidence, char_code FROM progress WHERE id = %s", (work_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

# อัปเดตผล AI กลับลงตาราง progress
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

# --- 4. Smart Model Loader ---
if hasattr(st, 'cache_resource'): 
    cache_decorator = st.cache_resource
else: 
    cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    file_id = '1UmI9gbQZ80sBh3Yj78quqKlQ6SZGkBUe' 
    model_name = 'best_hiragana_mobilenetv2.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path):
            model_name = local_path
        else:
            st.warning("📥 กำลังดาวน์โหลด Model...")
            try:
                gdown.download(url, model_name, quiet=False)
                st.success("✅ โหลดโมเดลสำเร็จ")
            except Exception as e:
                st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
                return None

    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except Exception as e:
        st.error(f"❌ ไฟล์โมเดลเสียหาย: {e}")
        return None

# Load Class Names (ตามลำดับที่เทรนมา)
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
        ระบบตรวจจับและจำแนกตัวอักษรฮิรางานะด้วย AI (เชื่อมต่อฐานข้อมูล Integrated)
    </p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🎯 ส่วนรับค่า Deep Link (?work_id=...) จากหน้า Teacher
# ---------------------------------------------------------
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    if target_work_id:
        filter_option = "ทั้งหมด (All)"
        st.info(f"🔍 กำลังตรวจสอบงานรหัส ID: {target_work_id}")
    else:
        filter_option = st.radio(
            "📂 เลือกดูข้อมูล:", 
            ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"], 
        )

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    # Logic การกระโดดไปยังรูปที่ต้องการ (Deep Linking)
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        
    # ป้องกัน Index error
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #333; margin-bottom: 15px; font-weight: normal; font-size: 1.1rem; background: #FFEBEE; padding: 10px; border-radius: 10px;'>📝 งานที่ {st.session_state.current_index + 1} / {len(id_list)} (ID: {current_id})</div>", unsafe_allow_html=True)

    data_row = get_work_data(current_id)
    
    if data_row:
        # data_row = (image_data, ai_result, ai_confidence, char_code)
        blob_data, saved_result, saved_conf, true_label = data_row
        
        # แปลง Blob (Binary) เป็น Image Object
        try:
            image = Image.open(io.BytesIO(blob_data))
        except Exception as e:
            st.error("❌ ไฟล์รูปภาพเสียหาย ไม่สามารถเปิดได้")
            image = None

        if image:
            col_img, col_act = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption=f"โจทย์: {true_label}", use_column_width=True)
            
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
                                    
                                    if idx < len(class_names):
                                        res_code = class_names[idx]
                                    else:
                                        res_code = "Unknown"

                                    # Map Romaji to Hiragana (Display)
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
                                    
                                    # บันทึกลงตาราง progress
                                    update_database(current_id, final_res, conf)
                                    st.success(f"อ่านได้ว่า: {final_res}")
                                    time.sleep(0.5)
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"💥 เกิดข้อผิดพลาด: {e}")
                        else:
                            st.error("ไม่พบโมเดล")

    # --- ปุ่มนำทาง ---
    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    
    with c_prev:
        if st.session_state.current_index > 0:
            if st.button("◀️ ย้อนกลับ"):
                st.session_state.current_index -= 1
                st.rerun()
            
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                st.rerun()
        else:
             if st.button("🔄 กลับไปรูปแรก"):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.warning("ยังไม่มีงานส่งเข้ามาในระบบ")

# --- Link กลับเว็บหลัก ---
# เปลี่ยน URL นี้ให้ตรงกับที่อยู่เว็บ PHP ของคุณ
base_url = "http://localhost/teacher.php" 

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <a href="{base_url}" target="_self" class="custom-home-btn">
            🏠 กลับสู่ห้องพักครู
        </a>
    </div>
    <div class="footer-credit">
        <strong>Hiragana Image Classification System V.2.0 (Integrated)</strong>
    </div>
""", unsafe_allow_html=True)