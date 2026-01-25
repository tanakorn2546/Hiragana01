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
        /* Decorations */
        .glass-card {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px);
            border-radius: 20px; border: 2px solid white; padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; position: relative; z-index: 10;
        }
        .result-card {
            background: white; border-radius: 15px; padding: 20px; text-align: center;
            border-top: 5px solid var(--japan-red); box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .big-char { font-size: 4rem; color: var(--japan-red); font-weight: bold; line-height: 1.2; font-family: 'Zen Maru Gothic'; }
        .hero-title {
            font-family: 'Zen Maru Gothic', sans-serif; font-size: 3rem; color: var(--japan-red);
            text-align: center; text-shadow: 2px 2px 0px white; margin-bottom: 0;
        }
        .hero-subtitle { text-align: center; color: #555; margin-bottom: 30px; }
        
        /* Button Styling */
        .stButton button { 
            border-radius: 50px !important; 
            font-weight: 600 !important; 
            border: none !important; 
            padding: 10px 24px !important;
            transition: all 0.3s !important;
        }
        .stButton button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Configuration ---
# 🔧 ตั้งค่า Mapping ให้ตรงกับ Database ของคุณ
TABLE_CONFIG = {
    "progress": {
        "pk": "id",
        "label_col": "char_code",       # ชื่อคอลัมน์เฉลย (Label)
        "image_col": "image_data",      # ชื่อคอลัมน์รูปภาพ (BLOB)
        "result_col": "ai_result",      # ชื่อคอลัมน์เก็บผล AI
        "conf_col": "ai_confidence"     # ชื่อคอลัมน์เก็บ % ความมั่นใจ
    },
    "quiz_submissions": {
        "pk": "id",
        "label_col": "char_label",      # ตรวจสอบชื่อคอลัมน์ใน DB ให้ถูกต้อง
        "image_col": "image_data",
        "result_col": "ai_result",      # *ต้องมีคอลัมน์นี้ใน Table quiz_submissions
        "conf_col": "ai_confidence"     # *ต้องมีคอลัมน์นี้ใน Table quiz_submissions
    }
}

def init_connection():
    # ⚠️ ใช้ st.secrets ในการใช้งานจริงเพื่อความปลอดภัย
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app" 
    )

def get_work_data(target_id, table_name="progress"):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cfg = TABLE_CONFIG.get(table_name)
        
        # Query ดึงข้อมูลรูปและผลลัพธ์เดิม
        sql = f"""
            SELECT {cfg['image_col']}, {cfg['result_col']}, {cfg['conf_col']}, {cfg['label_col']} 
            FROM {table_name} WHERE {cfg['pk']} = %s
        """
        cursor.execute(sql, (target_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

def update_database(target_id, table_name, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cfg = TABLE_CONFIG.get(table_name)
        
        sql = f"UPDATE {table_name} SET {cfg['result_col']} = %s, {cfg['conf_col']} = %s WHERE {cfg['pk']} = %s"
        cursor.execute(sql, (result, float(confidence), target_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Update Error: {e}")
        return False

# --- 4. Model Loading ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    # โหลดโมเดลจาก Google Drive (MobileNetV2)
    model_name = 'hiragana_model_best2.h5'
    file_id = '1Q0CFCi_0KFwbes3DhQV4LLVTciP8VmrK' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        try:
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
        st.error(f"❌ Model Error: {e}")
        return None

def load_class_names():
    return [
        'a', 'chi', 'e', 'fu', 'ha', 'he', 'hi', 'ho', 'i', 
        'ka', 'ke', 'ki', 'ko', 'ku', 'ma', 'me', 'mi', 'mo', 'mu', 
        'n', 'na', 'ne', 'ni', 'no', 'nu', 'o', 
        'ra', 're', 'ri', 'ro', 'ru', 
        'sa', 'se', 'shi', 'so', 'su', 
        'ta', 'te', 'to', 'tsu', 
        'u', 'wa', 'wo', 'ya', 'yo', 'yu'
    ]

# --- 5. Prediction Logic ---
def import_and_predict(image_data, model):
    # ปรับขนาดภาพให้เข้ากับ MobileNetV2 (224x224)
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    if image.mode != "L": image = image.convert("L") # แปลงเป็น Grayscale
    image = image.convert("RGB") # แปลงกลับเป็น RGB 3 channels
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# --- 6. Main App ---
model = load_model()
class_names = load_class_names()

# Dictionary สำหรับแปลง Romaji -> Hiragana
HIRAGANA_MAP = {
    'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
    'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
    'sa': 'さ', 'shi': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
    'ta': 'た', 'chi': 'ち', 'tsu': 'つ', 'te': 'て', 'to': 'と',
    'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
    'ha': 'は', 'hi': 'ひ', 'fu': 'ふ', 'he': 'へ', 'ho': 'ほ',
    'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
    'ya': 'や', 'yu': 'ゆ', 'yo': 'よ',
    'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
    'wa': 'わ', 'wo': 'を', 'n': 'ん'
}

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)

# 🔥 รับค่า Query Parameters (เชื่อมต่อกับ Teacher.php)
query_params = st.query_params
req_work_id = query_params.get("work_id", None)
req_quiz_id = query_params.get("quiz_id", None)

target_id = None
active_table = None
mode_title = ""
mode_color = "#D72638"

if req_quiz_id:
    # โหมดตรวจข้อสอบ
    target_id = req_quiz_id
    active_table = "quiz_submissions"
    mode_title = "🟣 ตรวจสอบผลการทดสอบ (Quiz)"
    mode_color = "#7c3aed"
    
elif req_work_id:
    # โหมดตรวจแบบฝึกหัด
    target_id = req_work_id
    active_table = "progress"
    mode_title = "🔴 ตรวจสอบแบบฝึกหัด (Practice)"
    mode_color = "#D72638"

# --- แสดงผลหน้าจอ ---
if target_id and active_table:
    # 📌 กรณีมี ID ส่งมา (Direct Link)
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:20px;">
        <span style="background:{mode_color}; color:white; padding:8px 20px; border-radius:30px; font-weight:bold; font-size:14px;">
            {mode_title} ID: {target_id}
        </span>
    </div>
    <style>
        .result-card {{ border-top-color: {mode_color} !important; }}
        div.stButton > button:first-child {{ background-color: {mode_color} !important; color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

    data_row = get_work_data(target_id, active_table)

    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        
        try:
            image = Image.open(io.BytesIO(blob_data))
        except:
            st.error("ไฟล์รูปภาพเสียหาย")
            image = None

        if image:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1.2], gap="large")
            
            with c1:
                st.caption("ลายมือของนักเรียน")
                # แสดงรูปภาพ
                st.image(image, use_container_width=True)
                st.markdown(f"<div style='text-align:center; margin-top:10px; font-size:18px;'>โจทย์: <b>{true_label}</b></div>", unsafe_allow_html=True)

            with c2:
                st.caption("ผลการวิเคราะห์จาก AI")
                
                # แสดงผลลัพธ์
                if saved_result:
                    # Parse format: "あ (99.5%)" or just "あ"
                    display_char = saved_result.split(' ')[0] if saved_result else "?"
                    conf_val = f"{saved_conf:.1f}%" if saved_conf else "0%"
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:14px; color:#888;">AI มองเห็นเป็น</div>
                        <div class="big-char">{display_char}</div>
                        <div style="font-size:14px; color:#555; margin-top:5px;">ความมั่นใจ</div>
                        <div style="font-size:24px; color:#22c55e; font-weight:bold;">{conf_val}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("🔄 ตรวจสอบใหม่ (Re-Analyze)", use_container_width=True):
                        # Reset ค่าแล้ว Rerun เพื่อเข้าสู่ Loop วิเคราะห์
                        update_database(target_id, active_table, None, 0)
                        st.rerun()
                else:
                    # ยังไม่มีผลตรวจ ให้แสดงปุ่มกด
                    st.markdown("""
                    <div class="result-card" style="border-style:dashed; border-color:#ccc; background:#fafafa;">
                        <h2 style="color:#ccc;">?</h2>
                        <p style="color:#999;">ยังไม่ได้รับการตรวจ</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if st.button("✨ วิเคราะห์ด้วย AI เดี๋ยวนี้", use_container_width=True):
                        if model:
                            with st.spinner("AI กำลังประมวลผล..."):
                                try:
                                    # 1. Predict
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
                                    # 2. Map Result
                                    res_romaji = class_names[idx]
                                    res_kana = HIRAGANA_MAP.get(res_romaji, res_romaji)
                                    
                                    # 3. Format String for DB
                                    final_res_str = f"{res_kana} ({res_romaji})" # Ex: あ (a)
                                    
                                    # 4. Update DB
                                    if update_database(target_id, active_table, final_res_str, conf):
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("บันทึกข้อมูลไม่สำเร็จ")
                                        
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("ไม่พบโมเดล AI")

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("❌ ไม่พบข้อมูลในฐานข้อมูล (ID อาจไม่ถูกต้อง)")

else:
    # 🏠 กรณีเข้าหน้าเว็บตรงๆ (ไม่มี ID) -> ให้เป็นหน้า Landing Page
    st.info("👋 ยินดีต้อนรับ! โปรดเลือก 'ตรวจด้วย AI' จากหน้า Dashboard ของครู")
    st.markdown("""
    <div style="text-align:center; margin-top:20px;">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" width="150" style="opacity:0.8;">
        <p style="color:#888; margin-top:10px;">ระบบพร้อมใช้งาน รอรับคำสั่งจาก Teacher Dashboard...</p>
        <br>
        <a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" style="background:#D72638; color:white; padding:10px 25px; text-decoration:none; border-radius:30px; font-weight:bold;">
            ⬅️ กลับไปที่ Dashboard
        </a>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 Hiragana Master AI | Powered by Streamlit & TensorFlow")