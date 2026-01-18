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
import requests # 📦 จำเป็นต้องมี: pip install requests

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

# --- 2. CSS ตกแต่ง (คงเดิม 100%) ---
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
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

# ฟังก์ชันดึงข้อมูลงานจาก ID (เชื่อมกับตาราง progress)
def get_work_by_id(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor(dictionary=True) # ใช้ Dictionary เพื่อเรียกชื่อ Column ง่ายๆ
        # ดึงข้อมูลจากตาราง progress ตามที่ teacher.php ใช้งาน
        sql = "SELECT id, user_id, char_code, image_path, ai_result, ai_confidence, status FROM progress WHERE id = %s"
        cursor.execute(sql, (work_id,))
        data = cursor.fetchone()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return None

# ฟังก์ชันอัปเดตผล AI (เชื่อมกับตาราง progress)
def update_progress(work_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # อัปเดตผลลัพธ์กลับไปที่ตาราง progress
        sql = "UPDATE progress SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), work_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Update Error: {e}")
        return False

# --- 4. Smart Model Loader (คงเดิม) ---
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
            st.warning("📥 กำลังดาวน์โหลด Model... (รอสักครู่)")
            try:
                gdown.download(url, model_name, quiet=False)
                st.success("✅ โหลด Model สำเร็จ!")
            except Exception as e:
                st.error(f"❌ โหลดไม่สำเร็จ: {e}")
                return None

    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except Exception as e:
        st.error(f"❌ ไฟล์โมเดลเสียหาย: {e}")
        return None

def load_class_names():
    # ใช้ Default Mapping ไปเลยเพื่อความชัวร์ ถ้าหาไฟล์ json ไม่เจอ
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

# ฟังก์ชันทำนายผล (คงเดิม)
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

# --- 5. Main UI (Logic ใหม่เพื่อรับงานจาก Teacher) ---
model = load_model()
class_names = load_class_names()

st.markdown("""
    <div class='app-header-icon'>🇯🇵</div>
    <h1>Hiragana Sensei AI</h1>
    <p style='text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1rem;'>
        ระบบช่วยครูตรวจงานอัตโนมัติ (AI Assistant)
    </p>
""", unsafe_allow_html=True)

# รับค่า work_id จาก URL (ที่ส่งมาจาก teacher.php)
# ตัวอย่าง URL: app.py?work_id=15
query_params = st.query_params
work_id = query_params.get("work_id", None)

# URL ฐานสำหรับดึงรูปภาพ (ต้องตรงกับ Server ที่ teacher.php อยู่)
BASE_URL = "http://www.cedubru.com/"

if work_id:
    # --- กรณีมี work_id ส่งมา (ครูกดมาจาก Teacher Dashboard) ---
    
    work_data = get_work_by_id(work_id)
    
    if work_data:
        st.markdown(f"""
            <div style='text-align: center; margin-bottom: 15px; background: #FFEBEE; padding: 10px; border-radius: 10px; color: #D32F2F; font-weight: bold;'>
                📍 กำลังตรวจงาน ID: {work_id} | โจทย์: {work_data['char_code']}
            </div>
        """, unsafe_allow_html=True)
        
        # สร้าง URL เต็มของรูปภาพ
        image_path = work_data['image_path'] # เช่น uploads/works/xxx.png
        full_image_url = BASE_URL + image_path
        
        try:
            # 🟢 [แก้ไขสำคัญ] ใส่ Headers หลอก Server ว่าเราเป็น Browser (Chrome) เพื่อไม่ให้โดนบล็อก 404/403
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(full_image_url, stream=True, headers=headers)
            response.raise_for_status() # เช็คว่าโหลดได้จริงไหม (200 OK)
            
            image = Image.open(io.BytesIO(response.content))
            
            # แบ่งคอลัมน์แสดงผล
            col_img, col_act = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption="ลายมือนักเรียน", use_column_width=True)
            
            with col_act:
                st.markdown("### ผลการวิเคราะห์")
                
                # แสดงผลลัพธ์ถ้ามีอยู่แล้ว
                if work_data['ai_result']:
                    st.markdown(f"""
                        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 15px; border: 2px dashed #2196f3; margin-bottom: 20px; text-align: center;">
                            <h1 style="color: #1565c0 !important; margin: 0; font-size: 3rem; font-weight: 800;">{work_data['ai_result']}</h1>
                            <p style="margin-top: 10px; font-size: 1rem; color: #555;">Confidence: <strong>{work_data['ai_confidence']}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ ยังไม่ได้ตรวจ")

                # ปุ่มกดตรวจ
                if st.button("🤖 ให้ AI ตรวจสอบ"):
                    if model:
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            # ทำนาย
                            preds = import_and_predict(image, model)
                            idx = np.argmax(preds)
                            conf = np.max(preds) * 100
                            
                            if idx < len(class_names):
                                res_code = class_names[idx]
                            else:
                                res_code = "Unknown"

                            # Mapping ตัวอักษร
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
                            
                            # อัปเดต Database
                            if update_progress(work_id, final_res, conf):
                                st.success(f"✅ บันทึกผลสำเร็จ: {final_res}")
                                st.balloons()
                                time.sleep(1.0)
                                st.rerun()
                            else:
                                st.error("❌ บันทึกข้อมูลล้มเหลว")
                    else:
                        st.error("Model not loaded")

        except Exception as e:
            st.error(f"ไม่สามารถโหลดรูปภาพได้: {e}")
            st.caption(f"URL: {full_image_url}")
            
    else:
        st.error(f"ไม่พบข้อมูลงาน ID: {work_id}")

else:
    # --- กรณีเปิดหน้านี้เฉยๆ โดยไม่มี ID ---
    st.info("👋 ยินดีต้อนรับคุณครู")
    st.markdown("""
        <div style="text-align:center; padding: 40px; color: #777;">
            <p>กรุณาเข้าใช้งานผ่านปุ่ม <strong>"AI Check"</strong> <br>ในระบบ Teacher Dashboard เพื่อเลือกงานที่จะตรวจ</p>
        </div>
    """, unsafe_allow_html=True)


# --- Link กลับเว็บหลัก ---
# เปลี่ยนลิงก์ให้กลับไปที่ teacher.php เพื่อความสะดวก
home_url = "http://www.cedubru.com/teacher.php" 

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <a href="{home_url}" class="custom-home-btn">
            🏠 กลับสู่ห้องพักครู (Teacher Room)
        </a>
    </div>
    <div class="footer-credit">
        <strong>Hiragana AI Assistant V.2.0</strong>
    </div>
""", unsafe_allow_html=True)