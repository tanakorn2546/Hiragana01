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

# --- 2. Advanced CSS Styling (Modern UI) ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;700&display=swap');
        
        /* Global Settings */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
            color: #2c3e50;
        }
        
        /* Background */
        .stApp {
            background-color: #f8f9fa;
            background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
            background-size: 20px 20px;
        }

        /* Container Card */
        div.block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        /* Custom Card Style */
        .css-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border: 1px solid #edf2f7;
            transition: transform 0.2s;
        }
        .css-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        }

        /* Header */
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #e74c3c; /* Japanese Red */
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 0px rgba(231, 76, 60, 0.1);
        }
        .header-subtitle {
            font-size: 1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Result Display */
        .result-box {
            background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
            border-left: 6px solid #e74c3c;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.1);
        }
        .result-char {
            font-size: 3.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 0;
            line-height: 1.2;
        }
        .result-conf {
            color: #e74c3c;
            font-weight: 600;
            font-size: 1rem;
            background: rgba(231, 76, 60, 0.1);
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 10px;
        }

        /* Navigation Buttons */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Primary Action Button (Check) */
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button {
             /* Target specific buttons if possible, otherwise generic styling applied */
        }

        /* Custom Link Button */
        .home-btn {
            display: inline-block;
            background: #2c3e50;
            color: white !important;
            padding: 12px 30px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
            transition: all 0.3s;
        }
        .home-btn:hover {
            background: #34495e;
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(44, 62, 80, 0.4);
        }

        /* Radio Group Styling */
        div[role="radiogroup"] {
            background: white;
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            justify-content: center;
            display: flex;
            gap: 10px;
        }
        
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Connection ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app" 
    )

# --- Database Functions ---
def get_work_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
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
        st.error(f"❌ Database Error: {e}")
        return []

def get_work_data(work_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
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

# --- 4. Smart Model Loader ---
@st.cache_resource
def load_model():
    file_id = '1Yw1YCu35oxQT5jpB0xqouZMD-MH2EGZO' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path):
            model_name = local_path
        else:
            try:
                gdown.download(url, model_name, quiet=False)
            except Exception:
                return None
    try:
        return tf.keras.models.load_model(model_name, compile=False)
    except Exception:
        return None

def load_class_names():
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
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 5. Main UI Logic ---
model = load_model()
class_names = load_class_names()

# --- Header Section ---
st.markdown("""
    <div class="header-title">🇯🇵 Hiragana Sensei AI</div>
    <div class="header-subtitle">ระบบช่วยตรวจลายมือฮิรางานะอัจฉริยะ</div>
""", unsafe_allow_html=True)

# --- Filter & Navigation Logic ---
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c_fil_1, c_fil_2, c_fil_3 = st.columns([1, 4, 1])
with c_fil_2:
    if target_work_id:
        st.info(f"🔍 Viewing Specific ID: {target_work_id}")
        filter_option = "ทั้งหมด (All)"
    else:
        # Styled Radio
        filter_option = st.radio(
            "",
            ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"],
            horizontal=True,
            label_visibility="collapsed"
        )

work_list = get_work_list(filter_option)

if len(work_list) > 0:
    id_list = [row[0] for row in work_list]
    
    # Sync Index with ID
    if target_work_id and int(target_work_id) in id_list:
        if 'current_index' not in st.session_state or id_list[st.session_state.current_index] != int(target_work_id):
            st.session_state.current_index = id_list.index(int(target_work_id))
    elif 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    # --- Progress Indicator ---
    progress = (st.session_state.current_index + 1) / len(id_list)
    st.progress(progress)
    st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#888; margin-top:-10px;'>Card {st.session_state.current_index + 1} / {len(id_list)}</div>", unsafe_allow_html=True)

    # --- Main Content Card ---
    st.markdown('<div class="css-card">', unsafe_allow_html=True) # Start Card

    data_row = get_work_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf, true_label = data_row
        
        try:
            image = Image.open(io.BytesIO(blob_data))
        except:
            image = None

        if image:
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown(f"**📝 โจทย์:** `{true_label}`")
                st.image(image, use_column_width=True, caption=f"ID: {current_id}")
            
            with col2:
                st.markdown("### 🤖 ผลลัพธ์ AI")
                
                if saved_result:
                    # Show Result Card
                    st.markdown(f"""
                        <div class="result-box">
                            <p style="color:#888; font-size:0.9rem; margin-bottom:5px;">AI อ่านว่า</p>
                            <p class="result-char">{saved_result.split(' ')[0]}</p>
                            <p style="color:#555; font-size:0.9rem;">{saved_result.split(' ')[1] if len(saved_result.split(' ')) > 1 else ''}</p>
                            <div class="result-conf">⚡ Confidence {saved_conf:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Spacer
                    if st.button("🔄 ตรวจใหม่", type="secondary", use_container_width=True):
                        update_database(current_id, None, 0)
                        st.rerun()
                else:
                    # Pending State
                    st.info("รอการตรวจ...")
                    if st.button("✨ วิเคราะห์ผล (Analyze)", type="primary", use_container_width=True):
                        if model:
                            with st.spinner("กำลังประมวลผล..."):
                                try:
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    
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
                                        'wa': 'わ (wa)', 'wo': 'を (wo)', 'n': 'ん (n)'
                                    }
                                    
                                    final_res = hiragana_map.get(res_code, res_code)
                                    update_database(current_id, final_res, conf)
                                    st.toast(f"✅ เรียบร้อย! อ่านว่า {final_res}")
                                    time.sleep(0.5)
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Model not found")

    st.markdown('</div>', unsafe_allow_html=True) # End Card

    # --- Navigation Bar ---
    c_prev, c_center, c_next = st.columns([1, 2, 1])
    
    with c_prev:
        if st.session_state.current_index > 0:
            if st.button("⬅️ ก่อนหน้า"):
                st.session_state.current_index -= 1
                st.rerun()
            
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ➡️"):
                st.session_state.current_index += 1
                st.rerun()
        else:
            if st.button("⏮ เริ่มใหม่"):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.markdown("""
        <div class="css-card" style="text-align:center; padding:50px;">
            <h3>📭 ไม่พบรายการงาน</h3>
            <p style="color:#888;">ไม่มีข้อมูลในหมวดที่คุณเลือก</p>
        </div>
    """, unsafe_allow_html=True)

# --- Footer ---
teacher_dashboard_url = "https://www.cedubru.com/hiragana/teacher.php?view_student=7" 

st.markdown(f"""
    <div style="text-align: center; margin-top: 40px;">
        <a href="{teacher_dashboard_url}" target="_self" class="home-btn">
            🏠 กลับสู่ห้องพักครู (Dashboard)
        </a>
    </div>
    <div style="text-align:center; color:#bdc3c7; font-size:0.8rem; margin-top: 20px;">
        Hiragana Image Classification System V.2.1 | Powered by MobileNetV2
    </div>
""", unsafe_allow_html=True)