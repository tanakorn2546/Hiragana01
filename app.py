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
    layout="wide", # เปลี่ยนเป็น Wide เพื่อให้ใช้พื้นที่เต็มจอมากขึ้น
    initial_sidebar_state="expanded" # เปิด Sidebar ไว้เลยเพื่อไม่ให้โล่ง
)

# --- 2. Advanced CSS Styling (Super Modern UI) ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;700&family=Mochiy+Pop+One&display=swap');
        
        /* --- Global Settings --- */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif !important;
            color: #2c3e50;
        }
        
        /* --- Background with Japanese Pattern --- */
        .stApp {
            background-color: #fff0f5; /* Lavender Blush */
            opacity: 1;
            background-image:  radial-gradient(#ffcdd2 1.5px, transparent 1.5px), radial-gradient(#ffcdd2 1.5px, #fff0f5 1.5px);
            background-size: 30px 30px;
            background-position: 0 0, 15px 15px;
        }

        /* --- Main Card Container --- */
        div.block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 5rem;
            margin: auto;
        }

        /* --- Card Style --- */
        .css-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 30px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.1);
            border: 2px solid #ffffff;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        
        /* แถบสีแดงด้านบนการ์ด */
        .css-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 8px;
            background: linear-gradient(90deg, #ff9a9e 0%, #ff5252 100%);
        }

        /* --- Image Styling (Polaroid Look) --- */
        div[data-testid="stImage"] {
            background-color: white;
            padding: 15px;
            padding-bottom: 40px; /* พื้นที่ด้านล่างเยอะหน่อยเหมือนโพลารอยด์ */
            border-radius: 4px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.15);
            transform: rotate(-2deg); /* เอียงนิดๆ ให้ดูอาร์ต */
            transition: transform 0.3s ease;
            border: 1px solid #eee;
        }
        div[data-testid="stImage"]:hover {
            transform: rotate(0deg) scale(1.02); /* ชี้แล้วตรงและขยาย */
            box-shadow: 10px 10px 25px rgba(0,0,0,0.2);
            z-index: 10;
        }
        div[data-testid="stImage"] img {
            border-radius: 0px !important; /* รูปข้างในไม่มน */
            border: 1px solid #ddd;
        }

        /* --- Header Text --- */
        .header-title {
            font-family: 'Mochiy Pop One', sans-serif; /* ฟอนต์น่ารักๆ สำหรับหัวข้อ */
            font-size: 3rem;
            color: #d32f2f;
            text-align: center;
            margin-bottom: 0px;
            text-shadow: 3px 3px 0px rgba(255,200,200,0.5);
            letter-spacing: 2px;
        }
        .header-subtitle {
            font-size: 1.1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2.5rem;
            font-weight: 400;
            background-color: rgba(255,255,255,0.6);
            display: inline-block;
            padding: 5px 20px;
            border-radius: 20px;
        }
        .header-container {
            text-align: center;
        }

        /* --- Result Box --- */
        .result-box {
            background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
            border: 2px dashed #ffcdd2;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            position: relative;
        }
        .result-char {
            font-family: 'Mochiy Pop One', sans-serif;
            font-size: 4rem;
            color: #d32f2f;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .result-conf {
            font-size: 0.9rem;
            background: #ffebee;
            color: #c62828;
            padding: 5px 15px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
        }

        /* --- Button Styling --- */
        .stButton button {
            border-radius: 15px !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s !important;
            box-shadow: 0 4px 0px rgba(0,0,0,0.1) !important; /* เงาแบบปุ่มกด */
            width: 100%;
        }
        
        /* ปุ่ม Analyze (Primary) */
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button[kind="primary"] {
             background: linear-gradient(to bottom, #ff7e7e, #d32f2f) !important;
             color: white !important;
        }
        /* ปุ่ม Analyze Hover */
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button[kind="primary"]:hover {
             transform: translateY(-2px);
             box-shadow: 0 6px 0px rgba(180, 0, 0, 0.2) !important;
        }
        /* ปุ่ม Analyze Active (กดลง) */
        div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button[kind="primary"]:active {
             transform: translateY(2px);
             box-shadow: 0 0px 0px rgba(0,0,0,0) !important;
        }

        /* ปุ่ม Next/Prev (Secondary) */
        .stButton button {
             background: white !important;
             color: #555 !important;
             border: 2px solid #eee !important;
        }
        .stButton button:hover {
             background: #f8f9fa !important;
             border-color: #d32f2f !important;
             color: #d32f2f !important;
        }

        /* --- Footer Link Button --- */
        .home-btn {
            background: #2c3e50;
            color: white !important;
            padding: 15px 40px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            box-shadow: 0 10px 20px rgba(44, 62, 80, 0.3);
            transition: all 0.3s;
            border: 4px solid rgba(255,255,255,0.2);
        }
        .home-btn:hover {
            background: #34495e;
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(44, 62, 80, 0.4);
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

def get_stats():
    """ดึงสถิติภาพรวม"""
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        total, checked = cursor.fetchone()
        conn.close()
        return total, checked
    except:
        return 0, 0

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

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📊 Dashboard Info")
    st.markdown("สรุปข้อมูลการตรวจงานล่าสุด")
    
    total_w, checked_w = get_stats()
    pending_w = total_w - checked_w
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ทั้งหมด", f"{total_w}", "ใบ")
    with c2:
        st.metric("ตรวจแล้ว", f"{checked_w}", f"{(checked_w/total_w*100) if total_w else 0:.0f}%")
        
    st.markdown("---")
    st.info(f"📍 รอการตรวจ: {pending_w} ใบ")
    st.markdown("---")
    st.markdown("### 💡 วิธีใช้งาน")
    st.markdown("""
    1. เลือกโหมดกรองข้อมูล (ทั้งหมด/รอตรวจ)
    2. กด **'วิเคราะห์ผล'** เพื่อให้ AI อ่าน
    3. กด **'ตรวจใหม่'** หากต้องการแก้ไข
    """)

# --- Header Section ---
st.markdown("""
    <div class="header-container">
        <div class="header-title">🌸 Hiragana Sensei 🌸</div>
        <div class="header-subtitle">AI Assistant for Japanese Handwriting</div>
    </div>
""", unsafe_allow_html=True)

# --- Filter & Navigation Logic ---
query_params = st.query_params
target_work_id = query_params.get("work_id", None)

c_fil_1, c_fil_2, c_fil_3 = st.columns([1, 6, 1])
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
    st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#888; margin-top:-10px; margin-bottom: 20px;'>Card {st.session_state.current_index + 1} / {len(id_list)}</div>", unsafe_allow_html=True)

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
                st.markdown(f"##### 📝 โจทย์: `{true_label}`")
                # รูปจะถูกแต่งด้วย CSS (Polaroid Style) อัตโนมัติ
                st.image(image, use_column_width=True, caption=f"ID: {current_id}")
            
            with col2:
                st.markdown("##### 🤖 ผลลัพธ์ AI")
                st.write("") # Spacer
                
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
                    if st.button("🔄 ตรวจใหม่", key="btn_retry", use_container_width=True):
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
            if st.button("⬅️ ก่อนหน้า", key="btn_prev"):
                st.session_state.current_index -= 1
                st.rerun()
            
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ➡️", key="btn_next"):
                st.session_state.current_index += 1
                st.rerun()
        else:
            if st.button("⏮ เริ่มใหม่", key="btn_reset"):
                st.session_state.current_index = 0
                st.rerun()

else:
    st.markdown("""
        <div class="css-card" style="text-align:center; padding:50px;">
            <h1 style="font-size: 50px;">📭</h1>
            <h3>ไม่พบรายการงาน</h3>
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
    <div style="text-align:center; color:#95a5a6; font-size:0.8rem; margin-top: 25px;">
        Hiragana Image Classification System V.2.2 Pro
    </div>
""", unsafe_allow_html=True)