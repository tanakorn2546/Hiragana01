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
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(15px);
            border-radius: 20px; border: 2px solid white; padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; position: relative; z-index: 10;
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
        
        .stButton button { border-radius: 12px !important; font-weight: 600 !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. Database Functions ---

# 🔧 CONFIG: ตั้งค่าชื่อคอลัมน์ของแต่ละตารางที่นี่
# ถ้าตาราง quiz_submissions ใช้ชื่อคอลัมน์เก็บตัวอักษรว่า 'question' หรือ 'answer' ให้แก้ตรงนี้
TABLE_COLUMNS = {
    "progress": "char_code",          # ตารางแบบฝึกหัด ใช้ char_code
    "quiz_submissions": "char_code"   # <--- ตรวจสอบชื่อคอลัมน์นี้ใน Database ของคุณ (อาจจะเป็น correct_answer หรือ char_text)
}

def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app" 
    )

def get_work_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        base_sql = "SELECT id, char_code, ai_result FROM progress WHERE image_data IS NOT NULL"
        if "ยังไม่ตรวจ" in filter_mode: sql = f"{base_sql} AND ai_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode: sql = f"{base_sql} AND ai_result IS NOT NULL ORDER BY id DESC"
        else: sql = f"{base_sql} ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ Database List Error: {e}")
        return []

def get_work_data(target_id, table_name="progress"):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        # เลือกชื่อคอลัมน์ให้ตรงกับตาราง
        target_char_col = TABLE_COLUMNS.get(table_name, "char_code")
        
        # Debug SQL Query
        sql = f"SELECT image_data, ai_result, ai_confidence, {target_char_col} FROM {table_name} WHERE id = %s"
        
        cursor.execute(sql, (target_id,))
        data = cursor.fetchone()
        conn.close()
        
        if data is None:
            # ถ้าไม่เจอข้อมูล ให้ลองเช็คว่ามี ID นี้จริงไหม (เพื่อการ Debug)
            st.warning(f"⚠️ คำสั่ง SQL ทำงานสำเร็จ แต่ไม่พบแถวข้อมูล ID: {target_id} ในตาราง {table_name}")
            return None
            
        return data 
    except mysql.connector.Error as err:
        st.error(f"❌ SQL Error: {err}")
        st.info(f"💡 ข้อแนะนำ: ตรวจสอบว่าตาราง `{table_name}` มีคอลัมน์ชื่อ `{TABLE_COLUMNS.get(table_name, 'char_code')}` หรือไม่?")
        return None
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

def update_database(target_id, table_name, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        # ตรวจสอบว่ามีคอลัมน์ ai_result, ai_confidence ในตารางนั้นจริงหรือไม่
        sql = f"UPDATE {table_name} SET ai_result = %s, ai_confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), target_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Update Error: {e}")
        return False

def get_stats():
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(ai_result) FROM progress WHERE image_data IS NOT NULL")
        return cursor.fetchone()
    except: return 0, 0

# --- 4. Model Loading ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    model_name = 'hiragana_model_best.h5'
    file_id = '1iPYeqEv8uYBvbcgb90pjHX-0cJBakOrI' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        local_path = os.path.join('saved_models', model_name)
        if os.path.exists(local_path): model_name = local_path
        else:
            try:
                gdown.download(url, model_name, quiet=False)
            except Exception as e:
                st.error(f"❌ Load Error: {e}")
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

# --- 5. Preprocessing ---
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    if image.mode != "L": image = image.convert("L")
    image = image.convert("RGB")
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# --- 6. Main Application Logic ---
model = load_model()
class_names = load_class_names()

# Sidebar Stats
with st.sidebar:
    st.markdown("### 🌸 สรุปข้อมูล (Practice)")
    total_w, checked_w = get_stats()
    st.info(f"ภาพทั้งหมด: {total_w}")
    st.success(f"ตรวจแล้ว: {checked_w}")
    
    st.markdown("---")
    st.caption("Database Config Info")
    st.code(str(TABLE_COLUMNS), language="json")

st.markdown('<div class="hero-title">HIRAGANA<br>SENSEI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ระบบตรวจลายมือด้วย MobileNetV2</div>', unsafe_allow_html=True)

# --- 🔥 รับค่า Parameters ---
query_params = st.query_params
req_work_id = query_params.get("work_id", None)
req_quiz_id = query_params.get("quiz_id", None)

current_id = None
active_table = "progress"
is_single_view = False
mode_color = "#D72638"

if req_quiz_id:
    # 🟣 โหมดตรวจแบบทดสอบ
    current_id = req_quiz_id
    active_table = "quiz_submissions"
    is_single_view = True
    mode_color = "#7c3aed"
    st.markdown(f"""
    <div style="background:#f3e8ff; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        📝 กำลังตรวจ: แบบทดสอบ (Quiz Submission ID: {current_id})
    </div>
    <style>
        .stApp {{ background: linear-gradient(180deg, #f3e8ff 0%, #fff 60%, #fff 100%) !important; }}
        .big-char {{ color: {mode_color} !important; }}
        .result-card {{ border-top-color: {mode_color} !important; }}
        div[data-testid="stVerticalBlock"] .stButton button {{ background: {mode_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

elif req_work_id:
    # 🔴 โหมดตรวจแบบฝึกหัด
    current_id = req_work_id
    active_table = "progress"
    is_single_view = True
    st.markdown(f"""
    <div style="background:#ffebee; padding:15px; border-radius:10px; border-left:5px solid {mode_color}; margin-bottom:20px; color:{mode_color}; font-weight:bold; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        ✍️ กำลังตรวจ: แบบฝึกหัด (Work ID: {current_id})
    </div>
    <style>
        div[data-testid="stVerticalBlock"] .stButton button {{ background: {mode_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- Logic การแสดงผล ---

if is_single_view:
    if current_id:
        data_row = get_work_data(current_id, active_table)
        
        if data_row:
            blob_data, saved_result, saved_conf, true_label = data_row
            try: image = Image.open(io.BytesIO(blob_data))
            except: image = None

            if image:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
                            <div style="color:green; font-weight:bold;">{saved_conf:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                        
                        st.write("")
                        if st.button("🔄 ตรวจใหม่", type="secondary", use_container_width=True):
                            update_database(current_id, active_table, None, 0)
                            st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="result-card" style="border: 2px dashed #ddd; background:#fffaf0;">
                            <h1 style="color:{mode_color}; opacity:0.5;">⏳</h1><p style="color:#888;">รอผล...</p>
                        </div>""", unsafe_allow_html=True)
                        
                        st.write("")
                        if st.button("✨ วิเคราะห์", type="primary", use_container_width=True):
                            if model:
                                with st.spinner("AI กำลังคิด..."):
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
                                        
                                        if update_database(current_id, active_table, final_res, conf):
                                            time.sleep(0.3)
                                            st.rerun()
                                    except Exception as e: st.error(f"Error: {e}")
                            else: st.error("ไม่พบโมเดล")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # กรณีหาไม่เจอ (Message จัดการใน get_work_data แล้ว)
            st.error(f"❌ ไม่สามารถดึงข้อมูล ID: {current_id} จากตาราง {active_table}")
            st.info("💡 หากด้านบนขึ้น SQL Error: Unknown Column ให้แก้ตัวแปร TABLE_COLUMNS ในโค้ดบรรทัดที่ 40-43")

else:
    # 🟡 Browse Mode
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        filter_option = st.selectbox("โหมด", ["ทั้งหมด (All)", "ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)"], label_visibility="collapsed")
    
    work_list = get_work_list(filter_option)

    if len(work_list) > 0:
        id_list = [row[0] for row in work_list]
        if 'current_index' not in st.session_state: st.session_state.current_index = 0
        
        if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0
        elif st.session_state.current_index < 0: st.session_state.current_index = len(id_list) - 1

        browse_id = id_list[st.session_state.current_index]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption(f"ID: {browse_id} | {st.session_state.current_index + 1}/{len(id_list)}")

        data_row = get_work_data(browse_id, "progress")
        if data_row:
            blob_data, saved_result, saved_conf, true_label = data_row
            try: image = Image.open(io.BytesIO(blob_data))
            except: image = None

            if image:
                col_img, col_res = st.columns([1, 1.2], gap="large")
                with col_img:
                    st.markdown(f"**โจทย์:** `{true_label}`")
                    st.image(image, use_container_width=True)
                with col_res:
                    st.markdown("**ผลการตรวจ**")
                    if saved_result:
                        st.success(f"{saved_result}\n\nConfidence: {saved_conf:.1f}%")
                        if st.button("🔄 ตรวจใหม่", key=f"re_{browse_id}"):
                            update_database(browse_id, "progress", None, 0)
                            st.rerun()
                    else:
                        if st.button("✨ วิเคราะห์", key=f"an_{browse_id}"):
                            if model:
                                with st.spinner("AI Thinking..."):
                                    preds = import_and_predict(image, model)
                                    idx = np.argmax(preds)
                                    conf = np.max(preds) * 100
                                    res_code = class_names[idx]
                                    final_res = f"{res_code} ({conf:.1f}%)"
                                    update_database(browse_id, "progress", final_res, conf)
                                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        c_prev, c_space, c_next = st.columns([1, 0.2, 1])
        with c_prev:
            if st.button("⬅️ ก่อนหน้า", use_container_width=True):
                st.session_state.current_index -= 1; st.rerun()
        with c_next:
            if st.button("ถัดไป ➡️", use_container_width=True):
                st.session_state.current_index += 1; st.rerun()
    else: st.info("ไม่พบข้อมูล")

st.markdown("""<div style="text-align: center; margin-top: 50px; position:relative; z-index:20;">
<a href="https://www.cedubru.com/hiragana/teacher.php" target="_self" style="color:#D72638; text-decoration:none; font-weight:bold; background:rgba(255,255,255,0.8); padding:5px 15px; border-radius:20px;">🏠 กลับสู่หน้าหลัก</a></div>""", unsafe_allow_html=True)