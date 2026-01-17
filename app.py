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

# --- [Config] ---
config_dir = ".streamlit"
if not os.path.exists(config_dir): os.makedirs(config_dir)
with open(os.path.join(config_dir, "config.toml"), "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#D32F2F"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF0F5"\ntextColor="#333333"\n')

st.set_page_config(page_title="Hiragana Sensei AI", page_icon="🇯🇵", layout="centered")

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif !important; color: #333; }
    .stApp { background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 100%); background-attachment: fixed; }
    div.block-container { background: rgba(255, 255, 255, 0.95); border-radius: 30px; padding: 2rem; box-shadow: 0 15px 50px rgba(0,0,0,0.1); border-top: 5px solid #D32F2F; }
    h1 { text-align: center; color: #D32F2F; font-weight: 800; }
    .result-box { background: #FFEBEE; padding: 20px; border-radius: 15px; border: 2px solid #D32F2F; text-align: center; margin-top: 20px; }
    .custom-btn { background: #333; color: white; text-decoration: none; padding: 10px 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Database ---
def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_hiragana_app",
        password="7gZ8gDJyufzJyzELZkdg",
        database="cedubruc_hiragana_app"
    )

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

# --- 🛠️ MAGIC FIX: แก้ปัญหา TensorFlow รุ่นเก่าอ่านโมเดลรุ่นใหม่ไม่ได้ ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None) # ตัดคำสั่ง groups ทิ้ง เพื่อไม่ให้ Error
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    file_id = '1ezDUsDxeabZX06ArdjtcWPk0uradYWDD' 
    model_name = 'hiragana_mobilenetv2_best.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_name):
        try:
            gdown.download(url, model_name, quiet=False)
        except: return None

    try:
        # ใช้ Custom Objects เพื่อยัดไส้ Class ที่เราแก้บั๊กแล้วเข้าไป
        return tf.keras.models.load_model(model_name, compile=False, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

def predict(image, model):
    class_names = ['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho', 'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo', 'n']
    hiragana_map = {'a': 'あ (a)', 'i': 'い (i)', 'u': 'う (u)', 'e': 'え (e)', 'o': 'お (o)', 'ka': 'か (ka)', 'ki': 'き (ki)', 'ku': 'く (ku)', 'ke': 'け (ke)', 'ko': 'こ (ko)', 'sa': 'さ (sa)', 'shi': 'し (shi)', 'su': 'す (su)', 'se': 'せ (se)', 'so': 'そ (so)', 'ta': 'た (ta)', 'chi': 'ち (chi)', 'tsu': 'つ (tsu)', 'te': 'て (te)', 'to': 'と (to)', 'na': 'な (na)', 'ni': 'に (ni)', 'nu': 'ぬ (nu)', 'ne': 'ね (ne)', 'no': 'の (no)', 'ha': 'は (ha)', 'hi': 'ひ (hi)', 'fu': 'ふ (fu)', 'he': 'へ (he)', 'ho': 'ほ (ho)', 'ma': 'ま (ma)', 'mi': 'み (mi)', 'mu': 'む (mu)', 'me': 'め (me)', 'mo': 'も (mo)', 'ya': 'や (ya)', 'yu': 'ゆ (yu)', 'yo': 'よ (yo)', 'ra': 'ら (ra)', 'ri': 'り (ri)', 'ru': 'る (ru)', 're': 'れ (re)', 'ro': 'ろ (ro)', 'wa': 'わ (wa)', 'wo': 'を (wo)', 'n': 'ん (n)'}
    
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS).convert("RGB")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(image, dtype=np.float32))
    data = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(data)
    idx = np.argmax(preds)
    res_code = class_names[idx] if idx < len(class_names) else "Unknown"
    return hiragana_map.get(res_code, res_code), np.max(preds) * 100

# --- MAIN ---
model = load_model()
st.markdown("<h1>🇯🇵 Hiragana Sensei AI</h1>", unsafe_allow_html=True)

try:
    qp = st.query_params
    work_id = qp.get("work_id")
    img_url = qp.get("image_url")
except:
    work_id, img_url = None, None

if work_id and img_url:
    st.info(f"กำลังตรวจงาน ID: {work_id}")
    try:
        # หลอก Server ว่าเป็น Chrome
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(img_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            col1, col2 = st.columns([1, 1])
            col1.image(img, caption="รูปนักเรียน", use_column_width=True)
            
            if model:
                with col2:
                    with st.spinner("AI กำลังคิด..."):
                        res, conf = predict(img, model)
                        st.markdown(f"<div class='result-box'><h1>{res}</h1><p>ความมั่นใจ: {conf:.1f}%</p></div>", unsafe_allow_html=True)
                        if st.button("💾 บันทึกผล", type="primary"):
                            if update_student_progress(work_id, res, conf):
                                st.success("บันทึกแล้ว!")
                                st.balloons()
                            else: st.error("บันทึกพลาด")
            else: st.error("โหลดโมเดลไม่ได้")
        else:
            st.error(f"โหลดรูปไม่ได้ (Code: {response.status_code})")
            st.caption(f"URL: {img_url}")
    except Exception as e: st.error(f"Error: {e}")
else:
    st.warning("กรุณาเข้าใช้งานผ่านหน้า Teacher Dashboard")