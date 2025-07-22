import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# ============================================
# 🎨 MODERN UI CONFIGURATION
# ============================================
st.set_page_config(
    page_title="SIBI Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan modern dark theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Poppins', sans-serif;
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Streamlit main container */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 0;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #374151;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Modern button container */
    .button-container {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }
    
    /* Custom buttons */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: white !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    
    /* Specific button colors */
    .start-btn button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
    }
    
    .stop-btn button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
    }
    
    .reset-btn button {
        background: linear-gradient(135deg, #f59e0b, #d97706) !important;
        color: white !important;
    }
    
    .learn-btn button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
    }
    
    /* Sentence display */
    .sentence-display {
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #6366f1;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        border: 1px solid #4b5563;
    }
    
    .sentence-display h3 {
        color: #f9fafb;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .sentence-text {
        background: #111827;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        color: #e5e7eb;
        min-height: 50px;
        display: flex;
        align-items: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        border: 1px solid #374151;
    }
    
    /* Camera container */
    .camera-container {
        background: #1f2937;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 1px solid #374151;
    }
    
    /* Info panel */
    .info-panel {
        background: linear-gradient(135deg, #1e3a8a, #3730a3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        border: 1px solid #4338ca;
    }
    
    .info-panel .info-icon {
        color: #60a5fa;
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Saved sentences */
    .saved-sentences {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin-top: 2rem;
        border: 1px solid #374151;
    }
    
    .saved-sentences h3 {
        color: #f9fafb;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sentence-item {
        background: #111827;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #6366f1;
        transition: all 0.3s ease;
        border: 1px solid #374151;
        color: #e5e7eb;
    }
    
    .sentence-item:hover {
        background: #1f2937;
        transform: translateX(5px);
        border-left: 4px solid #8b5cf6;
    }
    
    /* Image upload section */
    .upload-section {
        background: #1f2937;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin-top: 2rem;
        border: 1px solid #374151;
    }
    
    .upload-section h3 {
        color: #f9fafb;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .upload-section p {
        color: #d1d5db;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #d1fae5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        border: 1px solid #059669;
    }
    
    .success-message h4 {
        color: #ecfdf5;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #6366f1, transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: #111827;
        border: 2px dashed #4b5563;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stFileUploader > div:hover {
        border-color: #6366f1;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
    }
    
    /* Text elements */
    .stText, .stMarkdown, p {
        color: #e5e7eb;
    }
    
    /* Footer styling */
    .footer-info {
        background: #1f2937;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #9ca3af;
        border: 1px solid #374151;
    }
    
    .footer-info p {
        margin: 0.5rem 0;
    }
    
    .footer-info strong {
        color: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Label dictionary
labels_dict = {i: chr(65 + i) if i < 26 else str(i - 25) for i in range(36)}

# Session state initialization
for key, value in {
    'run': False,
    'sentence': "",
    'last_prediction': None,
    'reset_time': 0,
    'show_pembelajaran': False,
    'last_gesture_time': time.time(),
    'has_started': False,
    'saved_sentences': [],
    'show_help_panel': False,
    'was_over_help_box': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Constants for help icon
HELP_BOX_SIZE = 50
HELP_BOX_PADDING = 20

def draw_help_icon(frame):
    x = HELP_BOX_PADDING
    y = HELP_BOX_PADDING
    cv2.rectangle(frame, (x, y), (x + HELP_BOX_SIZE, y + HELP_BOX_SIZE), (50, 150, 250), -1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize("?", font, font_scale, font_thickness)[0]
    text_x = x + (HELP_BOX_SIZE - text_size[0]) // 2
    text_y = y + (HELP_BOX_SIZE + text_size[1]) // 2
    cv2.putText(frame, "?", (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

def draw_info_panel(frame):
    H, W, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (255, 255, 255)

    lines = [
        "Panduan:",
        "- Klik *Reset Kalimat*, Jika Ingin Memulai Kata Dari Ulang",
        "- Klik *Berhenti Deteksi*, Jika Ingin Berhenti Deteksi dan",
        "   Ingin Menggunakan/Menghapus Kalimat Yang Tersimpan",
        "- Klik *Pembelajaran Bahasa Isyarat*, Jika Ingin Melihat",
        "   Abjad Bahasa Isyarat",
        "- Pastikan gerakan tangan jelas",
        "- Tahan gesture 1 detik untuk input",
        "- Lepas gesture selama 1 detik untuk Spasi"
    ]

    line_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
    max_width = max(size[0] for size in line_sizes)
    line_height = max(size[1] for size in line_sizes) + 8

    panel_w = max_width + 20
    panel_h = line_height * len(lines) + 20

    x = HELP_BOX_PADDING
    y = HELP_BOX_SIZE + 2 * HELP_BOX_PADDING

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (50, 150, 250), -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(lines):
        text_y = y + 20 + i * line_height
        cv2.putText(frame, line, (x + 10, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def is_hand_over_help_box(hand_landmarks, W, H):
    idx_tip = hand_landmarks.landmark[8]
    cx, cy = int(idx_tip.x * W), int(idx_tip.y * H)
    x = HELP_BOX_PADDING
    y = HELP_BOX_PADDING
    return x <= cx <= x + HELP_BOX_SIZE and y <= cy <= y + HELP_BOX_SIZE

# ============================================
# 🎨 MODERN HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🤟 SIBI Sign Language Detection</h1>
    <p>Sistem Deteksi Bahasa Isyarat Indonesia yang Cerdas dan Modern</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 🎛️ MODERN CONTROL BUTTONS
# ============================================
st.markdown('<div class="button-container">', unsafe_allow_html=True)

start_col, stop_col, reset_col, belajar_col = st.columns([1, 1, 1, 1])

with start_col:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("▶️ Mulai Deteksi", key="start"):
        st.session_state.run = True
    st.markdown('</div>', unsafe_allow_html=True)

with stop_col:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("⏸️ Berhenti Deteksi", key="stop"):
        st.session_state.run = False
    st.markdown('</div>', unsafe_allow_html=True)

with reset_col:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("🔁 Reset Kalimat", key="reset"):
        if st.session_state.sentence.strip():
            st.session_state.saved_sentences.append(st.session_state.sentence.strip())
        st.session_state.sentence = ""
        st.session_state.last_prediction = None
        st.session_state.reset_time = 0
        st.session_state.has_started = False
    st.markdown('</div>', unsafe_allow_html=True)

with belajar_col:
    st.markdown('<div class="learn-btn">', unsafe_allow_html=True)
    if st.button("📚 Pembelajaran Bahasa Isyarat", key="learn"):
        st.session_state.show_pembelajaran = not st.session_state.show_pembelajaran
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 📱 MAIN INTERFACE LAYOUT
# ============================================
if st.session_state.run and st.session_state.show_pembelajaran:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### 📚 Panduan Bahasa Isyarat")
        pembelajaran_placeholder = st.empty()
    
    st.markdown('<div class="sentence-display">', unsafe_allow_html=True)
    sentence_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
elif st.session_state.run:
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sentence-display">', unsafe_allow_html=True)
    sentence_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    pembelajaran_placeholder = None
    
elif st.session_state.show_pembelajaran:
    st.markdown("### 📚 Panduan Bahasa Isyarat SIBI")
    st.image("abjad.png", caption="Abjad Bahasa Isyarat", use_container_width=True)
    frame_placeholder = None
    sentence_placeholder = None
    pembelajaran_placeholder = None
    
else:
    st.markdown("""
    <div class="info-panel">
        <div style="display: flex; align-items: center; color: #e5e7eb;">
            <span class="info-icon">ℹ️</span>
            <span>Klik tombol <strong>▶️ Mulai Deteksi</strong> untuk memulai deteksi bahasa isyarat</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    frame_placeholder = st.empty()
    sentence_placeholder = st.empty()
    pembelajaran_placeholder = None

# ============================================
# 🎥 CAMERA PROCESSING (unchanged logic)
# ============================================
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    cooldown = 1.0

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membuka kamera.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = ""
        gesture_detected = False

        if results.multi_hand_landmarks:
            gesture_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            is_over = is_hand_over_help_box(hand_landmarks, W, H)

            if is_over and not st.session_state.was_over_help_box:
                st.session_state.show_help_panel = not st.session_state.show_help_panel
            st.session_state.was_over_help_box = is_over

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            width = max_x - min_x if max_x - min_x != 0 else 1e-6
            height = max_y - min_y if max_y - min_y != 0 else 1e-6

            data_aux = []
            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                data_aux.extend([norm_x, norm_y])

            prediction = model.predict([np.array(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "")

            now = time.time()
            if predicted_character == st.session_state.last_prediction:
                if st.session_state.reset_time == 0:
                    st.session_state.reset_time = now
                elif now - st.session_state.reset_time >= cooldown:
                    st.session_state.sentence += predicted_character
                    st.session_state.reset_time = 0
                    st.session_state.last_prediction = None
                    st.session_state.has_started = True
            else:
                st.session_state.last_prediction = predicted_character
                st.session_state.reset_time = now

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max_x * W) + 10
            y2 = int(max_y * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            st.session_state.last_gesture_time = time.time()
        else:
            st.session_state.was_over_help_box = False
            if st.session_state.has_started and not gesture_detected and time.time() - st.session_state.last_gesture_time > 1:
                if not st.session_state.sentence.endswith(" "):
                    st.session_state.sentence += " "
                    st.session_state.last_prediction = None
                    st.session_state.reset_time = 0

        draw_help_icon(frame)
        if st.session_state.show_help_panel:
            draw_info_panel(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_placeholder:
            frame_placeholder.image(frame, channels='RGB')
        if sentence_placeholder:
            sentence_placeholder.markdown(f"""
            <div class="sentence-text">
                📝 Kalimat: {st.session_state.sentence}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.show_pembelajaran and pembelajaran_placeholder:
            pembelajaran_placeholder.image("abjad.png", caption="Abjad Bahasa Isyarat", use_container_width=True)

    cap.release()

# ============================================
# 💾 SAVED SENTENCES SECTION
# ============================================
if st.session_state.saved_sentences:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="saved-sentences">
        <h3>💾 Kalimat Tersimpan</h3>
    </div>
    """, unsafe_allow_html=True)
    
    saved_copy = st.session_state.saved_sentences.copy()
    for i, sent in enumerate(saved_copy):
        st.markdown(f"""
        <div class="sentence-item">
            <strong>Kalimat {i+1}:</strong> {sent}
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns([8, 1, 1])
        with cols[1]:
            if st.button("🗑️", key=f"hapus_{i}", help="Hapus kalimat"):
                st.session_state.saved_sentences.pop(i)
                st.rerun()
        with cols[2]:
            if st.button("📝", key=f"pakai_{i}", help="Gunakan kalimat"):
                st.session_state.sentence = sent
                st.rerun()

# ============================================
# 📁 IMAGE UPLOAD SECTION
# ============================================
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="upload-section">
    <h3>📁 Deteksi Dari Gambar</h3>
    <p>Unggah gambar tangan untuk mendapatkan deteksi bahasa isyarat secara otomatis</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Pilih satu atau lebih gambar tangan:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Format yang didukung: JPG, JPEG, PNG"
)

if uploaded_files:
    kalimat_gambar = ""
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Memproses gambar {idx + 1} dari {len(uploaded_files)}: {file.name}")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            width = max_x - min_x if max_x - min_x != 0 else 1e-6
            height = max_y - min_y if max_y - min_y != 0 else 1e-6

            data_aux = []
            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                data_aux.extend([norm_x, norm_y])

            prediction = model.predict([np.array(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "")
            kalimat_gambar += predicted_character
        else:
            kalimat_gambar += "?"

    status_text.text("✅ Proses selesai!")
    progress_bar.progress(1.0)
    
    st.markdown(f"""
    <div class="success-message">
        <h4>✅ Hasil Deteksi dari Gambar</h4>
        <div class="sentence-text">
            {kalimat_gambar}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 📊 FOOTER INFO
# ============================================
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer-info">
    <p>🤟 <strong>SIBI Sign Language Detection System</strong></p>
    <p>Sistem deteksi bahasa isyarat Indonesia yang menggunakan teknologi Computer Vision dan Machine Learning</p>
</div>
""", unsafe_allow_html=True)
