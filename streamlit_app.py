import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

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

st.title("Deteksi Tangan Bahasa Isyarat SIBI")

start_col, stop_col, reset_col, belajar_col = st.columns([1, 1, 1, 1])
with start_col:
    if st.button("▶️ Mulai Deteksi"):
        st.session_state.run = True
with stop_col:
    if st.button("⏸️ Berhenti Deteksi"):
        st.session_state.run = False
with reset_col:
    if st.button("🔁 Reset Kalimat"):
        if st.session_state.sentence.strip():
            st.session_state.saved_sentences.append(st.session_state.sentence.strip())
        st.session_state.sentence = ""
        st.session_state.last_prediction = None
        st.session_state.reset_time = 0
        st.session_state.has_started = False
with belajar_col:
    if st.button("📚 Pembelajaran Bahasa Isyarat"):
        st.session_state.show_pembelajaran = not st.session_state.show_pembelajaran

if st.session_state.run and st.session_state.show_pembelajaran:
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    pembelajaran_placeholder = col2.empty()
    sentence_placeholder = st.empty()
elif st.session_state.run:
    frame_placeholder = st.empty()
    sentence_placeholder = st.empty()
    pembelajaran_placeholder = None
elif st.session_state.show_pembelajaran:
    st.image("abjad.png", caption="Kosakata Bahasa Isyarat", use_container_width=True)
    frame_placeholder = None
    sentence_placeholder = None
    pembelajaran_placeholder = None
else:
    frame_placeholder = st.empty()
    sentence_placeholder = st.empty()
    pembelajaran_placeholder = None

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
        show_help = False

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
            sentence_placeholder.markdown(f"### 📝 Kalimat: `{st.session_state.sentence}`")

        if st.session_state.show_pembelajaran and pembelajaran_placeholder:
            pembelajaran_placeholder.image("abjad.png", caption="Kosakata Bahasa Isyarat", use_container_width=True)

    cap.release()
else:
    st.info("Klik tombol ▶️ Mulai Deteksi untuk memulai.")

if st.session_state.saved_sentences:
    st.markdown("---")
    st.subheader("Kalimat Tersimpan")
    saved_copy = st.session_state.saved_sentences.copy()
    for i, sent in enumerate(saved_copy):
        cols = st.columns([6, 1, 1])
        cols[0].write(f"`{sent}`")
        if cols[1].button("🗑️ Hapus", key=f"hapus_{i}"):
            st.session_state.saved_sentences.pop(i)
            st.rerun()
        if cols[2].button("📅 Pakai", key=f"pakai_{i}"):
            st.session_state.sentence = sent
            st.rerun()
