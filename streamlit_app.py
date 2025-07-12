import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import pickle
import mediapipe as mp
import numpy as np
import time

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Label dictionary
labels_dict = {i: chr(65 + i) if i < 26 else str(i - 25) for i in range(36)}

# Inisialisasi state
for key, value in {
    'sentence': "",
    'last_prediction': None,
    'reset_time': 0,
    'has_started': False,
    'last_gesture_time': time.time(),
    'saved_sentences': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cooldown = 1.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        results = self.hands.process(image_rgb)

        predicted_character = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                elif now - st.session_state.reset_time >= self.cooldown:
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

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            st.session_state.last_gesture_time = time.time()
        else:
            if st.session_state.has_started and time.time() - st.session_state.last_gesture_time > 1:
                if not st.session_state.sentence.endswith(" "):
                    st.session_state.sentence += " "
                    st.session_state.last_prediction = None
                    st.session_state.reset_time = 0

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Tampilan Streamlit
st.title("Deteksi Bahasa Isyarat SIBI (via Kamera)")
st.markdown("Gunakan kamera untuk mendeteksi gesture tangan.")

ctx = webrtc_streamer(
    key="deteksi-gambar",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandSignProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown(f"### 📝 Kalimat Terdeteksi: `{st.session_state.sentence}`")

if st.button("🔁 Reset Kalimat"):
    if st.session_state.sentence.strip():
        st.session_state.saved_sentences.append(st.session_state.sentence.strip())
    st.session_state.sentence = ""
    st.session_state.last_prediction = None
    st.session_state.reset_time = 0
    st.session_state.has_started = False

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
