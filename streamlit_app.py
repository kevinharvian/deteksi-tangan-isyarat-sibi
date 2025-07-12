import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
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

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Session state init
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'reset_time' not in st.session_state:
    st.session_state.reset_time = 0

COOLDOWN = 1.0  # seconds

class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_prediction = None
        self.reset_time = 0
        self.last_gesture_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

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
            if predicted_character == self.last_prediction:
                if self.reset_time == 0:
                    self.reset_time = now
                elif now - self.reset_time >= COOLDOWN:
                    st.session_state.sentence += predicted_character
                    self.reset_time = 0
                    self.last_prediction = None
            else:
                self.last_prediction = predicted_character
                self.reset_time = now

            x1 = int(min_x * image.shape[1]) - 10
            y1 = int(min_y * image.shape[0]) - 10
            cv2.rectangle(image, (x1, y1), (x1+60, y1+60), (0, 0, 255), 2)
            cv2.putText(image, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# UI
st.title("Deteksi Bahasa Isyarat SIBI (via Kamera)")
st.markdown("Gunakan kamera untuk mendeteksi gesture tangan.")

ctx = webrtc_streamer(
    key="deteksi-gambar",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandSignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(f"### 📝 Kalimat Terdeteksi: `{st.session_state.sentence}`")
