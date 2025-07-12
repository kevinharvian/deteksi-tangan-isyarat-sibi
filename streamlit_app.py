import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
import pickle
import cv2
import av
import time

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {i: chr(65 + i) if i < 26 else str(i - 25) for i in range(36)}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class HandSignTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = ""
        self.last_time = time.time()
        self.cooldown = 1.0
        self.sentence = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            x_ = [lm.x for lm in hand.landmark]
            y_ = [lm.y for lm in hand.landmark]
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            width = max_x - min_x if max_x - min_x != 0 else 1e-6
            height = max_y - min_y if max_y - min_y != 0 else 1e-6

            data_aux = []
            for lm in hand.landmark:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                data_aux.extend([norm_x, norm_y])

            prediction = model.predict([np.array(data_aux)])
            label = labels_dict[int(prediction[0])]

            now = time.time()
            if label == self.last_prediction:
                if now - self.last_time >= self.cooldown:
                    self.sentence += label
                    self.last_prediction = ""
            else:
                self.last_prediction = label
                self.last_time = now

            cv2.putText(img, label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.putText(img, f"Kalimat: {self.sentence}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img

# UI
st.title("Deteksi Bahasa Isyarat SIBI (via Kamera)")
st.markdown("Gunakan kamera untuk mendeteksi gesture tangan.")
ctx = webrtc_streamer(key="deteksi-gambar", video_transformer_factory=HandSignTransformer)
