import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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

st.title("Deteksi Tangan Bahasa Isyarat SIBI")

class HandSignTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = None
        self.reset_time = 0
        self.sentence = ""
        self.last_gesture_time = time.time()
        self.has_started = False
        self.cooldown = 1.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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
            if predicted_character == self.last_prediction:
                if self.reset_time == 0:
                    self.reset_time = now
                elif now - self.reset_time >= self.cooldown:
                    self.sentence += predicted_character
                    self.reset_time = 0
                    self.last_prediction = None
                    self.has_started = True
            else:
                self.last_prediction = predicted_character
                self.reset_time = now

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max_x * W) + 10
            y2 = int(max_y * H) + 10

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            self.last_gesture_time = now
        else:
            if self.has_started and time.time() - self.last_gesture_time > 1:
                if not self.sentence.endswith(" "):
                    self.sentence += " "
                    self.last_prediction = None
                    self.reset_time = 0

        return img

ctx = webrtc_streamer(key="deteksi-gambar", video_transformer_factory=HandSignTransformer)

if ctx.video_transformer:
    st.markdown(f"### 📝 Kalimat: `{ctx.video_transformer.sentence}`")
