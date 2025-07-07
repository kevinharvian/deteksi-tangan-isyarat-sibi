import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load model dan label
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
labels_dict = model_dict['labels']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sentence = ""
last_prediction = None
cooldown = 1  # detik
start_time = time.time()
detecting = False
show_help = False

# Setting kotak ikon help di kiri atas
help_box_size = 50
help_box_padding = 20

def draw_help_icon(frame):
    # Koordinat kiri atas
    x = help_box_padding
    y = help_box_padding
    # Kotak bg icon
    cv2.rectangle(frame, (x, y), (x + help_box_size, y + help_box_size), (50, 150, 250), -1, cv2.LINE_AA)
    # Tanda tanya putih
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "?", (x + 13, y + 37), font, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

def draw_info_panel(frame):
    H, W, _ = frame.shape
    panel_w, panel_h = 300, 150
    x = help_box_padding
    y = help_box_size + 2*help_box_padding

    # Panel bg semi transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (50, 150, 250), -1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Text panduan
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    line_height = 25
    lines = [
        "Panduan:",
        "- Tekan 'S' untuk mulai deteksi",
        "- Tekan 'R' untuk reset kalimat",
        "- Tekan ESC untuk keluar",
        "- Pastikan gerakan tangan jelas",
        "- Tahan gesture 1 detik untuk input"
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 10, y + 30 + i * line_height), font, 0.6, text_color, 1, cv2.LINE_AA)

def is_hand_over_help_box(hand_landmarks, W, H):
    # Cek landmark ujung jari telunjuk (id=8)
    idx_tip = hand_landmarks.landmark[8]
    cx, cy = int(idx_tip.x * W), int(idx_tip.y * H)

    x = help_box_padding
    y = help_box_padding

    if x <= cx <= x + help_box_size and y <= cy <= y + help_box_size:
        return True
    return False

print("Tekan 'S' untuk mulai deteksi, 'R' untuk reset, ESC untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        detecting = True
        start_time = time.time()
        print("Mulai deteksi...")
    elif key == ord('r'):
        sentence = ""
        last_prediction = None
        detecting = False
        print("Reset kalimat")
    elif key == 27:
        print("Keluar program")
        break

    predicted_character = '?'

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Cek apakah tangan di atas ikon help
        show_help = is_hand_over_help_box(hand_landmarks, W, H)

        # Gambar landmark dan koneksi
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Gambar titik koordinat manual
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        min_x, max_x = min(x_), max(x_)
        min_y, max_y = min(y_), max(y_)

        width = max_x - min_x
        height = max_y - min_y

        if width == 0: width = 1e-6
        if height == 0: height = 1e-6

        data_aux = []
        for lm in hand_landmarks.landmark:
            norm_x = (lm.x - min_x) / width
            norm_y = (lm.y - min_y) / height
            data_aux.append(norm_x)
            data_aux.append(norm_y)

        if detecting:
            prediction = model.predict([np.array(data_aux)])
            predicted_index = int(prediction[0])
            predicted_character = labels_dict.get(predicted_index, '?')

            if predicted_character != last_prediction and time.time() - start_time > cooldown:
                sentence += predicted_character
                print("Karakter terdeteksi:", predicted_character)
                print("Kalimat sementara:", sentence)
                last_prediction = predicted_character
                start_time = time.time()

        # Gambar bounding box dan prediksi
        x1 = int(min_x * W) - 10
        y1 = int(min_y * H) - 10
        x2 = int(max_x * W) + 10
        y2 = int(max_y * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        show_help = False

    # Gambar ikon help selalu di kiri atas
    draw_help_icon(frame)

    # Gambar panel help kalau show_help = True
    if show_help:
        draw_info_panel(frame)

    # Tampilkan kalimat di bawah frame
    kalimat_pos_x = 10
    kalimat_pos_y = H - 30
    cv2.putText(frame, f"Kalimat: {sentence}", (kalimat_pos_x, kalimat_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language to Text", frame)

cap.release()
cv2.destroyAllWindows()
