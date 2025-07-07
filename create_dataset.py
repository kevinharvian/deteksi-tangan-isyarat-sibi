import os
import pickle

import mediapipe as mp
import cv2
from tqdm import tqdm  # Tambahkan ini

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Lewati jika bukan folder

    print(f"\nMemproses label: {dir_}")  # Tampilkan label yang sedang diproses

    # Tambahkan tqdm untuk progress bar saat memproses gambar
    for img_path in tqdm(os.listdir(dir_path), desc=f"  {dir_}"):
        data_aux = []
        x_ = []
        y_ = []

        img_path_full = os.path.join(dir_path, img_path)
        img = cv2.imread(img_path_full)
        if img is None:
            print(f"  ⚠️ Gagal membaca gambar: {img_path_full}")
            continue

        img = cv2.resize(img, (640, 480))  # Resize untuk mempercepat
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Simpan dataset ke file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset selesai. Total data: {len(data)} gesture disimpan di 'data.pickle'")
