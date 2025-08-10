import os
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5  # Naikkan untuk deteksi yang lebih stabil
)

DATA_DIR = './data'
data = []
labels = []

# Loop tiap label (A-Z, dll)
for label in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n🔍 Memproses label: {label}")

    for file_name in tqdm(os.listdir(folder_path), desc=f"  {label}"):
        file_path = os.path.join(folder_path, file_name)

        # Baca gambar
        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️ Gagal membaca gambar: {file_path}")
            continue

        # Resize supaya tangan lebih kelihatan (opsional: 640x480 kalau terlalu kecil)
        img = cv2.resize(img, (480, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)

            width = max_x - min_x if max_x - min_x != 0 else 1e-6
            height = max_y - min_y if max_y - min_y != 0 else 1e-6

            landmark_vector = []
            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                landmark_vector.append(norm_x)
                landmark_vector.append(norm_y)

            data.append(landmark_vector)
            labels.append(label)
        else:
            print(f"⚠️ Landmark tidak terdeteksi: {file_path}")

# Simpan dataset ke pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset selesai. Total data tersimpan: {len(data)} gesture ke dalam 'data.pickle'")
