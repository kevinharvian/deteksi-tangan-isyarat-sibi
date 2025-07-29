import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load dan Ekstraksi Dataset
# -------------------------------
data_dir = './data'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)

label_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
label_dict = {i: label for i, label in enumerate(label_folders)}
label_to_index = {label: idx for idx, label in label_dict.items()}

print("Label mapping:", label_dict)

data = []
labels = []

for label in label_folders:
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue
        image = cv2.resize(image, (480, 480))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

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
                data_aux.append(norm_x)
                data_aux.append(norm_y)

            data.append(data_aux)
            labels.append(label_to_index[label])

if not data:
    print("❌ Tidak ada data yang valid.")
    exit()

# -------------------------------
# 2. Train/Test Split dan Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# ⬇️ Cetak jumlah data
print(f"\n🔢 Total data: {len(data)}")
print(f"🔹 Jumlah data latih : {len(X_train)}")
print(f"🔹 Jumlah data uji    : {len(X_test)}")

# ⬇️ Jumlah data per kelas di data latih
train_counts = Counter(y_train)
test_counts = Counter(y_test)
print("\n📊 Distribusi data latih per kelas:")
for idx in sorted(train_counts):
    print(f"{label_dict[idx]}: {train_counts[idx]} data")

print("\n📊 Distribusi data uji per kelas:")
for idx in sorted(test_counts):
    print(f"{label_dict[idx]}: {test_counts[idx]} data")

# ⬇️ Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# 3. Evaluasi Model
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi model: {acc:.4f} atau {acc*100:.2f}%")

# -------------------------------
# 4. Confusion Matrix (Visual)
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
labels_cm = [label_dict[i] for i in sorted(label_dict.keys())]

plt.figure(figsize=(16, 14), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels_cm, yticklabels=labels_cm,
            linewidths=0.5, linecolor='gray', square=True, cbar=True)

plt.title("Confusion Matrix Huruf SIBI (A–Z)", fontsize=16)
plt.xlabel("Prediksi", fontsize=14)
plt.ylabel("Label Sebenarnya", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix disimpan sebagai 'confusion_matrix.png'")

# -------------------------------
# 5. Precision, Recall, F1-score
# -------------------------------
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("\n=== Evaluasi Per Kelas ===")
for i, label in enumerate(labels_cm):
    print(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-score={f1[i]:.2f}")

# Simpan ke file evaluasi.txt
with open("evaluasi.txt", "w") as f:
    f.write("=== Classification Report ===\n")
    f.write(classification_report(y_test, y_pred, target_names=labels_cm))
print("✅ Hasil evaluasi disimpan di 'evaluasi.txt'")

# -------------------------------
# 6. Simpan Model
# -------------------------------
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': label_dict}, f)
print("✅ Model dan label mapping disimpan sebagai 'model.p'")
