import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Data dari data.pickle
# -------------------------------
with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = np.array(dataset['data'])
y_label = np.array(dataset['labels'])

# Buat label dict (alphabetical order)
unique_labels = sorted(list(set(y_label)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y = np.array([label_to_index[l] for l in y_label])  # Convert label ke index

print("🔤 Mapping label:", index_to_label)

# -------------------------------
# 2. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n🔢 Total data: {len(X)}")
print(f"🔹 Data latih : {len(X_train)}")
print(f"🔹 Data uji   : {len(X_test)}")

print("\n📊 Distribusi data latih per kelas:")
for idx in sorted(set(y_train)):
    print(f"{index_to_label[idx]}: {list(y_train).count(idx)} data")

print("\n📊 Distribusi data uji per kelas:")
for idx in sorted(set(y_test)):
    print(f"{index_to_label[idx]}: {list(y_test).count(idx)} data")

# -------------------------------
# 3. Latih Model Random Forest
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluasi Model
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi model: {acc:.4f} atau {acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
labels_cm = [index_to_label[i] for i in sorted(index_to_label)]

plt.figure(figsize=(16, 14), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels_cm, yticklabels=labels_cm,
            linewidths=0.5, linecolor='gray', square=True, cbar=True)
plt.title("Confusion Matrix Huruf SIBI (A–Z)", fontsize=16)
plt.xlabel("Prediksi")
plt.ylabel("Label Sebenarnya")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix disimpan sebagai 'confusion_matrix.png'")

precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("\n=== Evaluasi Per Kelas ===")
for i, label in enumerate(labels_cm):
    print(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-score={f1[i]:.2f}")

with open("evaluasi.txt", "w") as f:
    f.write("=== Classification Report ===\n")
    f.write(classification_report(y_test, y_pred, target_names=labels_cm))
print("✅ Hasil evaluasi disimpan di 'evaluasi.txt'")

# -------------------------------
# 5. Simpan Model
# -------------------------------
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': index_to_label}, f)
print("✅ Model dan label mapping disimpan sebagai 'model.p'")

# -------------------------------
# 6. Kurva Akurasi vs Jumlah Pohon
# -------------------------------
print("\n📈 Menghitung akurasi terhadap variasi jumlah pohon...")
n_values = list(range(10, 201, 10))
train_accuracies = []
test_accuracies = []

for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

plt.figure(figsize=(10, 6))
plt.plot(n_values, train_accuracies, label='Training Accuracy', marker='o', color='skyblue')
plt.plot(n_values, test_accuracies, label='Testing Accuracy', marker='o', color='salmon')
plt.title("Kurva Akurasi vs Jumlah Pohon (n_estimators)")
plt.xlabel("Jumlah Pohon (n_estimators)")
plt.ylabel("Akurasi")
plt.ylim(0.94, 1.01)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("akurasi_vs_jumlah_pohon.png", dpi=300)
plt.show()
print("✅ Grafik kurva akurasi disimpan sebagai 'akurasi_vs_jumlah_pohon.png'")
