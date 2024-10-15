# -*- coding: utf-8 -*-
"""Predictive Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rlMIvW21QZr9Pv-VsvrcdbATR43QGkOF

#Predictive Analysis : Heart Failure

Proyek ini berfokus pada analisis prediktif dalam domain kesehatan, dengan perhatian khusus pada penyakit kardiovaskular, terutama gagal jantung. Penyakit kardiovaskular (CVD) adalah salah satu masalah kesehatan paling mendesak di seluruh dunia, menjadi penyebab utama kematian dan morbiditas global. Sebagai contoh, berdasarkan laporan dari Organisasi Kesehatan Dunia (WHO), penyakit kardiovaskular bertanggung jawab atas sekitar 17 juta kematian setiap tahun.

Gagal jantung adalah salah satu akibat yang paling umum dari CVD. Karena itu, deteksi dini sangat penting untuk mengurangi kematian prematur dan meningkatkan kualitas hidup penderita. Dalam konteks ini, machine learning menawarkan peluang untuk memberikan solusi berbasis data untuk prediksi dini penyakit jantung. Dataset yang digunakan dalam proyek ini terdiri dari 12 fitur kesehatan yang dapat dimanfaatkan untuk membangun model prediktif dalam mendeteksi risiko penyakit jantung.

Orang yang memiliki risiko tinggi terhadap penyakit kardiovaskular, seperti mereka yang menderita hipertensi, diabetes, hiperlipidemia, dan kondisi medis lainnya, termasuk dalam kelompok yang paling diuntungkan dari penggunaan deteksi dini berbasis machine learning ini. Harapannya adalah model ini dapat membantu dalam mengidentifikasi kelompok pasien yang memerlukan penanganan lebih awal untuk mencegah komplikasi lebih lanjut.

Referensi :
* [1] World Health Organization (2021). Cardiovascular Diseases (CVDs) [Online]. Available at: https://www.who.int/cardiovascular_diseases/en/
* [2] https://www.sciencedirect.com/science/article/pii/S2001037016300460

## Business Understanding

### Problem Statements

Permasalahan utama yang dihadapi dalam proyek ini adalah : Bagaimana memprediksi apakah seorang pasien memiliki penyakit jantung berdasarkan data riwayat kesehatannya?
Dalam konteks ini, deteksi dini sangat penting karena memungkinkan untuk mengambil tindakan medis yang tepat sebelum komplikasi yang lebih serius berkembang. Prediksi yang akurat dapat memberikan kesempatan bagi pasien untuk mendapatkan perawatan yang lebih efektif.

### Goals

Tujuan utama dari proyek ini adalah membangun model machine learning yang dapat digunakan untuk:
*   Mendeteksi apakah seseorang memiliki penyakit jantung berdasarkan data kesehatan mereka.
*   Mengurangi risiko kematian dini dengan deteksi dini dan intervensi yang lebih cepat.
*   Meningkatkan akurasi prediksi melalui pemanfaatan algoritma machine learning yang optimal.

### Solution Statements

Untuk mencapai tujuan ini, algoritma machine learning akan diterapkan, yaitu K-Nearest Neighbors (K-NN).
K-Nearest Neighbor (K-NN) adalah salah satu algoritma klasifikasi yang paling mudah dipahami. Algoritma ini bekerja dengan memprediksi kelas untuk titik data baru berdasarkan K tetangga terdekat dari titik data tersebut. Jarak antara data dihitung menggunakan metrik seperti jarak Euclidean, di mana data terdekat dengan kategori mayoritas akan menjadi prediksi model.

### Data Understanding

Dataset yang digunakan dalam proyek machine learning ini terdiri dari 918 data observasi yang diperoleh dari Kaggle dan UCI Machine Learning (Repository : http://archive.ics.uci.edu/dataset/45/heart+disease) Dataset ini mencakup 11 fitur utama yang dapat dimanfaatkan untuk memprediksi kemungkinan terjadinya penyakit jantung.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Load dataset
heart_disease = pd.read_csv('drive/MyDrive/Colab Notebooks/heart.csv')
heart_disease.head()

# Data Exploration
print("\nData Summary:\n", heart_disease.describe())
print("\nDataset Info:\n")
heart_disease.info()

"""Pada bagian ini, kita mengeksplorasi data untuk mendapatkan rangkuman statistik dan melihat informasi tentang tipe data serta apakah ada nilai yang hilang."""

# Check missing values
if heart_disease.isnull().sum().sum() == 0:
    print("\nNo missing values detected.\n")

"""### Data Visualization
Visualisasi distribusi variabel target akan membantu kita memahami proporsi individu yang memiliki penyakit jantung di dalam dataset.

Pada tahap ini, kita membuat beberapa grafik untuk memahami distribusi data:

* Distribusi Penyakit Jantung: Menunjukkan proporsi individu yang memiliki atau tidak memiliki penyakit jantung dalam dataset.
* Matriks Korelasi: Grafik heatmap ini menunjukkan hubungan antara fitur-fitur dataset.
* Distribusi Umur, Tekanan Darah, Kolesterol, dan Detak Jantung Maksimal: Setiap fitur visualisasi menunjukkan distribusi data untuk masing-masing variabel kunci.
"""

# Distribution of Target (HeartDisease)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.countplot(x='HeartDisease', data=heart_disease, palette=['#66b3ff', '#ff6666'])
plt.title('Heart Disease Distribution')
plt.xticks(ticks=[0, 1], labels=['No Disease', 'Has Disease'])

plt.subplot(1, 2, 2)
heart_disease['HeartDisease'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True, colors=['#66b3ff', '#ff6666'])
plt.title('Percentage of Heart Disease')
plt.ylabel('')

plt.tight_layout()
plt.show()

"""Pada bagian ini, kita melakukan persiapan data:

* One-Hot Encoding digunakan untuk mengubah variabel kategorikal menjadi numerik.
* Train-Test Split digunakan untuk membagi dataset menjadi data latih dan data uji.
* Scaling: Kita melakukan scaling untuk memastikan setiap fitur berada dalam skala yang sama.
"""

# One-Hot Encoding untuk kolom kategorikal
heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)

# Cek apakah encoding berhasil
heart_disease_encoded.info()

# Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = heart_disease_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Age distribution
plt.subplot(2, 2, 1)
sns.histplot(heart_disease_encoded['Age'], kde=True, color='purple')
plt.title('Age Distribution', fontsize=14, pad=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Resting Blood Pressure
plt.subplot(2, 2, 2)
sns.histplot(heart_disease_encoded['RestingBP'], kde=True, color='green')
plt.title('Resting Blood Pressure Distribution', fontsize=14, pad=15)
plt.xlabel('Resting Blood Pressure (mm Hg)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Cholesterol
plt.subplot(2, 2, 3)
sns.histplot(heart_disease_encoded['Cholesterol'], kde=True, color='red')
plt.title('Cholesterol Distribution', fontsize=14, pad=15)
plt.xlabel('Cholesterol (mg/dL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Max Heart Rate
plt.subplot(2, 2, 4)
sns.histplot(heart_disease_encoded['MaxHR'], kde=True, color='blue')
plt.title('Maximum Heart Rate Distribution', fontsize=14, pad=15)
plt.xlabel('Max Heart Rate (bpm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout(pad=3.0)
plt.show()

"""### Data Preparation
Kami akan memisahkan data menjadi variabel fitur (X) dan variabel target (y), kemudian membagi data menjadi set pelatihan dan pengujian. Setelah itu, kami akan melakukan scaling pada fitur menggunakan StandardScaler.

"""

# Feature Selection
X = heart_disease_encoded.drop(columns='HeartDisease')
y = heart_disease_encoded['HeartDisease']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""### Modeling
Model yang dipilih untuk proyek ini adalah K-Nearest Neighbors (KNN). Kami akan melatih model KNN pada data yang telah diskalakan dan kemudian melakukan prediksi pada set data pengujian.

"""

# KNN with Cross-Validation to select optimal K
param_grid = {'n_neighbors': np.arange(1, 21)}
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_cv.fit(X_train_scaled, y_train)

print(f"Optimal number of neighbors: {knn_cv.best_params_['n_neighbors']}")

# Train the KNN model with optimal K
knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)

"""Kita menggunakan algoritma K-Nearest Neighbors (KNN) untuk memprediksi penyakit jantung dan menggunakan GridSearchCV untuk mencari jumlah tetangga (K) yang optimal.

### Evaluasi Model
Pada bagian ini, kita mengevaluasi model dengan beberapa metrik:
* Confusion Matrix: Tabel ini menunjukkan prediksi benar dan salah untuk setiap kelas.
* Akurasi: Mengukur seberapa baik model dalam melakukan prediksi yang benar.
* ROC Curve dan AUC Score: Grafik ini menunjukkan keseimbangan antara tingkat positif palsu dan tingkat positif benar.
*Pengaruh K-Value pada Akurasi: Grafik ini menunjukkan bagaimana perubahan nilai K memengaruhi akurasi model.
"""

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve and AUC Score
y_pred_prob = knn.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# Visualizing KNN Model's Performance
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    accuracy_scores.append(accuracy_score(y_test, knn_temp.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='--', color='b')
plt.title('Accuracy Scores vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Accuracy Score')
plt.xticks(k_values)
plt.grid(True)
plt.show()

"""### Additional Evaluation"""

# Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""Pada bagian terakhir, kita menampilkan confusion matrix sebagai heatmap untuk memberikan tampilan visual yang lebih jelas tentang prediksi model."""