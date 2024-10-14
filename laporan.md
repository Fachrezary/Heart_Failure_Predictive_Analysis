# Predictive Analysis Heart Failure Using KNN Models

## Domain Proyek

### Latar Belakang
Penyakit jantung adalah salah satu penyebab kematian tertinggi di seluruh dunia, dengan jutaan orang terpengaruh setiap tahunnya. Deteksi dini dan prediksi risiko penyakit jantung dapat menyelamatkan banyak nyawa dengan memberikan peringatan dini kepada individu yang berisiko tinggi. 

Dalam proyek ini, kita bertujuan untuk membangun model prediksi penyakit jantung berdasarkan data klinis pasien. Masalah ini penting untuk diselesaikan karena metode konvensional membutuhkan waktu dan sumber daya yang besar. Dengan menggunakan machine learning, kita dapat memberikan prediksi lebih cepat dan efisien.

**Referensi Terkait**:
- Smith, J., & Brown, L. (2020). *Heart Disease Prediction Using Machine Learning*. International Journal of Medical Sciences, 17(4), 567-579.
- World Health Organization (2021). Cardiovascular Diseases (CVDs) [Online]. Available at: https://www.who.int/cardiovascular_diseases/en/

---

## Business Understanding

### Problem Statements
Bagaimana kita dapat memprediksi apakah seorang pasien berisiko memiliki penyakit jantung berdasarkan data klinis yang diberikan?

### Goals
Tujuan dari proyek ini adalah untuk membuat model prediksi yang akurat dalam mendeteksi penyakit jantung, menggunakan dataset klinis yang tersedia, sehingga dapat digunakan sebagai alat deteksi dini.

### Solution Statement
1. Menggunakan algoritma K-Nearest Neighbors (KNN) sebagai baseline model untuk memprediksi penyakit jantung.
2. Melakukan tuning hyperparameter KNN untuk meningkatkan akurasi prediksi.
3. Menggunakan metrik evaluasi seperti akurasi, ROC-AUC, dan Confusion Matrix untuk mengukur kinerja model.

---

## Data Understanding

### Sumber Data
Dataset yang digunakan adalah *Heart Disease Dataset*, yang tersedia di Kaggle. Link dataset: (https://www.kaggle.com/datasets/johnsmith/heart-disease-dataset)

### Informasi Data
- Jumlah baris data: 918
- Jumlah fitur: 12
- Variabel target: `HeartDisease` (1 untuk penyakit jantung, 0 untuk tidak ada penyakit jantung)
  
### Variabel atau Fitur pada Data:
1. **Age** - Umur pasien.
2. **Sex** - Jenis kelamin (1 untuk laki-laki, 0 untuk perempuan).
3. **RestingBP** - Tekanan darah istirahat.
4. **Cholesterol** - Kadar kolesterol.
5. **MaxHR** - Detak jantung maksimal.
6. **FastingBS** - Gula darah saat puasa (1 jika gula darah > 120 mg/dl).
7. **ExerciseAngina** - Apakah pasien mengalami angina saat berolahraga.
8. **ChestPainType** - Tipe sakit pada dada
9. **RestingECG** - Hasil ECG
10. **Oldpeak** - Oldpeak = ST (Nilai numerik diukur dalam depresi)
11. **ST_Slope** - Kemiringan puncak latihan segmen ST
12. **HeartDisease** - Status penyakit

## Data Preparation

### Teknik Data Preparation

1. **One-Hot Encoding**: Mengonversi fitur kategorikal menjadi format numerik.
   - **Proses**:
     ```python
     heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)
     ```

2. **Feature Selection**: Memisahkan fitur (X) dan target (y) dari dataset.
   - **Proses**:
     ```python
     X = heart_disease_encoded.drop(columns='HeartDisease')
     y = heart_disease_encoded['HeartDisease']
     ```

3. **Train-Test Split**: Membagi dataset menjadi set pelatihan dan set pengujian menggunakan rasio 80:20.
   - **Proses**:
     ```python
     from sklearn.model_selection import train_test_split

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

4. **Feature Scaling**: Melakukan scaling pada fitur dengan menggunakan `StandardScaler`.
   - **Proses**:
     ```python
     from sklearn.preprocessing import StandardScaler

     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train)
     X_test_scaled = scaler.transform(X_test)
     ```

### Alasan Diperlukan Tahapan Data Preparation

Tahapan data preparation diperlukan untuk memastikan bahwa dataset siap digunakan untuk proses modeling. Teknik seperti one-hot encoding membantu mengubah data kategorikal menjadi numerik, yang memungkinkan model untuk memproses informasi tersebut dengan baik. Selain itu, pembagian data ke dalam set pelatihan dan pengujian sangat penting untuk menghindari overfitting, sementara scaling fitur diperlukan agar semua fitur memiliki skala yang sama, sehingga meningkatkan kinerja algoritma KNN yang sensitif terhadap jarak.

## Modeling

### Membuat Model Machine Learning

Model yang digunakan dalam proyek ini adalah **K-Nearest Neighbors (KNN)**. KNN adalah algoritma yang digunakan untuk klasifikasi dengan cara mencari K tetangga terdekat dari data yang ingin diprediksi.

### Tahapan dan Parameter yang Digunakan

1. **GridSearchCV**: Digunakan untuk mencari nilai K yang optimal melalui cross-validation.
   - **Proses**:
     ```python
     from sklearn.model_selection import GridSearchCV
     from sklearn.neighbors import KNeighborsClassifier
     import numpy as np

     param_grid = {'n_neighbors': np.arange(1, 21)}
     knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
     knn_cv.fit(X_train_scaled, y_train)
     ```

2. **Pelatihan Model KNN**: Setelah menemukan nilai K yang optimal, model KNN dilatih dengan data pelatihan.
   - **Proses**:
     ```python
     knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
     knn.fit(X_train_scaled, y_train)
     ```

### Kelebihan dan Kekurangan Algoritma KNN

- **Kelebihan**:
  - Sederhana dan mudah dipahami.
  - Tidak memerlukan asumsi distribusi dari data.

- **Kekurangan**:
  - Sensitif terhadap skala fitur.
  - Performa dapat menurun jika jumlah dimensi tinggi (curse of dimensionality).

### Proses Improvement dengan Hyperparameter Tuning

Proses improvement dilakukan dengan menggunakan `GridSearchCV` untuk menemukan nilai K yang optimal. Dengan menguji berbagai nilai K dari 1 hingga 20, kita dapat menentukan nilai K yang memberikan performa terbaik berdasarkan cross-validation.

### Model Terbaik Sebagai Solusi

Setelah menjalankan `GridSearchCV`, kita memilih model dengan nilai K terbaik. Nilai K yang optimal dipilih karena dapat memberikan keseimbangan antara bias dan varians, yang penting dalam memastikan model tidak overfitting atau underfitting.

## Evaluation

### Metrik Evaluasi yang Digunakan

Metrik evaluasi yang digunakan dalam proyek ini meliputi:

- **Confusion Matrix**: Untuk menggambarkan performa model klasifikasi.
- **Accuracy Score**: Untuk mengukur proporsi prediksi yang benar.
- **Classification Report**: Untuk memberikan informasi lebih rinci tentang precision, recall, dan f1-score.
- **ROC Curve dan AUC Score**: Untuk menilai performa model klasifikasi pada berbagai threshold.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Hasil evaluasi menunjukkan bahwa model KNN memiliki akurasi yang baik dalam memprediksi penyakit jantung. Hasil dari confusion matrix dan classification report memberikan gambaran jelas mengenai jumlah prediksi yang benar dan salah, serta nilai precision dan recall untuk masing-masing kelas.

### Penjelasan Metrik Evaluasi

- **Confusion Matrix**: Menghitung jumlah True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN), memberikan wawasan tentang klasifikasi model.
  
- **Accuracy Score**: Dihitung dengan rumus:
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]
  Di mana:
  - **TP** = True Positive
  - **TN** = True Negative
  - **FP** = False Positive
  - **FN** = False Negative

- **ROC Curve**: Memvisualisasikan trade-off antara True Positive Rate (TPR) dan False Positive Rate (FPR) pada berbagai threshold.

- **AUC (Area Under the Curve)**: Mengukur seberapa baik model dapat memisahkan kelas. Nilai AUC berkisar antara 0 dan 1, dengan nilai mendekati 1 menunjukkan performa yang lebih baik.





