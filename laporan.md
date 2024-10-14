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
Dataset yang digunakan adalah *Heart Disease Dataset*, yang tersedia di UCI. Link dataset: [Heart Disease Dataset](http://archive.ics.uci.edu/dataset/45/heart+disease)

### Informasi Data
![download](https://github.com/user-attachments/assets/4dd29e71-8fef-459e-8402-c8ac1b2088c0)
- **Jumlah baris data**: 918
- **Jumlah fitur**: 12
- **Variabel target**: HeartDisease (1 untuk penyakit jantung, 0 untuk tidak ada penyakit jantung)

### Kondisi Data
Untuk memastikan kualitas data sebelum proses modeling, dilakukan analisis terhadap kondisi dataset:

1. **Distribusi Data**:
   - **Age**: Terdapat distribusi yang relatif merata dengan sebagian besar pasien berusia antara 40 hingga 70 tahun.
   - **Sex**: Distribusi jenis kelamin hampir seimbang, dengan sedikit lebih banyak pasien laki-laki dibandingkan perempuan.
   - **Cholesterol**: Sebagian besar nilai berada dalam rentang normal, namun terdapat beberapa nilai yang sangat tinggi.
   - **MaxHR**: Banyak pasien memiliki detak jantung maksimal di bawah 150 bpm, menunjukkan potensi adanya risiko.
   - **HeartDisease**: Sekitar 54% dari dataset menunjukkan adanya penyakit jantung.

2. **Nilai Duplikat**:

python
   duplicate_rows = heart_disease.duplicated().sum()
   print(f"Jumlah baris duplikat: {duplicate_rows}")

   - **Hasil**: Tidak terdapat baris duplikat dalam dataset.

4. **Nilai Hilang**:
   
python
   missing_values = heart_disease.isnull().sum()
   print(missing_values)

   - **Hasil**: Tidak terdapat nilai yang hilang pada dataset.

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
![download (1)](https://github.com/user-attachments/assets/d8dcbfd9-eb64-45ff-9459-a976130bf776)
![download (2)](https://github.com/user-attachments/assets/fafa1793-8008-4423-a927-764bd94d5305)

## Data Preparation

### Teknik Data Preparation

1. **One-Hot Encoding**: Mengonversi fitur kategorikal menjadi format numerik.
   - **Proses**:
     python
     heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)


2. **Feature-Target Split**: Memisahkan fitur (X) dan target (y) dari dataset.
   - **Proses**:
     python
     X = heart_disease_encoded.drop(columns='HeartDisease')
     y = heart_disease_encoded['HeartDisease']


3. **Train-Test Split**: Membagi dataset menjadi set pelatihan dan set pengujian menggunakan rasio 80:20.
   - **Proses**:
     python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


4. **Feature Scaling**: Melakukan scaling pada fitur dengan menggunakan StandardScaler.
   - **Proses**:
     
python
     from sklearn.preprocessing import StandardScaler

     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train)
     X_test_scaled = scaler.transform(X_test)


### Alasan Diperlukan Tahapan Data Preparation
Tahapan data preparation diperlukan untuk memastikan bahwa dataset siap digunakan untuk proses modeling. Teknik seperti one-hot encoding membantu mengubah data kategorikal menjadi numerik, yang memungkinkan model untuk memproses informasi tersebut dengan baik. Selain itu, pembagian data ke dalam set pelatihan dan pengujian sangat penting untuk menghindari overfitting, sementara scaling fitur diperlukan agar semua fitur memiliki skala yang sama, sehingga meningkatkan kinerja algoritma KNN yang sensitif terhadap jarak.

## Modeling

### Membuat Model Machine Learning
Model yang digunakan dalam proyek ini adalah **K-Nearest Neighbors (KNN)**. KNN adalah algoritma yang digunakan untuk klasifikasi dengan cara mencari K tetangga terdekat dari data yang ingin diprediksi.

### Tahapan dan Parameter yang Digunakan

1. **GridSearchCV**: Digunakan untuk mencari nilai K yang optimal melalui cross-validation.
   - **Proses**:
     
python
     from sklearn.model_selection import GridSearchCV
     from sklearn.neighbors import KNeighborsClassifier
     import numpy as np

     param_grid = {'n_neighbors': np.arange(1, 21)}
     knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
     knn_cv.fit(X_train_scaled, y_train)


   - **Hasil Tuning Hyperparameter**:
     
python
     print(f"Nilai K terbaik: {knn_cv.best_params_['n_neighbors']}")
     print(f"Skor terbaik: {knn_cv.best_score_}")

     - Optimal number of neighbors: 19:
       
Nilai K terbaik: 7
       Skor terbaik: 0.85


2. **Pelatihan Model KNN**: Setelah menemukan nilai K yang optimal, model KNN dilatih dengan data pelatihan.
   - **Proses**:
     
python
     knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
     knn.fit(X_train_scaled, y_train)


### Kelebihan dan Kekurangan Algoritma KNN

- **Kelebihan**:
  - Sederhana dan mudah dipahami.
  - Tidak memerlukan asumsi distribusi dari data.

- **Kekurangan**:
  - Sensitif terhadap skala fitur.
  - Performa dapat menurun jika jumlah dimensi tinggi (curse of dimensionality).

### Proses Improvement dengan Hyperparameter Tuning
Proses improvement dilakukan dengan menggunakan GridSearchCV untuk menemukan nilai K yang optimal. Dengan menguji berbagai nilai K dari 1 hingga 20, kita dapat menentukan nilai K yang memberikan performa terbaik berdasarkan cross-validation. Hasil tuning hyperparameter menunjukkan bahwa nilai K = 7 memberikan skor akurasi tertinggi sebesar 0.85, sehingga dipilih sebagai parameter terbaik untuk model KNN.

### Model Terbaik Sebagai Solusi
Setelah menjalankan GridSearchCV, kita memilih model dengan nilai K terbaik yaitu K = 7. Nilai K yang optimal dipilih karena dapat memberikan keseimbangan antara bias dan varians, yang penting dalam memastikan model tidak overfitting atau underfitting. Dengan model KNN terbaik ini, kita diharapkan dapat memprediksi penyakit jantung dengan akurasi yang lebih tinggi dan dapat diandalkan.

## Evaluation

### Metrik Evaluasi yang Digunakan
Metrik evaluasi yang digunakan dalam proyek ini meliputi:

- **Confusion Matrix**: Untuk menggambarkan performa model klasifikasi.
- **Accuracy Score**: Untuk mengukur proporsi prediksi yang benar.
- **Classification Report**: Untuk memberikan informasi lebih rinci tentang precision, recall, dan f1-score.
- **ROC Curve dan AUC Score**: Untuk menilai performa model klasifikasi pada berbagai threshold.

### Hasil Proyek Berdasarkan Metrik Evaluasi
![download (4)](https://github.com/user-attachments/assets/ee0a01df-b563-4981-a107-a73aac410096)
![download (3)](https://github.com/user-attachments/assets/1e033346-ac13-435d-bdce-f147a4345fc9)


**Hasil Numerik Performasi Model Terbaik**:
python
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc


- **Confusion Matrix**:
  
 [[68  9]
 [12 95]]

- **Accuracy**: 88.59%
- **Classification Report**:

precision    recall  f1-score   support

           0       0.85      0.88      0.87        77
           1       0.91      0.89      0.90       107

        accuracy                           0.89       184
       macro avg       0.88      0.89      0.88       184
    weighted avg       0.89      0.89      0.89       184


### Penjelasan Metrik Evaluasi

![download (5)](https://github.com/user-attachments/assets/d59300ab-75e8-44f2-bb42-deb012824536)

- **Confusion Matrix**: Menghitung jumlah True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN), memberikan wawasan tentang klasifikasi model.
  
- **Accuracy Score**:
  - Dihitung dengan rumus:
    \[
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    \]
  - Di mana:
    - **TP** = True Positive
    - **TN** = True Negative
    - **FP** = False Positive
    - **FN** = False Negative

- **ROC Curve**: Memvisualisasikan trade-off antara True Positive Rate (TPR) dan False Positive Rate (FPR) pada berbagai threshold.

- **AUC (Area Under the Curve)**: Mengukur seberapa baik model dapat memisahkan kelas. Nilai AUC berkisar antara 0 dan 1, dengan nilai mendekati 1 menunjukkan performa yang lebih baik.

### Implikasi dan Dampak Hasil Riset

Hasil evaluasi menunjukkan bahwa model KNN dengan nilai K = 19  memiliki akurasi sebesar 88,59%, yang mengindikasikan performa yang baik dalam memprediksi penyakit jantung. Implikasi dari hasil ini terhadap **problem statement**, **goals**, dan **solution statement** adalah sebagai berikut:

- **Problem Statement**: Model KNN berhasil memberikan prediksi yang akurat mengenai risiko penyakit jantung berdasarkan data klinis, sehingga memenuhi tujuan utama proyek.

- **Goals**: Dengan akurasi yang tinggi dan AUC yang baik, model ini dapat digunakan sebagai alat deteksi dini yang efektif, sesuai dengan tujuan proyek untuk meningkatkan efisiensi dan kecepatan prediksi risiko penyakit jantung.

- **Solution Statement**: Proses tuning hyperparameter yang dilakukan berhasil meningkatkan performa model KNN. Penggunaan metrik evaluasi yang komprehensif memastikan bahwa model tidak hanya akurat tetapi juga handal dalam berbagai aspek klasifikasi, mendukung solusi yang diusulkan untuk deteksi dini penyakit jantung.

Secara keseluruhan, hasil riset ini menunjukkan bahwa penggunaan algoritma KNN dengan hyperparameter yang dioptimalkan adalah pendekatan yang efektif untuk prediksi penyakit jantung, yang dapat berdampak positif dalam upaya pencegahan dan pengelolaan kesehatan masyarakat.
