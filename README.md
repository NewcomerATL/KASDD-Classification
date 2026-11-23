# TP1 - Status Classification (Developed vs Developing)

Proyek ini bertujuan untuk memprediksi status perkembangan suatu negara (**Developed** atau **Developing**) berdasarkan indikator kesehatan dan sosio-ekonomi menggunakan pendekatan model klasifikasi. Performa model dievaluasi menggunakan metrik **F1-Score**.

## 👥 Pembagian Tugas Kelompok

Berikut adalah pembagian eksplorasi dan pemodelan machine learning untuk masing-masing anggota tim:

| Nama Anggota | Model Klasifikasi | Fokus Eksperimen & Pemodelan |
|---|---|---|
| Anderson Tirza | HistGradientBoosting | Gradient boosting berbasis histogram, imputasi KNN, feature engineering (agregasi & interaksi fitur), dan evaluasi via Stratified K-Fold CV |
| Raymundo Rafaelito | CatBoost | Gradient boosting dengan hyperparameter tuning via GridSearchCV, analisis PCA, dan penanganan outlier |
| Jovanus Irwan | XGBoost | Extreme Gradient Boosting (boosting berbasis pohon keputusan) dengan konfigurasi `max_depth` dan `learning_rate` |
| Angelo Benhanan | Random Forest | Ensemble tree-based (bagging), outlier cropping, feature engineering, dan validasi silang |

## 📊 Deskripsi Dataset

Dataset yang digunakan dalam proyek ini terdiri dari dua berkas utama:

- `train.csv` — Berisi 10.070 baris dan 21 kolom (termasuk variabel target `Status`).
- `test.csv` (`test_cla.csv`) — Berisi 1.259 baris dan 19 kolom untuk keperluan prediksi akhir. Berkas ini tidak memiliki kolom `Status`, `Life expectancy`, maupun `id`.

### Temuan Awal (Exploratory Data Analysis)

- **Missing Values:** Terdapat beberapa kolom dengan nilai kosong, dengan persentase tertinggi pada fitur `Population` (5,23%), `Hepatitis B` (4,53%), dan `GDP` (3,60%).
- **Distribusi Target:** Kelas target tidak seimbang. Pada data validasi (2.014 baris), terdapat 338 negara Developed dan 1.676 negara Developing, sehingga F1-Score dipilih sebagai metrik utama.
- **Pertanyaan EDA:** Eksplorasi mencakup pengaruh `Schooling` dan `Status` terhadap `Life expectancy`, korelasi `GDP` dan `Total expenditure` dengan `Life expectancy`, pengaruh cakupan imunisasi terhadap `infant deaths`, serta tren `Life expectancy` tahun 2000–2015 untuk negara Developed vs Developing.
- **Kebocoran Data (Data Leakage):** Kolom `Life expectancy` dibuang dari fitur karena berkorelasi kuat dengan `Status` dan tidak tersedia pada data test.

## ⚙️ Alur Pengerjaan (Workflow)

1. **Load Data & Imports:** Memuat dataset (`train.csv` dan `test.csv`) serta mengimpor pustaka analisis data dan pemodelan (pandas, numpy, scikit-learn, xgboost, catboost, dll.).
2. **Exploratory Data Analysis (EDA):** Menganalisis proporsi missing values, distribusi target, korelasi antar fitur, dan tren `Life expectancy` berdasarkan status negara.
3. **Data Preprocessing & Feature Engineering:**
   - Penanganan nilai kosong (imputasi median/mean atau `KNNImputer`) dan nilai placeholder yang tidak valid.
   - Penanganan outlier dan log-transform pada fitur yang skewed (misalnya `GDP`, `Population`).
   - Pembuatan fitur baru, seperti rata-rata imunisasi, total kematian anak, rata-rata thinness, dan interaksi `Schooling` × `Income composition of resources`.
   - Encoding target: `Developed` → 0, `Developing` → 1.
4. **Modeling & Hyperparameter Tuning:**
   - Membangun dan melatih masing-masing model sesuai pembagian tugas kelompok.
   - Melakukan validasi silang (K-Fold Cross-Validation) dan pencarian parameter terbaik (GridSearchCV pada CatBoost) untuk memaksimalkan F1-Score.
5. **Evaluasi & Submission:** Membandingkan metrik evaluasi (F1-Score, precision, recall, confusion matrix) pada data validasi, melatih ulang model pada seluruh data train, lalu menghasilkan berkas prediksi dengan format `id, Status`.


## 🛠️ Cara Menjalankan Proyek

Pastikan Python 3.10+ sudah terpasang beserta pustaka yang dibutuhkan:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost catboost
```

Letakkan `train.csv` dan `test.csv` (`test_cla.csv`) di direktori yang sama dengan notebook, lalu jalankan:

```bash
jupyter notebook TP01_B06_JMK48.ipynb
```

## Documentation Kaggle:
<img width="1522" height="832" alt="image" src="https://github.com/user-attachments/assets/f044b79c-42ce-4120-b392-1e86bb5ca19a" />
