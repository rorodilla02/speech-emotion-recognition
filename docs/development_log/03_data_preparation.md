# Checkpoint 03 - Data Preparation

## 📅 Progress Status

**Milestone:** Data Preparation (CRISP-DM)

Status: 🚧 In Progress

---

# 🎯 Tujuan Milestone

Merancang pipeline Data Preparation yang modular, reproducible, dan konsisten dengan metodologi penelitian sebagai fondasi sebelum memasuki tahap Modeling.

Tahap ini bertujuan untuk:

- menyiapkan pipeline preprocessing audio
- menentukan strategi pembagian dataset (RM1, RM2, RM3)
- menentukan target durasi audio berdasarkan distribusi data
- merancang proses augmentasi data
- menyiapkan pipeline ekstraksi fitur
- menghasilkan dataset yang siap digunakan pada proses training CNN

---

# ✅ Progress yang Telah Diselesaikan

## Review Hasil Audit Dataset

Hasil Dataset Audit pada Checkpoint 02 telah ditinjau kembali dan digunakan sebagai dasar perancangan Data Preparation.

Seluruh metadata dataset akan menggunakan:

- file_inventory.csv

sebagai **single source of truth**.

Pipeline preprocessing tidak melakukan scanning ulang terhadap folder dataset selama metadata telah tersedia.

---

## Pipeline Preprocessing

Urutan preprocessing :

```
Raw Audio

↓

Convert Mono

↓

Resample 16 kHz

↓

Silence Trimming

↓

RMS Amplitude Normalization

↓

Dataset Split

↓

Audio Augmentation (Train Only)

↓

Duration Normalization

↓

Feature Extraction

↓

Feature Validation
```

Urutan ini akan dipertahankan selama tidak terdapat revisi metodologi.

---

## Duration Analysis

Status:

✅ Selesai

Analisis distribusi durasi dilakukan menggunakan metadata hasil Dataset Audit (`file_inventory.csv`) tanpa melakukan pembacaan ulang file audio.

Komponen yang telah diimplementasikan:

- DurationAnalyzer
- DurationVisualizer
- DurationEvaluator

Seluruh komponen dirancang modular dengan prinsip Single Responsibility.

Output yang dihasilkan:

- duration_summary.csv
- duration_evaluation.csv

Visualisasi distribusi durasi digunakan untuk membantu proses penentuan target durasi audio.

---

### Hasil Analisis Durasi

Distribusi durasi dihitung menggunakan seluruh dataset training:

- RAVDESS
- TESS
- SAVEE

Hasil statistik gabungan:

| Statistik | Nilai |
|-----------|-------:|
| Mean | 2.74 detik |
| Median | 2.39 detik |
| P75 | 3.57 detik |
| P90 | 3.97 detik |
| P95 | 4.27 detik |
| Maximum | 7.14 detik |

Distribusi menunjukkan bahwa TESS memiliki durasi audio yang jauh lebih pendek dibandingkan RAVDESS dan SAVEE.

---

## Audio Preprocessing Architecture

Status:

🚧 In Progress

Komponen yang telah diimplementasikan:

- AudioData
- BasePreprocessor
- AudioLoader
- AudioPreprocessor
- MonoConverter
- Resampler
- SilenceTrimmer
- RMSNormalizer

Seluruh komponen dirancang modular dengan prinsip Single Responsibility sehingga setiap tahapan preprocessing dapat digunakan kembali secara independen.

---

## Standarisasi Audio

### Convert Mono

Audio akan dibaca menggunakan:

```
librosa.load(..., mono=True)
```

untuk memastikan seluruh dataset memiliki satu kanal audio.

---

### Resampling

Target sample rate:

```
16000 Hz
```

Keputusan ini didasarkan pada:

- standar umum Speech Emotion Recognition
- RAVDESS telah menggunakan 16 kHz
- downsampling lebih aman dibanding upsampling
- mencukupi rentang frekuensi sinyal ucapan

---

### Silence Trimming

Silence trimming direncanakan menggunakan:

```
librosa.effects.trim()
```

Parameter awal:

```
top_db = 30
```

Nilai ini akan divalidasi secara visual pada tahap implementasi.

---

### RMS Amplitude Normalization

Normalisasi amplitudo dilakukan menggunakan RMS Normalization.

Target RMS:

```
0.1
```

Normalisasi harus menjaga agar sinyal tidak mengalami clipping.

---

# Dataset Split

Split dilakukan **sebelum proses augmentasi** untuk menghindari data leakage.

Strategi split mengikuti rancangan penelitian.

## RM1

- RAVDESS menggunakan speaker-independent split.
- SAVEE menggunakan speaker-independent split.
- TESS menggunakan stratified split berdasarkan label karena hanya memiliki dua speaker.

Validasi yang harus dilakukan setelah split:

- tidak terdapat speaker overlap
- distribusi label tetap terjaga
- distribusi dataset sesuai rancangan penelitian

---

## RM2

Menggunakan Leave-One-Corpus-Out.

Eksperimen:

- RAVDESS + TESS → SAVEE
- RAVDESS + SAVEE → TESS
- TESS + SAVEE → RAVDESS

---

## RM3

Model hanya dilatih menggunakan dataset bahasa Inggris.

INESCO digunakan sepenuhnya sebagai dataset evaluasi cross-lingual dan tidak pernah digunakan pada proses training maupun validation.

---

# Audio Augmentation

Augmentasi hanya diterapkan pada data training.

Jenis augmentasi yang disepakati:

- Noise Injection
- Pitch Shifting

Augmentasi tidak diterapkan pada validation maupun testing.

---

# Duration Normalization

Status:

✅ Final

Target durasi dipilih berdasarkan distribusi gabungan dataset training:

- RAVDESS
- TESS
- SAVEE

Persentil yang dianalisis:

- P75
- P90
- P95

Keputusan akhir:

Target Duration:

4 detik

(hasil pembulatan dari P90 ≈ 3.97 detik)

Strategi normalisasi:

- Audio lebih pendek → Zero Padding
- Audio lebih panjang → Center Crop

Keputusan menggunakan P90 dipilih karena mampu mempertahankan sebagian besar informasi audio tanpa menghasilkan jumlah cropping yang berlebihan.

Distribusi padding dan cropping akan didokumentasikan sebagai karakteristik alami dataset, bukan sebagai proses balancing.

---

# Feature Extraction

Feature Fusion yang digunakan:

- MFCC
- Delta MFCC
- Delta-Delta MFCC
- Chroma

Parameter yang telah disepakati:

MFCC:

- 13 coefficients

Window:

- 25 ms

Hop Length:

- 10 ms

FFT:

- 512

Total feature:

```
13
+
13
+
13
+
12

=

51 feature/frame
```

Output feature akan menjadi input CNN.

---

# Penyimpanan Data

Dataset asli tidak akan diubah.

Pipeline hanya membaca data pada:

```
data/raw/
```

Seluruh hasil preprocessing akan disimpan pada folder baru:

```
data/processed/
```

Feature hasil ekstraksi disimpan pada:

```
data/features/
```

Nama file asli tetap dipertahankan.

Metadata menjadi sumber utama untuk melacak asal-usul setiap file.

---

# 🏗 Keputusan Desain

## 1. Metadata-first Pipeline

Seluruh pipeline menggunakan metadata hasil audit sebagai sumber informasi utama.

Tidak dilakukan scanning ulang dataset apabila metadata telah tersedia.

---

## 2. Immutable Raw Dataset

Seluruh file pada folder raw hanya bersifat read-only.

Pipeline tidak diperbolehkan:

- rename
- overwrite
- edit file asli

---

## 3. Modular Architecture

Setiap tahapan preprocessing akan dipisahkan menjadi modul yang memiliki satu tanggung jawab.

Implementasi akan mengikuti prinsip:

- modular
- reusable
- clean architecture
- single responsibility

---

## 4. Configuration-driven Parameter

Parameter preprocessing akan dipisahkan dari logika program agar mudah dikonfigurasi tanpa mengubah implementasi.

---

## 5. Reproducibility

Seluruh proses yang mengandung unsur acak akan menggunakan random seed yang konsisten agar eksperimen dapat direproduksi.

---

## 6. Metadata-driven Duration Analysis

Analisis durasi dilakukan menggunakan metadata hasil audit tanpa membaca ulang file audio.

Seluruh statistik durasi dihitung dari file_inventory.csv sehingga proses analisis bersifat cepat, reproducible, dan tidak bergantung pada dataset mentah.

---

## 7. Pipeline Visualization

Visualisasi distribusi durasi dipisahkan dari proses analisis statistik.

DurationAnalyzer hanya bertugas menghitung statistik.

DurationVisualizer hanya bertugas menghasilkan visualisasi.

DurationEvaluator hanya bertugas mengevaluasi dampak pemilihan target durasi.

---

# 📌 Progress Saat Ini

## Review

- [x] Review hasil audit
- [x] Sinkronisasi metodologi penelitian
- [x] Finalisasi pipeline preprocessing

---

## Duration Analysis

- [x] DurationAnalyzer
- [x] DurationVisualizer
- [x] DurationEvaluator
- [x] Duration summary
- [x] Duration evaluation
- [x] Target duration selection (4 detik)

---

## Preprocessing Architecture

- [x] AudioData
- [x] BasePreprocessor
- [x] AudioLoader
- [x] AudioPreprocessor
- [x] MonoConverter
- [x] Resampler
- [x] SilenceTrimmer
- [x] RMSNormalizer

---

## Dataset Preprocessing

- [ ] Data cleaning
- [ ] Speaker correction
- [ ] Remove invalid audio
- [ ] Process entire dataset
- [ ] Save processed audio

---

## Dataset Split

### RM1
- [ ] Speaker-independent split
- [ ] Stratified split (TESS)
- [ ] Metadata split

### RM2
- [ ] Leave-One-Corpus-Out

### RM3
- [ ] Cross-lingual evaluation split

---

## Audio Augmentation

- [ ] Noise Injection
- [ ] Pitch Shifting

---

## Duration Normalization

- [ ] Zero Padding
- [ ] Center Crop
- [ ] Fixed length (4 detik)

---

## Feature Extraction

- [ ] MFCC
- [ ] Delta MFCC
- [ ] Delta-Delta MFCC
- [ ] Chroma

---

## Validation

- [ ] Validasi preprocessing
- [ ] Validasi feature
- [ ] Validasi metadata

---

# Known Dataset Issues

Beberapa karakteristik dataset yang masih menjadi perhatian:

- TESS memiliki satu file dengan kode speaker `OA` yang akan dikoreksi menjadi `OAF` pada tahap preprocessing.
- INESCO memiliki satu file (`mbaz_h138.wav`) yang tidak dapat dibaca dan akan dikeluarkan dari pipeline preprocessing.
- TESS memiliki satu file dengan sample rate 96 kHz yang masih memerlukan investigasi lebih lanjut.

---

# 🎯 Target Output

Tahap Data Preparation nantinya akan menghasilkan:

- processed audio
- augmented audio
- metadata split
- feature dataset
- validation report preprocessing

---

# 🚀 Next Session

1. Implementasi Dataset Preprocessing Pipeline
2. Menyimpan processed audio
3. Implementasi RM1 Dataset Split
4. Implementasi RM2 Dataset Split
5. Implementasi RM3 Dataset Split
6. Implementasi Audio Augmentation
7. Implementasi DurationNormalizer
8. Implementasi Feature Extraction
9. Validasi hasil preprocessing

---

# 📝 Catatan

Checkpoint ini berada pada fase implementasi Data Preparation.

Implementasi yang telah selesai meliputi analisis durasi audio, penentuan target durasi, serta pembangunan arsitektur modular untuk audio preprocessing.

Tahap berikutnya akan berfokus pada implementasi preprocessing terhadap seluruh dataset, dataset split, augmentasi, duration normalization, dan ekstraksi fitur.

---

# Keputusan Arsitektur

- Pipeline preprocessing menggunakan metadata hasil audit sebagai single source of truth.
- Raw dataset bersifat immutable dan tidak boleh dimodifikasi.
- Seluruh hasil preprocessing disimpan pada folder `data/processed`.
- Nama file asli dipertahankan.
- Seluruh komponen preprocessing mengikuti prinsip Single Responsibility.
- AudioLoader hanya bertugas membaca file audio.
- AudioPreprocessor bertindak sebagai orchestrator pipeline preprocessing.
- Seluruh komponen preprocessing menggunakan interface `BasePreprocessor`.
- `AudioData` digunakan sebagai shared dataclass antar komponen preprocessing.
- Analisis durasi dilakukan menggunakan metadata tanpa membaca ulang file audio.
- Target durasi ditetapkan menggunakan distribusi gabungan dataset training.
- Seluruh pipeline dirancang modular, reproducible, dan mudah diperluas.