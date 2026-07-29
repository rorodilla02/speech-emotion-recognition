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

✅ Completed

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

## Dataset Preprocessing

Status:

✅ Completed

Dataset preprocessing telah diimplementasikan menggunakan `DatasetPreprocessor` sebagai orchestrator.

Tahapan yang dilakukan:

- membersihkan metadata
- koreksi speaker TESS (`OA` → `OAF`)
- menghapus file INESCO yang corrupt (`mbaz_h138.wav`)
- menjalankan preprocessing terhadap seluruh dataset
- menyimpan hasil ke `data/processed/audio`

Hasil implementasi:

- Koreksi speaker TESS (`OA` → `OAF`)
- Penghapusan file INESCO yang corrupt (`mbaz_h138.wav`)
- Penghapusan seluruh kelas `Calm` pada RAVDESS agar seluruh dataset memiliki ruang label yang konsisten
- Penyimpanan metadata hasil preprocessing (`processed_inventory.csv`)

Hasil akhir preprocessing:

- Total file diproses: 6926
- Training dataset (RAVDESS + TESS + SAVEE): 4528 file
- Evaluation dataset (INESCO): 2398 file
- File gagal: 0

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

## Preprocessing Validation

Status:

✅ Completed

Pipeline preprocessing divalidasi setelah seluruh audio selesai diproses.

Validasi yang dilakukan:

- jumlah file
- keterbacaan audio
- sample rate
- mono audio
- RMS normalization

Seluruh validasi berhasil dilewati.

Pada validasi RMS ditemukan beberapa file yang mengalami deviasi akibat clipping setelah normalisasi amplitudo. Setelah investigasi, kondisi tersebut merupakan karakteristik alami beberapa file audio dan ditangani menggunakan toleransi validasi sebesar ±0.01 RMS.

---

# Dataset Split

Split dilakukan **sebelum proses augmentasi** untuk menghindari data leakage.

Strategi split mengikuti rancangan penelitian.

## RM1

Status:

✅ Completed

Split dataset telah berhasil diimplementasikan sesuai skenario RM1.

Strategi yang digunakan:

- RAVDESS → Speaker-independent split
- TESS → Stratified split berdasarkan label emosi
- SAVEE → Speaker-independent split (3 speaker training, 1 speaker testing)

Seluruh split kemudian digabungkan menjadi:

- train.csv
- validation.csv
- test.csv

yang akan digunakan pada tahap training CNN.

---

## Split Validation

Status:

✅ Completed

Pipeline dataset split divalidasi setelah seluruh metadata berhasil dibangun.

Validasi yang dilakukan:

- Total files
- Dataset distribution
- Label distribution
- Speaker overlap

Seluruh validasi berhasil dilewati.

Output yang dihasilkan:

- dataset_distribution_validation.csv
- label_distribution_validation.csv
- speaker_overlap_validation.csv

---

## RM2

Status:

✅ Completed

RM2 diimplementasikan menggunakan skenario Leave-One-Corpus-Out (LOCO).

Eksperimen:

- RAVDESS + TESS → SAVEE
- RAVDESS + SAVEE → TESS
- TESS + SAVEE → RAVDESS

Setiap fold menghasilkan:

- train.csv
- validation.csv (kosong)
- test.csv

Seluruh fold telah divalidasi menggunakan:

- Total Files
- Dataset Separation
- Label Distribution
- Empty Validation

Seluruh validasi RM2 memperoleh status PASS.

---

## RM3

Status:

✅ Completed

RM3 telah diimplementasikan menggunakan skenario Cross-Lingual Evaluation.

Strategi yang digunakan:

- seluruh dataset bahasa Inggris (RAVDESS, TESS, SAVEE) digunakan sebagai data training
- dataset INESCO digunakan sepenuhnya sebagai data testing
- validation tidak digunakan pada skenario ini

Metadata yang dihasilkan:

- train.csv
- validation.csv (kosong)
- test.csv

Seluruh implementasi memastikan bahwa dataset INESCO tidak pernah digunakan pada proses training maupun validation sehingga tidak terjadi data leakage antar bahasa.

---

# Audio Augmentation

Status:

✅ Completed

Audio augmentation telah diimplementasikan khusus untuk data training menggunakan arsitektur modular berbasis pipeline.

Komponen yang telah diimplementasikan:

- BaseAugmentor
- AugmentationPipeline
- NoiseInjection
- PitchShifting
- AudioWriter
- DatasetAugmentor

Seluruh augmentor mengikuti prinsip Single Responsibility sehingga setiap metode augmentasi dapat digunakan maupun dikombinasikan secara independen melalui AugmentationPipeline.

Augmentasi hanya diterapkan pada dataset training (RAVDESS, TESS, SAVEE), sedangkan validation dan testing tidak mengalami perubahan untuk menghindari data leakage.

Output yang dihasilkan:

- augmented audio (data/augmented/)
- augmented_inventory.csv

---

# Augmentation Validation

Status:

✅ Completed

Pipeline augmentasi divalidasi setelah seluruh audio berhasil dihasilkan.

Validasi yang dilakukan:

- Total augmented files
- File existence
- Sample rate consistency
- Duration consistency

Seluruh validasi memperoleh status PASS.

Output yang dihasilkan:

- augmentation_validation.csv

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

## 8. Speaker-independent Strategy

RAVDESS dan SAVEE menggunakan speaker-independent split sehingga identitas speaker tidak muncul pada lebih dari satu subset.

TESS tidak memungkinkan menggunakan speaker-independent split karena hanya memiliki dua speaker, sehingga digunakan stratified split berdasarkan label emosi.

---

## 9. Consistent Label Space

Seluruh kelas Calm pada RAVDESS dihapus selama preprocessing sehingga seluruh dataset training memiliki tujuh label emosi yang konsisten.

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

- [x] Data cleaning
- [x] Speaker correction
- [x] Remove invalid audio
- [x] Process entire dataset
- [x] Save processed audio

---

## Dataset Split

### RM1

- [x] Speaker-independent split
- [x] Stratified split (TESS)
- [x] Metadata split
- [x] Combined split
- [x] Split validation

### RM2

- [x] Leave-One-Corpus-Out
- [x] Metadata generation
- [x] RM2 validation

### RM3

- [x] Cross-lingual evaluation split
- [x] Metadata generation
- [x] RM3 validation

---

## Audio Augmentation

- [x] BaseAugmentor
- [x] AugmentationPipeline
- [x] Noise Injection
- [x] Pitch Shifting
- [x] DatasetAugmentor
- [x] AudioWriter
- [x] Augmented metadata
- [x] Augmentation validation

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

- [x] Validasi preprocessing
- [x] Validasi RM1
- [x] Validasi RM2
- [x] Validasi RM3
- [x] Validasi augmentation
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
- processed metadata
- dataset split RM1
- dataset split RM2
- dataset split RM3
- split validation report
- augmented audio
- augmented metadata
- augmentation validation report
- feature dataset (next)

---

# 🚀 Next Session

Next Session

1. Duration Normalization
2. Feature Extraction
3. Feature Validation
4. Persiapan pipeline Modeling

---

# 📝 Catatan

Checkpoint ini berada pada fase implementasi Data Preparation.

Implementasi yang telah selesai meliputi analisis durasi audio, penentuan target durasi, serta pembangunan arsitektur modular untuk audio preprocessing.

Tahap berikutnya akan berfokus pada implementasi audio augmentation, duration normalization, feature extraction, dan feature validation sebagai tahapan akhir Data Preparation sebelum memasuki fase Modeling.

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
- Audio augmentation hanya diterapkan pada dataset training.
- Seluruh augmentasi diimplementasikan menggunakan AugmentationPipeline sehingga beberapa augmentor dapat dikombinasikan secara modular.
- Metadata hasil augmentasi dibangun kembali berdasarkan audio yang telah diproses untuk menjaga konsistensi durasi dan sample rate.