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

Target durasi ditetapkan berdasarkan distribusi durasi **setelah preprocessing**
(`duration_summary_processed.csv`), bukan durasi mentah. Analisis awal yang
memakai `file_inventory.csv` tidak sah sebagai dasar keputusan karena silence
trimming memperpendek durasi secara signifikan.

Statistik gabungan dataset training setelah preprocessing:

| Statistik | Nilai |
|-----------|-------:|
| Mean | 2.125 detik |
| Median | 1.984 detik |
| Std | 0.728 detik |
| P75 | 2.301 detik |
| P90 | 2.720 detik |
| P95 | 3.622 detik |
| Maximum | 7.139 detik |

Keputusan akhir:

Target Duration:

4 detik

Alasan pemilihan 4 detik dibanding 3.6 dan 2.7 detik:

- prioritas diberikan pada minimalisasi cropping, karena informasi yang
  dipotong hilang permanen, sedangkan padding masih dapat diabaikan model
- pada target 4 detik hanya 33.54% berkas SAVEE yang ter-crop,
  dibanding 48.75% pada 3.6 detik dan 78.33% pada 2.7 detik
- 4 detik mendekati mean SAVEE (3.661 detik), yaitu korpus dengan
  sebaran durasi terpanjang

Distribusi padding dan cropping pada target 4 detik:

| Dataset | Padding | Cropping | Rata-rata padding |
|---------|--------:|---------:|------------------:|
| RAVDESS | 99.92% | 0.08% | 2.127 detik |
| TESS | 100.00% | 0.00% | 2.026 detik |
| SAVEE | 66.25% | 33.54% | 0.650 detik |
| Combined | 96.40% | 3.58% | 1.908 detik |

## Strategi Normalisasi

Normalisasi durasi dilakukan pada **domain fitur**, bukan domain gelombang.

- Audio lebih panjang dari 4 detik → center crop pada domain gelombang
- Audio lebih pendek → diekstraksi apa adanya, tanpa padding gelombang
- Matriks fitur di-zero-pad simetris pada sumbu waktu hingga 401 frame

Keputusan ini diambil setelah pengujian menunjukkan bahwa zero padding pada
domain gelombang menghasilkan frame keheningan digital yang nilai MFCC-nya
jatuh ke lantai skala desibel. Nilai konstan tersebut merusak statistik
normalisasi fitur dengan dua cara:

- bila normalisasi dihitung atas seluruh frame, representasi wilayah ucapan
  bergeser sekitar 0.6 sigma mengikuti proporsi padding, dan proporsi padding
  berkorelasi dengan identitas korpus
- bila normalisasi dihitung hanya atas frame ucapan, wilayah padding menjadi
  outlier hingga -100 sigma

Padding pada domain fitur bebas dari kedua masalah tersebut. Hasil pengujian
menunjukkan statistik wilayah ucapan menjadi identik (mean 0.000, std 1.000)
untuk seluruh durasi dari 1.056 detik hingga 7.139 detik.

Komponen `DurationNormalizer` karenanya hanya bertugas melakukan center crop.

---

# Feature Extraction

Status:

✅ Completed

Feature Fusion yang digunakan:

- MFCC
- Delta MFCC
- Delta-Delta MFCC
- Chroma

## Parameter Final

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Sample rate | 16000 Hz | mengikuti tahap preprocessing |
| Window | 25 ms (400 sampel) | konfigurasi baku pemrosesan wicara |
| Hop length | 10 ms (160 sampel) | |
| n_fft | 512 | |
| n_mels | 40 | dibatasi agar tidak menghasilkan filter Mel kosong |
| n_mfcc | 13 | |
| Delta width | 5 | setara N=2 pada rumus Delta di subbab 2.3.3 |
| Pre-emphasis | 0.97 | hanya pada cabang MFCC |
| n_chroma | 12 | |
| Chroma tuning | 0.0 | tetap, agar basis Chroma identik antar berkas |

Total fitur per frame:
- 13 (MFCC)
- 13 (Delta)
- 13 (Delta-Delta)
- 12 (Chroma) = 51 fitur/frame

Bentuk keluaran per sampel:
(51, 401)

401 frame merupakan hasil dari 1 + 64000 // 160 pada panjang target 4 detik.

## Pre-emphasis

`librosa.feature.mfcc` tidak menerapkan pre-emphasis secara otomatis, padahal
pipeline MFCC pada subbab 2.3.2 mencantumkannya sebagai tahap pertama.
Pre-emphasis karenanya diterapkan secara eksplisit menggunakan
`librosa.effects.preemphasis` dengan koefisien 0.97.

Pre-emphasis hanya diterapkan pada cabang MFCC. Chroma diekstraksi dari sinyal
tanpa pre-emphasis, karena penguatan frekuensi tinggi akan mendistorsi
distribusi energi tonal yang justru ingin ditangkap Chroma.

## Normalisasi Skala Fitur

Sebelum normalisasi, rentang nilai keempat blok berbeda hingga tiga orde
besaran (MFCC -583 s.d. +34; Chroma 0 s.d. 1). Tanpa penyeragaman skala,
lapisan konvolusi pertama akan didominasi MFCC sehingga feature fusion tidak
benar-benar terjadi.

Strategi yang digunakan:

- MFCC, Delta, dan Delta-Delta → CMVN per berkas (zero mean, unit variance
  per koefisien sepanjang sumbu waktu)
- Chroma → dibiarkan apa adanya, karena nilainya sudah terbatas pada [0, 1]
  dan CMVN akan menghapus profil tonal rata-rata

CMVN per berkas dipilih dibanding StandardScaler global karena:

- bebas kebocoran data secara konstruksi, tidak perlu scaler terpisah untuk
  RM1, tiga fold RM2, dan RM3
- merupakan teknik baku untuk menghilangkan karakteristik kanal rekaman,
  sehingga membantu generalisasi lintas-korpus pada RM2 dan RM3
- tidak menghasilkan artefak scaler yang harus ikut dikirim ke prototipe

## Penyimpanan

Fitur diekstraksi satu kali per berkas unik, lalu disimpan pada satu larik
tunggal. Setiap skenario penelitian hanya mengacu pada indeks baris.

| Berkas | Isi |
|--------|-----|
| `data/features/features.npy` | Larik (11454, 51, 401) float32, ± 0.94 GB |
| `data/features/feature_index.csv` | Indeks tiap baris |
| `data/features/feature_validation.csv` | Laporan validasi |

Komposisi:

- 6926 berkas hasil preprocessing
- 4528 berkas hasil augmentasi
- Total 11454 baris

Pendekatan indeks dipilih karena penyimpanan terpisah per skenario akan
mengekstraksi berkas yang sama sampai lima kali dan membengkakkan ukuran
menjadi sekitar 4 GB. Pendekatan ini sekaligus merupakan mitigasi risiko R-01.

## Aturan Augmentasi pada Penyusunan Subset

Seluruh korpus berbahasa Inggris diaugmentasi, termasuk berkas yang pada RM1
masuk split validasi dan uji, karena RM2 dan RM3 melatih model pada gabungan
korpus yang berbeda.

Pemisahannya ditegakkan pada `FeatureDataset`:

- data augmentasi hanya disertakan pada peran `train`
- salinan augmentasi hanya ikut bila berkas sumbernya berada pada split yang
  sedang dibangun
- peran `validation` dan `test` menolak data augmentasi secara hard-coded

---

# Feature Validation

Status:

✅ Completed

Validasi yang dilakukan:

- Feature row count
- Feature shape
- Feature dtype
- Feature finite (NaN dan Inf, cuplikan acak 500 baris)
- Label space
- INESCO label subset
- Augmentation source
- RM1 validation no augmentation
- RM1 test no augmentation
- RM1 filepath overlap
- RM1 speaker overlap (non-TESS)

Output yang dihasilkan:

- feature_validation.csv

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

- [x] Center crop (domain gelombang)
- [x] Zero padding (domain fitur)
- [x] Fixed length (4 detik / 401 frame)

---

## Feature Extraction

- [x] MFCC
- [x] Delta MFCC
- [x] Delta-Delta MFCC
- [x] Chroma
- [x] Pre-emphasis
- [x] CMVN per berkas
- [x] Feature index

---

## Validation

- [x] Validasi preprocessing
- [x] Validasi RM1
- [x] Validasi RM2
- [x] Validasi RM3
- [x] Validasi augmentation
- [x] Validasi feature
- [x] Validasi metadata

---

# Known Dataset Issues

Beberapa karakteristik dataset yang masih menjadi perhatian:

- TESS memiliki satu file dengan kode speaker `OA` yang akan dikoreksi menjadi `OAF` pada tahap preprocessing.
- INESCO memiliki satu file (`mbaz_h138.wav`) yang tidak dapat dibaca dan akan dikeluarkan dari pipeline preprocessing.
- TESS memiliki satu file dengan sample rate 96 kHz yang masih memerlukan investigasi lebih lanjut.
- TESS memiliki satu berkas dengan sample rate 96 kHz. Berkas tersebut ditangani melalui resampling seragam ke 16 kHz pada tahap preprocessing dan dilaporkan sebagai temuan T-05 pada Bab 3.

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
- feature dataset
- feature index
- feature validation report

---

# 🚀 Next Session

1. Perancangan arsitektur CNN
2. Konfigurasi training (batch size menyesuaikan VRAM 4 GB)
3. Training RM1
4. Training RM2 (tiga fold LOCO)
5. Evaluasi RM3 menggunakan model RM1

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
- Target durasi ditetapkan dari distribusi durasi setelah preprocessing,
  bukan dari durasi mentah.
- Normalisasi durasi dilakukan pada domain fitur untuk menghindari artefak
  keheningan digital pada wilayah padding.
- DurationNormalizer hanya bertugas melakukan center crop.
- Pre-emphasis diterapkan secara eksplisit karena tidak disediakan oleh
  librosa.feature.mfcc.
- Pre-emphasis hanya diterapkan pada cabang MFCC, tidak pada Chroma.
- Normalisasi skala fitur menggunakan CMVN per berkas agar bebas kebocoran
  data dan tidak memerlukan artefak scaler saat deployment.
- Chroma tidak dinormalisasi karena nilainya sudah terbatas pada [0, 1].
- Fitur diekstraksi satu kali per berkas unik dan diacu melalui indeks baris.
- Data augmentasi hanya boleh masuk ke subset dengan peran train, dan hanya
  bila berkas sumbernya berada pada split yang sama.
- Seluruh proses acak menggunakan RANDOM_SEED agar memenuhi KNF-06.