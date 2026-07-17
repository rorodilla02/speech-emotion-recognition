# Checkpoint 02 - Dataset Audit

## 📅 Progress Status

**Milestone:** Data Understanding (CRISP-DM)

Status: 🟨 In Progress

---

# 🎯 Tujuan Milestone

Membangun sistem audit dataset yang reusable untuk seluruh dataset Speech Emotion Recognition (SER) tanpa melakukan preprocessing, feature extraction, maupun training model.

Audit bertujuan menghasilkan metadata lengkap mengenai kualitas dan karakteristik dataset sebagai fondasi tahap Data Understanding.

---

# ✅ Progress yang Telah Diselesaikan

## Struktur Project

Struktur project telah disesuaikan dengan pendekatan modular.

```
src/
└── ser/
    ├── audit/
    │   └── auditor.py
    │
    ├── datasets/
    │   ├── common.py
    │   ├── ravdess.py
    │   ├── tess.py
    │   ├── savee.py
    │   └── inesco.py
    │
    ├── output/
    │   └── writer.py
    │
    ├── statistics/
    │   └── generator.py
    │
    └── utils/
        └── audio.py
```

---

## Shared Components (`common.py`)

Telah dibuat beberapa komponen bersama agar seluruh parser menggunakan struktur data yang konsisten.

### AudioMetadata

Digunakan untuk menyimpan metadata setiap file audio yang berhasil diparse.

Field:

- dataset
- filepath
- filename
- speaker
- raw_label
- emotion
- sample_rate
- duration
- status
- error_message

---

### ParseErrorType

Enum untuk standarisasi jenis error parser.

Saat ini terdiri dari:

- INVALID_FILENAME
- UNKNOWN_EMOTION
- AUDIO_READ_ERROR
- UNEXPECTED_ERROR

---

### FailedFile

Digunakan untuk mencatat file yang gagal diproses selama audit.

Field:

- dataset
- filepath
- filename
- error_type
- error_message

---

### ParseResult

Digunakan sebagai object hasil parser.

Berisi:

- metadata
- failed_files

---

### Emotion Mapping

Telah dibuat:

- RAVDESS_EMOTION_MAP
- TESS_EMOTION_MAP
- SAVEE_EMOTION_MAP
- INESCO_EMOTION_MAP

Seluruh mapping digunakan untuk mengubah label asli dataset menjadi label emosi yang terstandarisasi.

---

### BaseParser

Digunakan sebagai abstract interface seluruh parser dataset.

Method:

- parse()

---

### DatasetStatistics

Digunakan untuk menyimpan hasil statistik setiap dataset.

Field:

- dataset
- total_files
- speakers
- total_speakers
- sample_rates
- min_duration
- max_duration
- mean_duration
- emotion_distribution

---

# Dataset Parser

Status:

✅ Selesai

Parser yang telah selesai:

- RAVDESS
- TESS
- SAVEE
- INESCO

Seluruh parser memiliki interface yang sama dan mengembalikan ParseResult.

---

# Audio Reader

Status:

✅ Selesai

Fungsi:

- membaca sample rate
- membaca durasi audio
- mengisi AudioMetadata
- melempar RuntimeError apabila audio gagal dibaca

---

# Statistics Generator

✅ Selesai

Menghasilkan statistik setiap dataset.

Statistik meliputi:

- jumlah file
- jumlah speaker
- sample rate unik
- durasi minimum
- durasi maksimum
- durasi rata-rata
- distribusi emosi

---

# Output Writer

Status:

✅ Selesai

Bertanggung jawab menghasilkan seluruh output audit.

Output yang dihasilkan:

- audit_summary.csv
- audit_summary.json
- dataset_statistics.csv
- file_inventory.csv
- emotion_distribution.csv
- failed_files.csv
- label_mapping.csv
- audit_report.md

---

# 🏗 Keputusan Desain

Selama implementasi diputuskan beberapa desain penting.

## 1. Shared Dataclass

Seluruh parser menggunakan dataclass yang sama.

Tujuan:

- konsisten
- reusable
- mudah dikembangkan

---

## 2. Error Handling Terpusat

Parser tidak menghentikan proses audit ketika menemukan file bermasalah.

Seluruh error dicatat sebagai FailedFile.

---

## 3. ParseResult

Parser tidak mengembalikan tuple.

Sebagai gantinya digunakan ParseResult agar interface parser tetap konsisten.

---

## 4. DATASET_NAME

Setiap parser memiliki konstanta DATASET_NAME untuk menghindari hardcoded string yang berulang.

---

# 📌 Progress Saat Ini

## Parser

- [x] RAVDESS
- [x] TESS
- [x] SAVEE
- [x] INESCO

---

## Audio

- [x] Audio Reader

---

## Auditor

- [x] Dataset Auditor

---

## Statistics

- [x] Statistics Generator

---

## Output Writer

- [x] CSV Writer
- [x] JSON Writer
- [x] Markdown Report

---

## Validation

- [ ] Verify generated metadata
- [ ] Verify dataset statistics

---

# 🎯 Target Output

Folder metadata nantinya akan menghasilkan:

- audit_summary.csv
- audit_summary.json
- dataset_statistics.csv
- file_inventory.csv
- emotion_distribution.csv
- failed_files.csv
- audit_report.md
- label_mapping.csv

---

# 🚀 Next Session

Target implementasi berikutnya:

1. Implementasi Entry Point (01_dataset_audit.py)
2. Generate seluruh metadata
3. Verifikasi output audit

---

# 📝 Catatan

Tahap ini masih berada pada fase **Data Understanding**.

Belum dilakukan:

- preprocessing
- feature extraction
- data balancing
- training model
- evaluasi model

Semua implementasi saat ini hanya berfokus pada proses audit dataset.

---

# Keputusan Arsitektur

- Parser hanya membaca metadata dari filename.
- Parser tidak membaca isi audio.
- Pembacaan audio dipisahkan ke Audio Reader.
- Audio Reader hanya membaca informasi metadata audio (sample rate dan duration).
- Audio Reader tidak melakukan preprocessing.
- Seluruh parser menggunakan ParseResult sebagai output.
- DatasetAuditor juga menggunakan ParseResult sebagai output.
- Seluruh parser menggunakan dataclass bersama pada common.py.
- Seluruh parser memiliki mekanisme error handling yang sama.
- Seluruh parser menggunakan DATASET_NAME untuk menghindari hardcoded string.
- DatasetAuditor bertugas mengintegrasikan hasil parser dan Audio Reader.
- AudioReader bertanggung jawab membaca isi file audio.
- DatasetAuditor mengorkestrasi parser dan AudioReader.
- StatisticsGenerator hanya menghitung statistik tanpa menulis output.
- StatisticsGenerator hanya menghasilkan statistik dan tidak melakukan penulisan file.
- OutputWriter bertanggung jawab menghasilkan seluruh artefak output audit.
- Setiap komponen memiliki single responsibility.
- Seluruh output dihasilkan melalui OutputWriter agar proses ekspor tetap terpusat.