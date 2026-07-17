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
    └── datasets/
        ├── common.py
        ├── ravdess.py
        ├── tess.py
        ├── savee.py
        └── inesco.py
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

- [ ] Audio Reader

---

## Auditor

- [ ] Dataset Auditor

---

## Statistics

- [ ] Statistics Generator

---

## Output Writer

- [ ] CSV Writer
- [ ] JSON Writer
- [ ] Markdown Report

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

1. Implementasi Audio Reader
2. Implementasi Dataset Auditor
3. Implementasi Statistics Generator
4. Implementasi Output Writer

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
- Seluruh parser menggunakan ParseResult sebagai output.
- Seluruh parser menggunakan dataclass bersama pada common.py.
- Seluruh parser memiliki mekanisme error handling yang sama.
- Seluruh parser menggunakan DATASET_NAME untuk menghindari hardcoded string.