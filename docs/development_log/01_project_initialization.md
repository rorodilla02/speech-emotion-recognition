# Checkpoint 01 - Project Initialization

## 📅 Progress Status

**Milestone:** Business Understanding (CRISP-DM)

Status: ✅ Completed

---

# 🎯 Tujuan Milestone

Membangun fondasi proyek yang terstruktur sebelum implementasi sistem Speech Emotion Recognition dimulai.

Tahap ini berfokus pada penyusunan arsitektur project, struktur folder, workflow pengembangan, serta dokumentasi agar seluruh implementasi berikutnya berjalan secara konsisten.

---

# ✅ Progress yang Telah Diselesaikan

## Repository

- Git repository dibuat
- GitHub repository dibuat
- Remote repository dikonfigurasi
- Initial commit dilakukan

---

## Project Structure

Struktur project dibuat mengikuti pendekatan modular.

```
app/
configs/
data/
docs/
scripts/
src/
tests/
```

---

## Python Package

Package utama dibuat pada:

```
src/ser/
```

Seluruh implementasi berikutnya akan ditempatkan di dalam package ini.

---

## Dataset Organization

Direktori dataset dipisahkan menjadi:

```
raw/
processed/
features/
metadata/
```

Agar setiap tahap pipeline memiliki lokasi penyimpanan yang jelas.

---

## Documentation

Dokumentasi awal telah dibuat:

- README.md
- docs/
- checkpoint log

---

## Version Control

Git digunakan sebagai version control.

Workflow yang digunakan:

```
edit
↓

git add

↓

git commit

↓

git push
```

---

# 🏗 Keputusan Desain

## 1. Modular Architecture

Project dipisahkan berdasarkan tanggung jawab setiap komponen.

---

## 2. Reproducibility

Seluruh eksperimen akan menggunakan struktur folder yang konsisten.

---

## 3. Original Dataset Preservation

Dataset asli tidak boleh dimodifikasi.

Seluruh preprocessing dilakukan pada folder terpisah.

---

## 4. Documentation First

Setiap milestone akan didokumentasikan melalui checkpoint development log.

---

# 📌 Progress Saat Ini

## Repository

- [x] Initialized

---

## Documentation

- [x] README

---

## Project Structure

- [x] Folder structure

---

## Dataset

- [x] Dataset organization

---

## Git

- [x] Version control

---

# 🚀 Next Session

1. Implementasi dataset parser
2. Dataset audit
3. Metadata generation

---

# 📝 Catatan

Belum dilakukan:

- dataset parsing
- dataset audit
- preprocessing
- feature extraction
- training model

Tahap ini hanya berfokus pada pembangunan fondasi proyek.

---

# Keputusan Arsitektur

- Menggunakan struktur package Python (`src/ser`).
- Dataset dipisahkan menjadi raw, processed, features, dan metadata.
- Seluruh eksperimen akan mengikuti workflow CRISP-DM.
- Setiap milestone didokumentasikan melalui checkpoint.
- Repository menggunakan Git dan GitHub sebagai version control.