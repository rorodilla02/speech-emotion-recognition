# Experiment 07

| Field | Isi |
|-------|-----|
| ID | EXP-07 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_1 (Leave-One-Corpus-Out) |
| Data latih | RAVDESS + TESS |
| Data uji | SAVEE |
| Seed | 43 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 64 |
| Epoch terbaik | 52 |
| Macro F1 validasi | 0.9065 |
| Accuracy uji | 0.1562 |
| Macro F1 uji | 0.1420 |
| Waktu per epoch | 17.7 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada SAVEE | 0.4123 |
| Rentang RM1 | 0,3490-0,4819 |
| Macro F1 RM2 | 0.1420 |
| Selisih | -0.2703 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Sad 31.5%, Fear 27.9%, Disgust 24.2%.
Konsentrasi dua kelas teratas 59.4%.
Kelas dengan F1 nol: Neutral.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_1/seed_43/`
