# Experiment 06

| Field | Isi |
|-------|-----|
| ID | EXP-06 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_1 (Leave-One-Corpus-Out) |
| Data latih | RAVDESS + TESS |
| Data uji | SAVEE |
| Seed | 42 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 53 |
| Epoch terbaik | 41 |
| Macro F1 validasi | 0.9004 |
| Accuracy uji | 0.1708 |
| Macro F1 uji | 0.1191 |
| Waktu per epoch | 17.9 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada SAVEE | 0.4123 |
| Rentang RM1 | 0,3490-0,4819 |
| Macro F1 RM2 | 0.1191 |
| Selisih | -0.2932 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Sad 60.2%, Fear 29.0%, Disgust 5.6%.
Konsentrasi dua kelas teratas 89.2%.
Kelas dengan F1 nol: Neutral.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_1/seed_42/`
