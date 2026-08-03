# Experiment 08

| Field | Isi |
|-------|-----|
| ID | EXP-08 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_1 (Leave-One-Corpus-Out) |
| Data latih | RAVDESS + TESS |
| Data uji | SAVEE |
| Seed | 44 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 48 |
| Epoch terbaik | 36 |
| Macro F1 validasi | 0.9041 |
| Accuracy uji | 0.1521 |
| Macro F1 uji | 0.1297 |
| Waktu per epoch | 17.8 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada SAVEE | 0.4123 |
| Rentang RM1 | 0,3490-0,4819 |
| Macro F1 RM2 | 0.1297 |
| Selisih | -0.2826 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Fear 32.5%, Sad 31.7%, Disgust 22.1%.
Konsentrasi dua kelas teratas 64.2%.
Kelas dengan F1 nol: Neutral.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_1/seed_44/`
