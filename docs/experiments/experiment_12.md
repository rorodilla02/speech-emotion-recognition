# Experiment 12

| Field | Isi |
|-------|-----|
| ID | EXP-12 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_3 (Leave-One-Corpus-Out) |
| Data latih | TESS + SAVEE |
| Data uji | RAVDESS |
| Seed | 42 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 37 |
| Epoch terbaik | 25 |
| Macro F1 validasi | 0.8916 |
| Accuracy uji | 0.1771 |
| Macro F1 uji | 0.1417 |
| Waktu per epoch | 14.7 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada RAVDESS | 0.4660 |
| Rentang RM1 | 0,4375-0,4981 |
| Macro F1 RM2 | 0.1417 |
| Selisih | -0.3243 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Happy 39.4%, Angry 38.1%, Surprise 8.7%.
Konsentrasi dua kelas teratas 77.5%.
Kelas dengan F1 nol: tidak ada.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_3/seed_42/`
