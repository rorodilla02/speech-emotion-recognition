# Experiment 09

| Field | Isi |
|-------|-----|
| ID | EXP-09 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_2 (Leave-One-Corpus-Out) |
| Data latih | RAVDESS + SAVEE |
| Data uji | TESS |
| Seed | 42 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 29 |
| Epoch terbaik | 17 |
| Macro F1 validasi | 0.4553 |
| Accuracy uji | 0.1921 |
| Macro F1 uji | 0.1543 |
| Waktu per epoch | 7.8 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada TESS | 0.9986 |
| Rentang RM1 | 0,9976-1,0000 |
| Macro F1 RM2 | 0.1543 |
| Selisih | -0.8443 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Surprise 42.6%, Angry 21.4%, Happy 20.3%.
Konsentrasi dua kelas teratas 64.0%.
Kelas dengan F1 nol: tidak ada.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_2/seed_42/`
