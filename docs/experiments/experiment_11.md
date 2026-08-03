# Experiment 11

| Field | Isi |
|-------|-----|
| ID | EXP-11 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_2 (Leave-One-Corpus-Out) |
| Data latih | RAVDESS + SAVEE |
| Data uji | TESS |
| Seed | 44 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 33 |
| Epoch terbaik | 21 |
| Macro F1 validasi | 0.5322 |
| Accuracy uji | 0.3646 |
| Macro F1 uji | 0.3320 |
| Waktu per epoch | 7.7 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada TESS | 0.9986 |
| Rentang RM1 | 0,9976-1,0000 |
| Macro F1 RM2 | 0.3320 |
| Selisih | -0.6666 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Fear 24.6%, Disgust 22.1%, Happy 15.0%.
Konsentrasi dua kelas teratas 46.8%.
Kelas dengan F1 nol: tidak ada.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_2/seed_44/`
