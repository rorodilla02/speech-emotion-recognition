# Experiment 13

| Field | Isi |
|-------|-----|
| ID | EXP-13 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_3 (Leave-One-Corpus-Out) |
| Data latih | TESS + SAVEE |
| Data uji | RAVDESS |
| Seed | 43 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 41 |
| Epoch terbaik | 29 |
| Macro F1 validasi | 0.8900 |
| Accuracy uji | 0.1619 |
| Macro F1 uji | 0.1201 |
| Waktu per epoch | 14.5 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada RAVDESS | 0.4660 |
| Rentang RM1 | 0,4375-0,4981 |
| Macro F1 RM2 | 0.1201 |
| Selisih | -0.3459 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Angry 49.4%, Happy 32.2%, Fear 7.7%.
Konsentrasi dua kelas teratas 81.7%.
Kelas dengan F1 nol: Sad.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_3/seed_43/`
