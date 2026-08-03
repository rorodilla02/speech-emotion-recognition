# Experiment 14

| Field | Isi |
|-------|-----|
| ID | EXP-14 |
| Tanggal | 2026-08-03 |
| Skenario | RM2 fold_3 (Leave-One-Corpus-Out) |
| Data latih | TESS + SAVEE |
| Data uji | RAVDESS |
| Seed | 44 |
| Status | berhasil |

## Perbedaan terhadap baseline

Tidak ada. Konfigurasi training identik dengan RM1. Data validasi dibentuk
dari data latih memakai `ValidationSplitter` dengan seed tetap 42, sehingga
komposisinya sama pada seluruh seed.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 39 |
| Epoch terbaik | 27 |
| Macro F1 validasi | 0.8935 |
| Accuracy uji | 0.1795 |
| Macro F1 uji | 0.1498 |
| Waktu per epoch | 14.5 detik |

## Perbandingan terhadap RM1

| | Nilai |
|---|---|
| Baseline RM1 pada RAVDESS | 0.4660 |
| Rentang RM1 | 0,4375-0,4981 |
| Macro F1 RM2 | 0.1498 |
| Selisih | -0.3162 |

## Pengamatan

Tiga kelas terbanyak diprediksi: Angry 46.6%, Happy 27.7%, Neutral 8.2%.
Konsentrasi dua kelas teratas 74.4%.
Kelas dengan F1 nol: tidak ada.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm2/fold_3/seed_44/`
