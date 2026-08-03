# Experiment 18

| Field | Isi |
|-------|-----|
| ID | EXP-18 |
| Tanggal | 2026-08-03 |
| Skenario | RM3 (cross-lingual, INESCO) |
| Model | RM1 seed 45, tanpa pelatihan ulang |
| Status | berhasil |

## Tujuan

Menguji kemampuan model RM1 pada ucapan berbahasa Indonesia yang belum
pernah dilihat. Pembanding variansi.

## Perbedaan terhadap baseline

Tidak ada pelatihan. `data/splits/rm3/train.csv` sengaja tidak dipakai
sesuai Opsi A. Macro F1-score dihitung terhadap tiga kelas target saja.

## Hasil

| Mode | Accuracy | Macro F1 | Chance macro F1 | Rasio | Prediksi ke luar kelas target |
|------|----------|----------|-----------------|-------|-------------------------------|
| 1 (tujuh kelas) | 0.2469 | 0.3085 | 0.1999 | 1.54x | 46.0% |
| 2 (tiga kelas) | 0.4233 | 0.4105 | 0.3332 | 1.23x | 0,0% |

Jumlah berkas uji: 2398, seluruhnya INESCO.

### Per kelas, mode 2

emotion  precision  recall  f1_score  support
  Angry     0.4195  0.5283    0.4676      759
  Happy     0.3885  0.5148    0.4429      843
    Sad     0.5538  0.2261    0.3211      796

## Perbandingan terhadap RM1

Pembanding memakai RM1 pada tiga kelas yang sama **tanpa TESS**, karena data
uji TESS tidak independen sehingga akan melebih-lebihkan penurunan.

| Pembanding | Mode 1 | Mode 2 |
|-----------|--------|--------|
| RM1 tiga kelas, seluruh korpus | 0.8239 | 0.8663 |
| RM1 tiga kelas, tanpa TESS (rerata lima seed) | 0,5350 | 0,6605 |
| RM3 | 0.3085 | 0.4105 |
| Selisih terhadap tanpa TESS | -0.2265 | -0.2500 |

## Pengamatan

Distribusi prediksi mode 2: Happy 46.6%, Angry 39.9%, Sad 13.6%.
Ketiga kelas terpakai dan tidak ada kelas ber-F1 nol, berbeda dari RM2 yang
mengalami kolaps prediksi. Model mempertahankan kemampuan diskriminatif yang
lemah namun nyata pada lintas-bahasa.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm3/seed_45/`
