# Experiment 15

| Field | Isi |
|-------|-----|
| ID | EXP-15 |
| Tanggal | 2026-08-03 |
| Skenario | RM3 (cross-lingual, INESCO) |
| Model | RM1 seed 42, tanpa pelatihan ulang |
| Status | berhasil |

## Tujuan

Menguji kemampuan model RM1 pada ucapan berbahasa Indonesia yang belum
pernah dilihat. Model of record.

## Perbedaan terhadap baseline

Tidak ada pelatihan. `data/splits/rm3/train.csv` sengaja tidak dipakai
sesuai Opsi A. Macro F1-score dihitung terhadap tiga kelas target saja.

## Hasil

| Mode | Accuracy | Macro F1 | Chance macro F1 | Rasio | Prediksi ke luar kelas target |
|------|----------|----------|-----------------|-------|-------------------------------|
| 1 (tujuh kelas) | 0.2252 | 0.3002 | 0.1999 | 1.50x | 51.5% |
| 2 (tiga kelas) | 0.4395 | 0.4313 | 0.3332 | 1.29x | 0,0% |

Jumlah berkas uji: 2398, seluruhnya INESCO.

### Per kelas, mode 2

emotion  precision  recall  f1_score  support
  Angry     0.4531  0.4387    0.4458      759
  Happy     0.3968  0.5836    0.4724      843
    Sad     0.5414  0.2877    0.3757      796

## Perbandingan terhadap RM1

Pembanding memakai RM1 pada tiga kelas yang sama **tanpa TESS**, karena data
uji TESS tidak independen sehingga akan melebih-lebihkan penurunan.

| Pembanding | Mode 1 | Mode 2 |
|-----------|--------|--------|
| RM1 tiga kelas, seluruh korpus | 0.8478 | 0.8732 |
| RM1 tiga kelas, tanpa TESS (rerata lima seed) | 0,5350 | 0,6605 |
| RM3 | 0.3002 | 0.4313 |
| Selisih terhadap tanpa TESS | -0.2348 | -0.2292 |

## Pengamatan

Distribusi prediksi mode 2: Happy 51.7%, Angry 30.7%, Sad 17.6%.
Ketiga kelas terpakai dan tidak ada kelas ber-F1 nol, berbeda dari RM2 yang
mengalami kolaps prediksi. Model mempertahankan kemampuan diskriminatif yang
lemah namun nyata pada lintas-bahasa.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm3/seed_42/`
