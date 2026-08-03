# Experiment 16

| Field | Isi |
|-------|-----|
| ID | EXP-16 |
| Tanggal | 2026-08-03 |
| Skenario | RM3 (cross-lingual, INESCO) |
| Model | RM1 seed 43, tanpa pelatihan ulang |
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
| 1 (tujuh kelas) | 0.2127 | 0.2811 | 0.1999 | 1.41x | 52.2% |
| 2 (tiga kelas) | 0.4153 | 0.4162 | 0.3332 | 1.25x | 0,0% |

Jumlah berkas uji: 2398, seluruhnya INESCO.

### Per kelas, mode 2

emotion  precision  recall  f1_score  support
  Angry     0.4550  0.5323    0.4906      759
  Happy     0.3309  0.3227    0.3267      843
    Sad     0.4651  0.4020    0.4313      796

## Perbandingan terhadap RM1

Pembanding memakai RM1 pada tiga kelas yang sama **tanpa TESS**, karena data
uji TESS tidak independen sehingga akan melebih-lebihkan penurunan.

| Pembanding | Mode 1 | Mode 2 |
|-----------|--------|--------|
| RM1 tiga kelas, seluruh korpus | 0.8378 | 0.8735 |
| RM1 tiga kelas, tanpa TESS (rerata lima seed) | 0,5350 | 0,6605 |
| RM3 | 0.2811 | 0.4162 |
| Selisih terhadap tanpa TESS | -0.2539 | -0.2443 |

## Pengamatan

Distribusi prediksi mode 2: Angry 37.0%, Happy 34.3%, Sad 28.7%.
Ketiga kelas terpakai dan tidak ada kelas ber-F1 nol, berbeda dari RM2 yang
mengalami kolaps prediksi. Model mempertahankan kemampuan diskriminatif yang
lemah namun nyata pada lintas-bahasa.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm3/seed_43/`
