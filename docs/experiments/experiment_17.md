# Experiment 17

| Field | Isi |
|-------|-----|
| ID | EXP-17 |
| Tanggal | 2026-08-03 |
| Skenario | RM3 (cross-lingual, INESCO) |
| Model | RM1 seed 44, tanpa pelatihan ulang |
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
| 1 (tujuh kelas) | 0.2339 | 0.2775 | 0.1999 | 1.39x | 44.2% |
| 2 (tiga kelas) | 0.3978 | 0.3699 | 0.3332 | 1.11x | 0,0% |

Jumlah berkas uji: 2398, seluruhnya INESCO.

### Per kelas, mode 2

emotion  precision  recall  f1_score  support
  Angry     0.3571  0.5679    0.4385      759
  Happy     0.4134  0.4899    0.4484      843
    Sad     0.5729  0.1382    0.2227      796

## Perbandingan terhadap RM1

Pembanding memakai RM1 pada tiga kelas yang sama **tanpa TESS**, karena data
uji TESS tidak independen sehingga akan melebih-lebihkan penurunan.

| Pembanding | Mode 1 | Mode 2 |
|-----------|--------|--------|
| RM1 tiga kelas, seluruh korpus | 0.8392 | 0.8569 |
| RM1 tiga kelas, tanpa TESS (rerata lima seed) | 0,5350 | 0,6605 |
| RM3 | 0.2775 | 0.3699 |
| Selisih terhadap tanpa TESS | -0.2575 | -0.2906 |

## Pengamatan

Distribusi prediksi mode 2: Angry 50.3%, Happy 41.7%, Sad 8.0%.
Ketiga kelas terpakai dan tidak ada kelas ber-F1 nol, berbeda dari RM2 yang
mengalami kolaps prediksi. Model mempertahankan kemampuan diskriminatif yang
lemah namun nyata pada lintas-bahasa.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm3/seed_44/`
