# Experiment 19

| Field | Isi |
|-------|-----|
| ID | EXP-19 |
| Tanggal | 2026-08-03 |
| Skenario | RM3 (cross-lingual, INESCO) |
| Model | RM1 seed 46, tanpa pelatihan ulang |
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
| 1 (tujuh kelas) | 0.2419 | 0.2724 | 0.1999 | 1.36x | 44.7% |
| 2 (tiga kelas) | 0.4091 | 0.3661 | 0.3332 | 1.10x | 0,0% |

Jumlah berkas uji: 2398, seluruhnya INESCO.

### Per kelas, mode 2

emotion  precision  recall  f1_score  support
  Angry     0.3854  0.6271    0.4774      759
  Happy     0.4113  0.5089    0.4549      843
    Sad     0.6333  0.0955    0.1659      796

## Perbandingan terhadap RM1

Pembanding memakai RM1 pada tiga kelas yang sama **tanpa TESS**, karena data
uji TESS tidak independen sehingga akan melebih-lebihkan penurunan.

| Pembanding | Mode 1 | Mode 2 |
|-----------|--------|--------|
| RM1 tiga kelas, seluruh korpus | 0.8501 | 0.8628 |
| RM1 tiga kelas, tanpa TESS (rerata lima seed) | 0,5350 | 0,6605 |
| RM3 | 0.2724 | 0.3661 |
| Selisih terhadap tanpa TESS | -0.2626 | -0.2944 |

## Pengamatan

Distribusi prediksi mode 2: Angry 51.5%, Happy 43.5%, Sad 5.0%.
Ketiga kelas terpakai dan tidak ada kelas ber-F1 nol, berbeda dari RM2 yang
mengalami kolaps prediksi. Model mempertahankan kemampuan diskriminatif yang
lemah namun nyata pada lintas-bahasa.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm3/seed_46/`
