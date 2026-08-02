# Experiment 04

| Field | Isi |
|-------|-----|
| ID | EXP-04 |
| Tanggal | 2026-08-03 |
| Skenario | RM1 (within-corpus) |
| Seed | 45 |
| Status | berhasil |
| Durasi | ±0 menit |

## Tujuan

Pembanding variansi. Tidak dipakai sebagai model final.

## Perbedaan terhadap baseline

| Parameter | Baseline | Run ini |\n|---|---|---|\n| random_seed | 42 | 45 |

Seluruh run memakai `tf.config.experimental.enable_op_determinism()` sehingga
hasilnya reproduksibel sesuai KNF-06.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 64 |
| Epoch terbaik | 52 |
| Macro F1 validasi | 0.8451 |
| Waktu per epoch | (isi) |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| TESS | 421 | 1.0000 | 1.0000 |
| RAVDESS | 156 | 0.4359 | 0.4375 |
| SAVEE | 120 | 0.4167 | 0.3490 |
| Rata-rata antar korpus | 697 | - | 0.5955 |
| Gabungan | 697 | 0.7733 | 0.7754 |

## Pengamatan

TESS mencapai 1.0000 sementara korpus lain berada jauh
di bawahnya. Penyebabnya struktural: 270 dari 271 pasangan speaker dan kata
pembawa pada data uji TESS juga terdapat pada data latih (99,6 persen),
karena TESS hanya memiliki dua speaker. Angka gabungan karenanya lebih
menggambarkan proporsi korpus daripada kemampuan model.

## Keputusan

Diterima sebagai pembanding saja.

## Artefak

`data/models/rm1/seed_45/` (lima berkas)
