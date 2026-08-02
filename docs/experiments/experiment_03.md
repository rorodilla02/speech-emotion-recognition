# Experiment 03

| Field | Isi |
|-------|-----|
| ID | EXP-03 |
| Tanggal | 2026-08-03 |
| Skenario | RM1 (within-corpus) |
| Seed | 44 |
| Status | berhasil |
| Durasi | ±0 menit |

## Tujuan

Pembanding variansi. Tidak dipakai sebagai model final.

## Perbedaan terhadap baseline

| Parameter | Baseline | Run ini |\n|---|---|---|\n| random_seed | 42 | 44 |

Seluruh run memakai `tf.config.experimental.enable_op_determinism()` sehingga
hasilnya reproduksibel sesuai KNF-06.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 49 |
| Epoch terbaik | 37 |
| Macro F1 validasi | 0.8210 |
| Waktu per epoch | (isi) |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| TESS | 421 | 0.9976 | 0.9976 |
| RAVDESS | 156 | 0.4487 | 0.4377 |
| SAVEE | 120 | 0.5583 | 0.4819 |
| Rata-rata antar korpus | 697 | - | 0.6391 |
| Gabungan | 697 | 0.7991 | 0.7984 |

## Pengamatan

TESS mencapai 0.9976 sementara korpus lain berada jauh
di bawahnya. Penyebabnya struktural: 270 dari 271 pasangan speaker dan kata
pembawa pada data uji TESS juga terdapat pada data latih (99,6 persen),
karena TESS hanya memiliki dua speaker. Angka gabungan karenanya lebih
menggambarkan proporsi korpus daripada kemampuan model.

## Keputusan

Diterima sebagai pembanding saja.

## Artefak

`data/models/rm1/seed_44/` (lima berkas)
