# Experiment 01

| Field | Isi |
|-------|-----|
| ID | EXP-01 |
| Tanggal | 2026-08-03 |
| Skenario | RM1 (within-corpus) |
| Seed | 42 |
| Status | berhasil |
| Durasi | ±0 menit |

## Tujuan

Model of record. Dipakai untuk RM3 dan prototipe Streamlit.

## Perbedaan terhadap baseline

Tidak ada.

Seluruh run memakai `tf.config.experimental.enable_op_determinism()` sehingga
hasilnya reproduksibel sesuai KNF-06.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 60 |
| Epoch terbaik | 48 |
| Macro F1 validasi | 0.8553 |
| Waktu per epoch | (isi) |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| TESS | 421 | 0.9976 | 0.9976 |
| RAVDESS | 156 | 0.4679 | 0.4727 |
| SAVEE | 120 | 0.5333 | 0.4720 |
| Rata-rata antar korpus | 697 | - | 0.6474 |
| Gabungan | 697 | 0.7991 | 0.8004 |

## Pengamatan

TESS mencapai 0.9976 sementara korpus lain berada jauh
di bawahnya. Penyebabnya struktural: 270 dari 271 pasangan speaker dan kata
pembawa pada data uji TESS juga terdapat pada data latih (99,6 persen),
karena TESS hanya memiliki dua speaker. Angka gabungan karenanya lebih
menggambarkan proporsi korpus daripada kemampuan model.

## Keputusan

Diterima sebagai hasil yang dilaporkan.

## Artefak

`data/models/rm1/seed_42/` (lima berkas)
