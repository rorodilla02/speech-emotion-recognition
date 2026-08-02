# Experiment 05

| Field | Isi |
|-------|-----|
| ID | EXP-05 |
| Tanggal | 2026-08-03 |
| Skenario | RM1 (within-corpus) |
| Seed | 46 |
| Status | berhasil |
| Durasi | ±0 menit |

## Tujuan

Pembanding variansi. Tidak dipakai sebagai model final.

## Perbedaan terhadap baseline

| Parameter | Baseline | Run ini |\n|---|---|---|\n| random_seed | 42 | 46 |

Seluruh run memakai `tf.config.experimental.enable_op_determinism()` sehingga
hasilnya reproduksibel sesuai KNF-06.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 64 |
| Epoch terbaik | 52 |
| Macro F1 validasi | 0.8433 |
| Waktu per epoch | (isi) |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| TESS | 421 | 1.0000 | 1.0000 |
| RAVDESS | 156 | 0.5192 | 0.4981 |
| SAVEE | 120 | 0.4667 | 0.3961 |
| Rata-rata antar korpus | 697 | - | 0.6314 |
| Gabungan | 697 | 0.8006 | 0.8002 |

## Pengamatan

TESS mencapai 1.0000 sementara korpus lain berada jauh
di bawahnya. Penyebabnya struktural: 270 dari 271 pasangan speaker dan kata
pembawa pada data uji TESS juga terdapat pada data latih (99,6 persen),
karena TESS hanya memiliki dua speaker. Angka gabungan karenanya lebih
menggambarkan proporsi korpus daripada kemampuan model.

## Keputusan

Diterima sebagai pembanding saja.

## Artefak

`data/models/rm1/seed_46/` (lima berkas)

## Rekapitulasi Lima Seed

Sumber: `data/models/rm1/seed_summary.csv`.

| Scope | Rerata | SD | Rentang |
|-------|--------|-----|---------|
| RAVDESS | 0,4660 | 0,0275 | 0,4375-0,4981 |
| SAVEE | 0,4123 | 0,0616 | 0,3490-0,4819 |
| TESS | 0,9986 | 0,0013 | 0,9976-1,0000 |
| Rata-rata antar korpus | 0,6256 | 0,0207 | 0,5955-0,6474 |
| Gabungan | 0,7925 | 0,0108 | 0,7754-0,8004 |

## Acuan Penilaian RM2

Ditetapkan sebelum RM2 dijalankan. Perbandingan dilakukan terhadap baseline
RM1 pada korpus yang sama, bukan terhadap angka gabungan. Penurunan hanya
diklaim bermakna bila seluruh seed RM2 pada suatu fold jatuh di luar rentang
RM1 pada korpus yang sama.

| Fold | Korpus uji | Baseline RM1 | Ambang klaim penurunan |
|------|-----------|--------------|------------------------|
| 1 | SAVEE | 0,4123 | seluruh seed di bawah 0,3490 |
| 2 | TESS | 0,9986 | tidak sebanding, baseline speaker-dependent |
| 3 | RAVDESS | 0,4660 | seluruh seed di bawah 0,4375 |

Uji signifikansi statistik tidak digunakan, karena seed bukan sampel acak
dari suatu populasi dan jumlahnya terlalu kecil.