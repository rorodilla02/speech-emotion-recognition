# Experiment 01

| Field | Isi |
|-------|-----|
| ID | EXP-NN |
| Tanggal | YYYY-MM-DD |
| Skenario | RM1 / RM2 fold_n / RM3 |
| Seed | 42 |
| Commit | `git rev-parse --short HEAD` |
| Status | berhasil / gagal / dibatalkan |
| Durasi | mm menit |

## Tujuan

Satu sampai dua kalimat. Apa yang ingin diketahui dari run ini, bukan apa
yang dijalankan.

## Perbedaan terhadap baseline

Baseline adalah `data/models/training_config.json`. Isi hanya parameter yang
berbeda. Bila tidak ada, tulis "tidak ada".

| Parameter | Baseline | Run ini | Alasan |
|-----------|----------|---------|--------|
| | | | |

## Komposisi data

| Subset | Jumlah asli | Setelah augmentasi | Korpus |
|--------|-------------|--------------------|--------|
| Train | | | |
| Validation | | tanpa augmentasi | |
| Test | | tanpa augmentasi | |

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | |
| Epoch terbaik | |
| Macro F1 validasi | |
| Macro F1 uji | |
| Accuracy uji | |
| Waktu per epoch | |
| Puncak VRAM | |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| | | | |

### Per kelas

Rujuk `metrics_per_class.csv`. Sebutkan di sini hanya kelas dengan F1
terendah dan tertinggi beserta dugaan penyebabnya.

## Pengamatan

Hal yang terlihat dari learning curve, confusion matrix, atau perilaku
training. Tulis yang mengganggu, bukan hanya yang bagus. Contoh yang layak
dicatat: validasi berhenti naik jauh sebelum training loss datar, satu kelas
konsisten tertukar dengan kelas lain, early stopping tidak pernah terpicu.

## Keputusan

Pilih satu dan beri alasan:

- diterima sebagai hasil yang dilaporkan
- diterima sebagai pembanding saja
- ditolak, run diulang dengan perubahan (sebutkan perubahannya)

## Artefak

- `data/models/.../best_model.keras`
- `data/models/.../training_log.csv`
- `data/models/.../predictions.csv`
- `data/models/.../metrics_summary.csv`
- `data/models/.../metrics_per_class.csv`

## Tindak lanjut

Apa yang dikerjakan berikutnya sebagai akibat run ini. Bila tidak ada,
tulis "tidak ada".