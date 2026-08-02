# Experiment 02

| Field | Isi |
|-------|-----|
| ID | EXP-02 |
| Tanggal | 2026-08-02 |
| Skenario | RM1 |
| Seed | 42 |
| Commit | (hash commit perbaikan MonitorGuard) |
| Status | berhasil |
| Durasi | ±10 menit |

## Tujuan

Training dan evaluasi RM1 within-corpus setelah perbaikan nama metrik
pantauan pada EXP-01.

## Perbedaan terhadap baseline

Tidak ada.

## Hasil

| Metrik | Nilai |
|--------|-------|
| Epoch dijalankan | 64 |
| Epoch terbaik | 52 |
| Macro F1 validasi | 0,8408 |
| Macro F1 uji (gabungan) | 0,7886 |
| Macro F1 uji (rata-rata antar korpus) | 0,6091 |
| Accuracy uji | 0,7891 |
| Waktu per epoch | 9,0 detik |
| Puncak VRAM | 1,01 GB |

### Per korpus (data uji)

| Korpus | n | Accuracy | Macro F1 |
|--------|---|----------|----------|
| TESS | 421 | 1,0000 | 1,0000 |
| RAVDESS | 156 | 0,4808 | 0,4756 |
| SAVEE | 120 | 0,4500 | 0,3518 |

### Per kelas

Terendah: Fear (F1 0,6872, recall 0,6768).
Tertinggi: Neutral (F1 0,8559), namun precision 0,7917 dengan recall 0,9314
menunjukkan kelas ini terlalu sering diprediksi.
Angka ini bersifat gabungan tiga korpus dan didominasi TESS yang sempurna,
sehingga rincian per korpus dihitung ulang pada checkpoint 6.

## Pengamatan

TESS mencapai macro F1 sempurna pada 421 sampel. Penyebabnya bersifat
struktural: split TESS bersifat stratified sehingga kedua speaker muncul
pada train, validation, dan test sekaligus, dan seluruh korpus memakai
kalimat pembawa serta kondisi rekaman yang sama.

Akibatnya angka gabungan 0,7886 lebih menggambarkan proporsi TESS pada data
uji (60,4%) daripada kemampuan model. Performa pada kondisi
speaker-independent berada di kisaran 0,35 sampai 0,48.

Verifikasi kuantitatif: irisan nama berkas train-test 0, sehingga tidak ada
kebocoran berkas. Namun 270 dari 271 pasangan (speaker, kata) pada data uji
TESS juga terdapat pada data latih (99,6%). Data uji TESS karenanya tidak
independen secara akustik maupun leksikal, meski splitnya sah.

ReduceLROnPlateau terpicu tujuh kali hingga mencapai batas bawah 1e-5 pada
epoch 63. Dibanding EXP-01 yang berjalan tanpa penurunan learning rate,
puncak val_macro_f1 praktis sama (0,8408 berbanding 0,8435), tetapi
fluktuasi 15 epoch terakhir menyempit dari ±0,036 menjadi ±0,011.
Konfigurasi dipertahankan.

## Keputusan

Diterima sebagai hasil yang dilaporkan, dengan tiga angka sekaligus:
per korpus, gabungan, dan rata-rata antar korpus.

Perbandingan RM2 dilakukan terhadap baseline RM1 pada korpus yang sama,
bukan terhadap angka gabungan.

## Artefak

- `data/models/rm1/best_model.keras`
- `data/models/rm1/training_log.csv`
- `data/models/rm1/predictions.csv`
- `data/models/rm1/metrics_summary.csv`
- `data/models/rm1/metrics_per_class.csv`

## Tindak lanjut

1. Verifikasi kuantitatif penyebab TESS sempurna
2. EXP-03 dan EXP-04: RM1 seed 43 dan 44 untuk estimasi variansi