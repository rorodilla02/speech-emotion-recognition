# Experiment 01

| Field | Isi |
|-------|-----|
| ID | EXP-01 |
| Tanggal | 2026-08-02 |
| Skenario | RM1 |
| Seed | 42 |
| Commit | feat(models): CNN architecture and training configuration |
| Status | gagal |
| Durasi | (isi) |

## Tujuan

Training dan evaluasi pertama skenario RM1 within-corpus.

## Perbedaan terhadap baseline

Tidak ada perubahan yang disengaja. Terdapat kesalahan penulisan pada
`MONITOR_METRIC`, tertulis `valmacro_f1` alih-alih `val_macro_f1`.

## Hasil

Run dibatalkan. Metrik tidak dilaporkan.

## Pengamatan

Kesalahan nama metrik tidak menghentikan proses. Keras hanya memberi
peringatan ketika nama monitor tidak dikenal, sehingga:

- EarlyStopping tidak berfungsi, training berjalan penuh 100 epoch
- ReduceLROnPlateau tidak pernah menurunkan learning rate
- ModelCheckpoint menyimpan bobot setiap epoch tanpa syarat, sehingga
  `best_model.keras` berisi bobot epoch terakhir, bukan bobot terbaik

Kegagalan baru terdeteksi saat pembacaan training log setelah seluruh
training selesai.

## Keputusan

Ditolak. Run diulang setelah perbaikan `MONITOR_METRIC` dan penambahan
callback `MonitorGuard` yang memvalidasi ketersediaan metrik pantauan pada
akhir epoch pertama.

## Artefak

- `data/models/_failed/exp01_training_log.csv` (kurva 100 epoch, tetap
  dipakai untuk menilai kesesuaian nilai patience)

## Tindak lanjut

EXP-02, RM1 seed 42 dengan konfigurasi yang sudah diperbaiki.