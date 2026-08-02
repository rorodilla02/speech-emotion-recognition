# Experiment Log

Catatan setiap kali model dilatih atau dievaluasi. Satu berkas untuk satu
run, bukan satu berkas untuk satu checkpoint.

## Aturan

- Satu run menghasilkan satu berkas `experiment_NN.md`. Run yang gagal atau
  dibatalkan tetap dicatat, karena kegagalan juga informasi.
- Nomor tidak pernah dipakai ulang dan tidak pernah dihapus.
- Berkas diisi **setelah** run selesai, memakai angka dari berkas metrik,
  bukan dari ingatan atau dari tangkapan layar terminal.
- Bagian "Perbedaan terhadap baseline" wajib diisi. Bila ada parameter yang
  diubah tanpa dicatat di sini, perbandingan antar run kehilangan dasar.
- Commit hash dicatat agar setiap angka bisa ditelusuri ke keadaan kode saat
  run dijalankan. Ini hanya berguna bila commit dilakukan per checkpoint.

## Indeks

| ID | Tanggal | Skenario | Seed | Macro F1 uji | Status | Keterangan |
|----|---------|----------|------|--------------|--------|------------|
| | | | | | | |

## Hubungan dengan development log

| Berkas | Isi |
|--------|-----|
| `docs/development_log/04_modeling.md` | Keputusan desain dan status checkpoint. Berubah jarang. |
| `docs/experiments/experiment_NN.md` | Hasil satu kali eksekusi. Bertambah terus. |

Angka final yang masuk Bab 4 diambil dari berkas metrik, dirujuk lewat
experiment log, dan diringkas pada development log. Jangan menyalin angka
antar dokumen secara manual.