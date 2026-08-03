# Experiment Log

Catatan setiap kali model dilatih atau dievaluasi. Satu berkas untuk satu
run, bukan satu berkas untuk satu checkpoint.

## Aturan

- Satu run menghasilkan satu berkas `experiment_NN.md`. Run yang gagal atau
  dibatalkan tetap dicatat, karena kegagalan juga informasi.
- Nomor tidak pernah dipakai ulang dan tidak pernah dihapus.
- Berkas diisi setelah run selesai, memakai angka dari berkas metrik, bukan
  dari ingatan atau tangkapan layar terminal.
- Bagian "Perbedaan terhadap baseline" wajib diisi. Bila ada parameter yang
  diubah tanpa dicatat di sini, perbandingan antar run kehilangan dasar.
- Seluruh run pada indeks ini dijalankan dengan `enable_op_determinism`
  aktif, sehingga hasilnya dapat direproduksi dari seed yang sama.

## Indeks

| ID | Tanggal | Skenario | Seed | Macro F1 uji | Status | Keterangan |
|----|---------|----------|------|--------------|--------|------------|
| 01 | 2026-08-03 | RM1 | 42 | 0,8004 / 0,6474 | berhasil | Model of record, dipakai untuk RM3 dan prototipe |
| 02 | 2026-08-03 | RM1 | 43 | 0,7882 / 0,6148 | berhasil | Pembanding variansi |
| 03 | 2026-08-03 | RM1 | 44 | 0,7984 / 0,6391 | berhasil | Pembanding variansi |
| 04 | 2026-08-03 | RM1 | 45 | 0,7754 / 0,5955 | berhasil | Pembanding variansi |
| 05 | 2026-08-03 | RM1 | 46 | 0,8002 / 0,6314 | berhasil | Pembanding variansi, rekapitulasi lima seed |
| 06 | 2026-08-03 | RM2 fold_1 | 42 | 0,1191 | berhasil | Uji SAVEE, selisih -0,2932 |
| 07 | 2026-08-03 | RM2 fold_1 | 43 | 0,1420 | berhasil | Uji SAVEE, selisih -0,2703 |
| 08 | 2026-08-03 | RM2 fold_1 | 44 | 0,1297 | berhasil | Uji SAVEE, selisih -0,2826 |
| 09 | 2026-08-03 | RM2 fold_2 | 42 | 0,1543 | berhasil | Uji TESS, selisih -0,8443 |
| 10 | 2026-08-03 | RM2 fold_2 | 43 | 0,2124 | berhasil | Uji TESS, selisih -0,7862 |
| 11 | 2026-08-03 | RM2 fold_2 | 44 | 0,3320 | berhasil | Uji TESS, selisih -0,6666 |
| 12 | 2026-08-03 | RM2 fold_3 | 42 | 0,1417 | berhasil | Uji RAVDESS, selisih -0,3243 |
| 13 | 2026-08-03 | RM2 fold_3 | 43 | 0,1201 | berhasil | Uji RAVDESS, selisih -0,3459 |
| 14 | 2026-08-03 | RM2 fold_3 | 44 | 0,1498 | berhasil | Uji RAVDESS, selisih -0,3162 |
| 15 | 2026-08-03 | RM3 | 42 | 0,3002 / 0,4313 | berhasil | Model of record, keluar kelas target 51,5% |
| 16 | 2026-08-03 | RM3 | 43 | 0,2811 / 0,4162 | berhasil | Pembanding variansi, keluar kelas target 52,2% |
| 17 | 2026-08-03 | RM3 | 44 | 0,2775 / 0,3699 | berhasil | Pembanding variansi, keluar kelas target 44,2% |
| 18 | 2026-08-03 | RM3 | 45 | 0,3085 / 0,4105 | berhasil | Pembanding variansi, keluar kelas target 46,0% |
| 19 | 2026-08-03 | RM3 | 46 | 0,2724 / 0,3661 | berhasil | Pembanding variansi, keluar kelas target 44,7% |

Kolom Macro F1 pada RM1 ditulis dua angka, yaitu gabungan lalu rata-rata antar korpus. Pada RM2 korpus ujinya tunggal sehingga hanya satu angka.
Pada RM3 ditulis mode 1 lalu mode 2.

## Rekapitulasi

| Skenario | Korpus uji | Rerata | SD | Rentang |
|----------|-----------|--------|-----|---------|
| RM1 | TESS | 0,9986 | 0,0013 | 0,9976-1,0000 |
| RM1 | RAVDESS | 0,4660 | 0,0275 | 0,4375-0,4981 |
| RM1 | SAVEE | 0,4123 | 0,0616 | 0,3490-0,4819 |
| RM1 | gabungan | 0,7925 | 0,0108 | 0,7754-0,8004 |
| RM2 fold_1 | SAVEE | 0,1303 | 0,0115 | 0,1191-0,1420 |
| RM2 fold_2 | TESS | 0,2329 | 0,0906 | 0,1543-0,3320 |
| RM2 fold_3 | RAVDESS | 0,1372 | 0,0154 | 0,1201-0,1498 |
| RM3 mode 1 | INESCO | 0,2879 | 0,0156 | 0,2724-0,3085 |
| RM3 mode 2 | INESCO | 0,3988 | 0,0292 | 0,3661-0,4313 |
| RM3 prediksi ke luar kelas target | INESCO | 47,7% | - | 44,3%-52,2% |

Sumber: `data/models/rm1/seed_summary.csv`, `data/models/rm2/rm2_summary.csv`, dan `data/models/rm3/rm3_summary.csv`.

## Fase Eksplorasi

Sembilan run pertama tahap Modeling dijalankan sebelum
`enable_op_determinism` diaktifkan. Berkasnya tidak dipertahankan, tetapi
temuannya diringkas pada bagian Catatan Fase Eksplorasi di
`docs/development_log/04_modeling.md`.

## Hubungan dengan development log

| Berkas | Isi |
|--------|-----|
| `docs/development_log/04_modeling.md` | Keputusan desain dan status checkpoint. Berubah jarang. |
| `docs/experiments/experiment_NN.md` | Hasil satu kali eksekusi. Bertambah terus. |

Angka final yang masuk Bab 4 diambil dari berkas metrik, dirujuk lewat
experiment log, dan diringkas pada development log. Jangan menyalin angka
antar dokumen secara manual.