# Modeling

Tahap ini mencakup perancangan arsitektur CNN, konfigurasi training, serta
pelatihan dan evaluasi model untuk skenario RM1, RM2, dan RM3.

Seluruh input model bersumber dari `data/features/features.npy` melalui
`FeatureDataset`, bukan dibaca langsung, agar aturan pemisahan data
augmentasi tetap ditegakkan.

Spesifikasi input yang sudah terkunci dari tahap Data Preparation:

| Properti | Nilai |
|----------|-------|
| Bentuk fitur per sampel | (51, 401) |
| Bentuk input CNN | (51, 401, 1) |
| Tipe data | float32 |
| Jumlah kelas | 7 |
| Urutan label | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| Total baris fitur | 11454 (6926 processed + 4528 augmented) |

## Konvensi Lokasi Berkas

Draf awal log ini menyebut `src/ser/modeling/`, `artifacts/models/`, dan
`artifacts/metrics/`. Implementasi diseragamkan ke konvensi yang sudah
dipakai seluruh tahap sebelumnya:

| Jenis | Lokasi |
|-------|--------|
| Kode | `src/ser/models/` |
| Skrip eksekusi | `scripts/11_*.py` dan seterusnya |
| Artefak model dan metrik | `data/models/<skenario>/` |
| Gambar untuk Bab 4 | `reports/figures/` (checkpoint 6) |

---

# Checkpoint 1 — Desain Arsitektur CNN

Status:

✅ Completed

## Tujuan

Merancang arsitektur CNN yang sesuai dengan bentuk input (51, 401, 1),
dapat dilatih pada batasan VRAM 4 GB, dan setiap komponennya dapat
dijustifikasi terhadap uraian teori pada subbab 2.4.

## Pertimbangan Desain

Sumbu input bersifat asimetris. Sumbu frekuensi hanya 51, sedangkan sumbu
waktu 401. Selain itu sumbu frekuensi bukan sumbu yang homogen, melainkan
gabungan empat blok fitur:

| Indeks baris | Blok | Jumlah |
|--------------|------|--------|
| 0-12 | MFCC | 13 |
| 13-25 | Delta | 13 |
| 26-38 | Delta-Delta | 13 |
| 39-50 | Chroma | 12 |

Keputusan yang diambil: **kernel dibiarkan simetris 3x3, asimetri ditangani
lewat ukuran pooling.** Alasannya, ketidakseimbangan yang perlu dikoreksi
ada pada panjang sumbu (51 berbanding 401), bukan pada skala pola lokal yang
ingin ditangkap. Pooling dibuat 2x2 pada dua blok awal lalu 2x4 pada dua
blok akhir, sehingga sumbu waktu direduksi 64 kali sementara sumbu frekuensi
hanya 16 kali.

Konsekuensi kernel 3x3 pada sumbu frekuensi: kernel akan melintasi batas
antar blok fitur pada baris 12-14, 25-27, dan 38-40, sehingga koefisien dari
dua blok berbeda tercampur pada tiga posisi. Kondisi ini diterima karena
hanya menyangkut 3 dari 51 posisi, dan pencampuran justru sejalan dengan
premis subbab 2.3.5 bahwa fusi menghasilkan representasi gabungan.

## Keputusan Arsitektur

| # | Layer | Konfigurasi | Output Shape | Justifikasi |
|---|-------|-------------|--------------|-------------|
| 0 | Input | - | (51, 401, 1) | Feature fusion sebagai citra satu kanal (subbab 2.4.2) |
| 1 | Conv2D | 32 filter, 3x3, same, tanpa bias | (51, 401, 32) | Ekstraksi pola lokal spektral-temporal (subbab 2.4.2) |
| 2 | BatchNormalization | - | (51, 401, 32) | Stabilisasi distribusi aktivasi antar batch |
| 3 | Activation | ReLU | (51, 401, 32) | Non-linearitas (subbab 2.4.2) |
| 4 | MaxPooling2D | 2x2 | (25, 200, 32) | Reduksi dimensi (subbab 2.4.2) |
| 5 | Conv2D + BN + ReLU | 64 filter, 3x3, same | (25, 200, 64) | Jumlah filter dinaikkan seiring dimensi menyusut |
| 6 | MaxPooling2D | 2x2 | (12, 100, 64) | |
| 7 | Conv2D + BN + ReLU | 128 filter, 3x3, same | (12, 100, 128) | Pola tingkat lebih abstrak |
| 8 | MaxPooling2D + Dropout | 2x4, rate 0.3 | (6, 25, 128) | Pooling waktu diperbesar; regularisasi |
| 9 | Conv2D + BN + ReLU | 128 filter, 3x3, same | (6, 25, 128) | Filter tidak dinaikkan lagi (risiko R-01) |
| 10 | MaxPooling2D + Dropout | 2x4, rate 0.3 | (3, 6, 128) | |
| 11 | Flatten | - | (2304,) | Jembatan menuju klasifikasi (subbab 2.4.2) |
| 12 | Dense + BN + ReLU + Dropout | 128 unit, rate 0.5 | (128,) | Kombinasi fitur hasil ekstraksi (subbab 2.4.2) |
| 13 | Dense | 7 unit, softmax | (7,) | Tujuh kelas emosi beririsan (Bab 1, Ruang Lingkup) |

Total parameter: **537.639** (trainable 536.679, non-trainable 960)

Ukuran bobot: 2,05 MB

Bias pada Conv2D dan Dense dimatikan karena parameter beta pada
BatchNormalization sudah berperan sebagai bias, sehingga bias terpisah
bersifat redundan.

## Reseptif Field

Setelah empat kali pooling, satu sel pada peta fitur terakhir mencakup:

- sumbu waktu: sekitar 67 frame, setara **670 ms**, sesuai skala penanda
  prosodi suprasegmental yang jadi dasar teoretis subbab 2.2.2
- sumbu frekuensi: sekitar 17 baris, satu orde dengan lebar satu blok fitur
  (12-13 baris)

## Kebutuhan Memori

| Ukuran batch | Estimasi kebutuhan |
|--------------|--------------------|
| 16 | 571 MB |
| 32 | 1.143 MB |
| 64 | 2.286 MB |
| 128 | 4.571 MB |

Aktivasi per sampel: 3.744.778 float (14,29 MB). Estimasi memakai faktor
2,5 kali untuk gradien dan state optimizer.

Angka terukur pada eksekusi nyata dilaporkan pada checkpoint 2.

## Penyesuaian yang Diperlukan pada Bab 2

Subbab 2.4.2 saat ini hanya mendefinisikan Convolution, ReLU, Pooling,
Flatten, Dense, dan Softmax. Dua komponen yang dipakai arsitektur ini belum
punya dasar teori di naskah dan perlu ditambahkan:

- Batch Normalization (Ioffe & Szegedy, 2015)
- Dropout (Srivastava dkk., 2014)

Penambahan dikerjakan setelah seluruh tahap Modeling selesai, agar isi Bab 2
mencerminkan konfigurasi final, bukan konfigurasi sementara.

## Validasi

- [x] `model.summary()` menampilkan input shape (51, 401, 1)
- [x] Output layer berjumlah 7 unit dengan aktivasi softmax
- [x] Satu forward pass dengan batch dummy berhasil tanpa OOM
- [x] Setiap jenis layer dapat dirujuk ke subbab 2.4
- [x] `MODEL_INPUT_SHAPE` diturunkan dari `FEATURE_SHAPE`, bukan ditulis ulang
- [x] Bentuk fitur di disk diverifikasi sama dengan bentuk input model

## Output

- `src/ser/models/constants.py`
- `src/ser/models/cnn_architecture.py`
- `scripts/11_model_architecture.py`
- `data/models/architecture_summary.txt`
- `data/models/architecture_layers.csv`

---

# Checkpoint 2 — Konfigurasi Training

Status:

✅ Completed

## Tujuan

Menetapkan konfigurasi training yang dipakai seragam pada seluruh skenario,
sehingga perbandingan antar RM1, RM2, dan RM3 bersifat setara.

## Konfigurasi

| Parameter | Nilai | Justifikasi |
|-----------|-------|-------------|
| Loss function | Categorical Cross-Entropy | Berpasangan dengan aktivasi Softmax pada layer output (subbab 2.4.2). Label di-one-hot agar metrik macro F1-score dapat dihitung selama training |
| Optimizer | Adam | Nilai baku yang stabil untuk arsitektur ber-BatchNormalization (Kingma & Ba, 2015) |
| Learning rate | 0.001 | Nilai baku Adam; penyetelan manual tidak diperlukan karena BatchNormalization sudah menstabilkan skala gradien |
| Batch size | 32 | Estimasi 1.143 MB, terukur 1.537 MB, aman terhadap VRAM efektif |
| Epoch maksimum | 100 | Batas atas; penghentian sebenarnya ditentukan early stopping |
| Metrik utama | Macro F1-score | Mitigasi risiko R-02 |
| Early stopping | monitor `val_macro_f1`, mode max, patience 12, restore best weights | Seleksi model memakai metrik utama, bukan `val_loss`, agar konsisten dengan metrik yang dilaporkan |
| ReduceLROnPlateau | faktor 0.5, patience 5, lr minimum 1e-5 | Membantu konvergensi tahap akhir tanpa menambah hyperparameter yang perlu disetel |
| Model checkpoint | simpan bobot terbaik menurut `val_macro_f1` | |
| Random seed | 42 | KNF-06 |

Konfigurasi tersimpan sebagai `data/models/training_config.json` dan dipakai
identik oleh RM1, ketiga fold RM2, serta evaluasi RM3.

## Verifikasi Perangkat

GPU semula tidak terdeteksi. `nvidia-smi` berjalan normal, sehingga masalah
bukan pada driver melainkan pada jalur pencarian pustaka: pustaka CUDA
terpasang sebagai paket pip di `site-packages/nvidia/*/lib` dan tidak masuk
`LD_LIBRARY_PATH` secara baku.

Penanganan:

- pemasangan `tensorflow[and-cuda]==2.21.0`
- penambahan `/usr/lib/wsl/lib` dan direktori `site-packages/nvidia/*/lib`
  ke `LD_LIBRARY_PATH` melalui skrip aktivasi virtual environment

Catatan penting untuk Bab 3: VRAM 4 GB pada Tabel 3.2 adalah kapasitas
fisik. Pada WSL2, GPU dibagi dengan sistem host Windows. Pengamatan
`nvidia-smi` menunjukkan pemakaian oleh desktop sebesar 1.048 MB saat
peramban terbuka dan sekitar 500 MB saat hanya editor yang berjalan,
sehingga kapasitas efektif bagi proses training berkisar 3,0 sampai 3,5 GB.
Ambang peringatan pada skrip ditetapkan 3.000 MB.

## Hasil Uji Konfigurasi

| Aspek | Hasil |
|-------|-------|
| Perangkat aktif | GPU |
| Bentuk x_train | (512, 51, 401, 1) |
| Bentuk y_train | (512, 7) |
| Metrik tercatat | accuracy, loss, macro_f1, val_accuracy, val_loss, val_macro_f1, learning_rate |
| Puncak pemakaian VRAM | 1.537 MB |
| Artefak callback | `best_model.keras` dan `training_log.csv` terbentuk |

Waktu per step yang dilaporkan smoke test (0,81 detik) **tidak dipakai**
sebagai dasar perencanaan. Uji hanya menjalankan 48 step, sehingga waktu
graph tracing dan autotuning kernel pada epoch pertama tidak sempat
teramortisasi dan ikut terbagi rata. Pengukuran waktu yang sah dilakukan
pada checkpoint 3 dengan mengecualikan epoch pertama.

## Catatan Distribusi Kelas

Distribusi kelas pada data latih gabungan relatif seimbang dengan rasio
antar kelas terbesar dan terkecil sekitar 1.06 (Tabel 3.12), sehingga
penerapan class weight tidak diperlukan pada RM1.

Kondisi ini tidak berlaku untuk seluruh fold RM2. SAVEE memiliki
ketimpangan internal pada kelas netral yang jumlahnya dua kali lipat kelas
lain, sehingga saat SAVEE berperan sebagai korpus uji, accuracy saja bisa
memberi gambaran yang menyesatkan dan macro F1-score menjadi metrik utama.

`class_weight` tidak dipakai pada skenario mana pun, karena mitigasi
risiko R-02 pada Bab 3 menetapkan stratifikasi, macro F1-score, dan
pelaporan terpisah per korpus, bukan pembobotan kelas.

## Keputusan atas Validation Kosong pada RM2 dan RM3

Opsi yang dipilih: **memisahkan sebagian data latih tiap fold sebagai
validation internal.**

Alasan penolakan opsi lain:

- Memakai data uji sebagai validasi menimbulkan kebocoran seleksi model.
  Korpus target ikut menentukan kapan training berhenti, sehingga seluruh
  klaim RM2 dan RM3 kehilangan dasar.
- Memakai jumlah epoch tetap hasil RM1 tidak dapat dipertanggungjawabkan,
  sebab ukuran data latih antar fold berbeda jauh (1.728 sampai 4.048 file
  menurut Tabel 3.18), sehingga titik konvergensinya tidak mungkin sama.

Aturan pemisahan mengikuti kebijakan RM1 pada subbab 3.4.2, yaitu berbasis
speaker untuk RAVDESS dan SAVEE, serta stratified berdasarkan label untuk
TESS yang hanya punya dua speaker. Tidak ada aturan baru yang perlu
dijustifikasi terpisah.

`build_callbacks` sengaja selalu memantau metrik validasi dan akan
menghentikan proses bila data validasi tidak tersedia, sehingga tidak
mungkin ada skenario yang berjalan tanpa validasi tanpa disadari.

Implementasi `ValidationSplitter` dikerjakan pada checkpoint 4.

## Validasi

- [x] Konfigurasi tersimpan pada satu file dan dipakai seluruh skenario
- [x] Batch size terverifikasi tidak menimbulkan OOM pada VRAM 4 GB
- [x] Seed diterapkan pada Python, NumPy, dan TensorFlow
- [x] GPU terdeteksi dan dipakai
- [x] Alokasi memori bertahap aktif agar TensorFlow tidak mengambil seluruh VRAM

## Output

- `src/ser/models/training_config.py`
- `src/ser/models/data_adapter.py`
- `scripts/12_training_config.py`
- `data/models/training_config.json`

---

# Checkpoint 3 — Training dan Evaluasi RM1

Status:

🔄 Sedang dikerjakan

## Skenario

Within-corpus. Model dilatih pada gabungan RAVDESS, TESS, dan SAVEE dengan
split speaker-independent, lalu diuji pada test set yang tidak pernah
dilihat model.

Model hasil checkpoint ini juga dipakai pada RM3 tanpa pelatihan ulang.

## Temuan Awal: Validation RM1 Tidak Memuat SAVEE

Pemeriksaan `SplitGenerator` menunjukkan SAVEE memakai pembagian 3 speaker
latih dan 1 speaker uji, tanpa speaker untuk validasi. Akibatnya data
validasi RM1 hanya berisi RAVDESS dan TESS.

Konsekuensi: early stopping dan pemilihan bobot terbaik dilakukan
berdasarkan metrik yang buta terhadap SAVEE, satu-satunya korpus beraksen
British pada penelitian ini. Bila macro F1 SAVEE pada data uji jauh lebih
rendah dari korpus lain, sebagian penyebabnya adalah seleksi model, bukan
semata kesulitan korpus.

Penanganan: split tidak diubah, karena SAVEE hanya punya 4 speaker sehingga
menyisakan satu untuk validasi berarti mengurangi data latih secara
signifikan. Sebagai gantinya, metrik dilaporkan terpisah per korpus pada
data uji, sesuai mitigasi risiko R-02, dan keterbatasan ini dicatat pada
subbab 3.5 serta Bab 5.

## Rencana Jumlah Seed

Menjalankan beberapa seed **tidak** bertujuan memperoleh model yang lebih
baik, melainkan mengukur besar fluktuasi hasil yang murni disebabkan
inisialisasi acak. Angka tersebut diperlukan untuk menilai apakah selisih
macro F1 antara RM1 dan RM2 melampaui variasi antar-run.

Dua aturan yang ditetapkan sebelum eksekusi:

1. Model of record adalah hasil seed 42. Seed inilah yang dipakai untuk
   RM3 dan prototipe Streamlit.
2. Pemilihan seed terbaik berdasarkan macro F1 pada data uji tidak
   diperbolehkan, karena merupakan seleksi model memakai data uji.

Jumlah seed ditentukan setelah durasi satu kali training terukur:

| Durasi satu kali training | Rencana |
|---------------------------|---------|
| kurang dari 30 menit | 3 seed untuk RM1 dan ketiga fold RM2 |
| 30 sampai 60 menit | 3 seed untuk RM1 saja |
| lebih dari 60 menit | 1 seed, ketiadaan estimasi variansi dicatat sebagai keterbatasan |

## Komposisi Data

_(diisi setelah checkpoint dijalankan)_

| Subset | Jumlah asli | Jumlah setelah augmentasi |
|--------|-------------|---------------------------|
| Train | | |
| Validation | | (tanpa augmentasi) |
| Test | | (tanpa augmentasi) |

## Hasil

Model of record: seed 42. Lima seed dijalankan untuk mengestimasi variansi
antar-run, bukan untuk memilih model terbaik.

| Metrik | Seed 42 | Rerata 5 seed | SD | Rentang |
|--------|---------|---------------|-----|---------|
| Macro F1 uji, gabungan | 0,7886 | 0,7974 | 0,0133 | 0,0298 |
| Macro F1 uji, rata-rata antar korpus | 0,6091 | 0,6323 | 0,0245 | 0,0571 |
| Accuracy uji, gabungan | 0,7891 | | | |
| Macro F1 validasi | 0,8408 | 0,8404 | 0,0021 | 0,0042 |
| Epoch terbaik | 52 | 39 | | 27-52 |
| Waktu per epoch | 9,0 detik | ±9,5 detik | | |
| Puncak VRAM | 1,01 GB | | | |
| Baseline mayoritas kelas | ±14,40% | | | |
| Chance level | 14,29% | | | |

### Per Korpus

| Korpus | Seed 42 | Rerata 5 seed | SD | Rentang |
|--------|---------|---------------|-----|---------|
| TESS | 0,9976 | 0,9986 | 0,0013 | 0,9976–1,0000 |
| RAVDESS | 0,4727 | 0,4660 | 0,0275 | 0,4375–0,4981 |
| SAVEE | 0,4720 | 0,4123 | 0,0616 | 0,3490–0,4819 |
| Rata-rata antar korpus | 0,6474 | 0,6256 | 0,0207 | 0,5955-0,6474 |
| Gabungan | 0,8004 | 0,7925 | 0,0108 | 0,7754–0,8004 |

Sumber angka: `data/models/rm1/seed_summary.csv`.

## Temuan

**Angka gabungan tidak mewakili kemampuan model.** Komposisi data uji timpang,
TESS 60,4 persen dari 697 sampel. Karena TESS praktis sempurna, angka
gabungan 0,7974 lebih menggambarkan proporsi korpus daripada kemampuan
model. Performa pada kondisi speaker-independent berada di kisaran 0,41
sampai 0,49.

**TESS tidak independen.** Irisan nama berkas train-test nol, sehingga tidak
ada kebocoran berkas. Namun 270 dari 271 pasangan speaker dan kata pembawa
pada data uji juga terdapat pada data latih, yaitu 99,6 persen. TESS hanya
memiliki dua speaker sehingga pemisahan berbasis speaker tidak mungkin
dilakukan. Satu kesalahan pada seed 46 menunjukkan model tetap melakukan
klasifikasi, hanya pada tugas yang terlalu mudah.

**Metrik validasi lemah membedakan kemampuan lintas-speaker.** Rentang macro
F1 validasi antar-seed hanya 0,0042, sementara rentang macro F1 SAVEE pada
data uji mencapai 0,1063, yaitu dua puluh lima kali lipat. Data validasi
tidak memuat SAVEE, dan bagian TESS di dalamnya bernilai konstan sehingga
tidak ikut membedakan. Rentang epoch terbaik 27 sampai 52 pada tingkat skor
validasi yang setara menegaskan kurva validasi sudah datar sejak epoch
pertengahan.

**ReduceLROnPlateau menstabilkan tahap akhir.** Pada EXP-01 yang berjalan
tanpa penurunan learning rate, fluktuasi val_macro_f1 pada 15 epoch terakhir
mencapai kurang lebih 0,036. Pada EXP-02 dengan callback aktif, fluktuasi
menyempit menjadi kurang lebih 0,011 dengan puncak yang praktis sama.
Konfigurasi dipertahankan.

## Acuan Penilaian RM2

Ditetapkan sebelum RM2 dijalankan. Perbandingan dilakukan terhadap baseline
RM1 pada korpus yang sama, bukan terhadap angka gabungan.

| Fold RM2 | Korpus uji | Baseline RM1 | Rentang RM1 | Sifat baseline |
|----------|-----------|--------------|-------------|----------------|
| 1 | SAVEE | 0,4087 | 0,3517-0,4580 | speaker-independent, sebanding |
| 2 | TESS | 0,9995 | 0,9976-1,0000 | speaker-dependent, tidak sebanding |
| 3 | RAVDESS | 0,4888 | 0,4581-0,5405 | speaker-independent, sebanding |

Penurunan hanya diklaim bermakna bila rentang antar-seed RM2 pada suatu fold
tidak bertumpang tindih dengan rentang RM1 pada korpus yang sama. Uji
signifikansi statistik tidak digunakan, karena seed bukan sampel acak dari
suatu populasi dan jumlahnya terlalu kecil.

## Validasi

- [x] Data augmentasi tidak muncul pada validation dan test
- [x] Tidak ada speaker yang muncul di train sekaligus test (selain TESS)
- [x] Training berhenti lewat early stopping pada seluruh seed
- [x] Learning curve tidak menunjukkan overfitting ekstrem
- [x] Hasil melampaui baseline mayoritas kelas
- [x] Variansi antar-run terestimasi dari lima seed

## Output

- `data/models/rm1/best_model.keras`
- `data/models/rm1/training_log.csv`
- `data/models/rm1/predictions.csv`
- `data/models/rm1/metrics_summary.csv`
- `data/models/rm1/metrics_per_class.csv`

---

# Checkpoint 4 — Training dan Evaluasi RM2

Status:

⬜ Belum dikerjakan

## Skenario

Cross-corpus dengan skema Leave-One-Corpus-Out. Model dilatih ulang
sebanyak tiga kali, satu untuk setiap fold, sehingga model yang dievaluasi
pada RM2 bukan model yang sama dengan RM1.

| Fold | Data latih | Data uji |
|------|-----------|----------|
| 1 | RAVDESS + TESS | SAVEE |
| 2 | RAVDESS + SAVEE | TESS |
| 3 | TESS + SAVEE | RAVDESS |

## Pekerjaan Tambahan

Implementasi `ValidationSplitter` sesuai keputusan checkpoint 2, yaitu
pemisahan validation internal dari data latih tiap fold secara
speaker-independent untuk RAVDESS dan SAVEE, serta stratified untuk TESS.

## Hasil

_(diisi setelah checkpoint dijalankan)_

| Fold | Accuracy | Macro F1 | Selisih Macro F1 terhadap RM1 |
|------|----------|----------|-------------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| Rata-rata | | | |

## Catatan Penafsiran

Penurunan performa pada RM2 tidak dapat langsung diatribusikan sepenuhnya
pada perbedaan korpus. Terdapat confound yang sudah didokumentasikan pada
Tabel 3.13, yaitu perbedaan gender, aksen, jumlah speaker, serta perbedaan
sistematis proporsi padding antar korpus.

Perbedaan ukuran data latih antar fold (1.728 sampai 4.048 file) merupakan
confound tambahan yang sudah dicatat pada subbab 3.4.

## Validasi

- [ ] Korpus uji tiap fold tidak muncul sama sekali pada data latihnya
- [ ] Ketiga fold memakai konfigurasi training yang identik
- [ ] Validation internal tidak mengandung speaker yang sama dengan data latih
- [ ] Macro F1 dilaporkan sebagai metrik utama, bukan accuracy

## Output

- `data/models/rm2/fold_1/`, `fold_2/`, `fold_3/`

---

# Checkpoint 5 — Evaluasi RM3

Status:

⬜ Belum dikerjakan

## Skenario

Cross-lingual. Model hasil RM1 diuji pada INESCO tanpa pelatihan ulang,
karena yang diuji adalah kemampuan model menghadapi bahasa yang belum
pernah dilihatnya (Opsi A, lihat Known Issues).

Checkpoint ini tidak melakukan training sama sekali.
`data/splits/rm3/train.csv` sengaja tidak dipakai.

## Dua Mode Pelaporan

Sesuai draf subbab 3.2 (Bab 3, masih dalam penulisan), evaluasi RM3
dilaporkan dalam dua mode karena jumlah kelas korpus uji berbeda dari
korpus latih.

| Mode | Perlakuan output | Status pelaporan |
|------|-----------------|------------------|
| 1 | Tanpa pembatasan, model tetap dapat memprediksi tujuh kelas. Prediksi ke kelas di luar tiga kelas target dihitung sebagai kesalahan | Hasil utama |
| 2 | Output softmax dibatasi pada tiga kelas target sebelum pemilihan kelas dengan probabilitas tertinggi | Analisis pelengkap |

Selisih antara kedua mode bersifat informatif karena menunjukkan seberapa
sering model mengalihkan prediksinya ke kelas emosi yang sebenarnya tidak
tersedia pada korpus uji.

Kedua mode dihitung dari berkas `predictions.csv` yang sama, karena berkas
tersebut menyimpan probabilitas seluruh tujuh kelas per berkas audio.

## Hasil

_(diisi setelah checkpoint dijalankan)_

| Mode | Accuracy | Macro F1 |
|------|----------|----------|
| 1 (tujuh kelas) | | |
| 2 (tiga kelas) | | |

Chance level tiga kelas: 33.33%

## Catatan Penafsiran

Nilai accuracy RM3 yang lebih tinggi daripada RM1 belum tentu menandakan
generalisasi lintas-bahasa yang baik, sebab tingkat kesulitan kedua tugas
memang berbeda. Perbandingan yang sah hanya dilakukan terhadap baseline
masing-masing skema.

Selain itu, sebagian klip INESCO terpotong oleh center crop pada target
durasi 4 detik. Proporsinya dilaporkan sebagai keterbatasan.

## Validasi

- [ ] Model yang dipakai benar-benar model hasil RM1, bukan model baru
- [ ] INESCO tidak pernah dipakai pada training maupun validation
- [ ] Kedua mode dilaporkan, bukan hanya salah satu

## Output

- `data/models/rm3/`

---

# Checkpoint 6 — Analisis Hasil

Status:

⬜ Belum dikerjakan

## Tujuan

Menyusun seluruh keluaran ketiga skenario menjadi tabel dan gambar yang
siap dipakai pada Bab 4.

## Artefak yang Dihasilkan

| Artefak | Cakupan | Keperluan Bab 4 |
|---------|---------|-----------------|
| Confusion matrix | RM1, RM2 (3 fold), RM3 (2 mode) | |
| Learning curve | RM1, RM2 (3 fold) | |
| Tabel metrik gabungan | Seluruh skenario | |
| Tabel perbandingan RM1 vs RM2 vs RM3 | Selisih macro F1 | |
| Analisis proporsi padding per korpus | RM2 | Menguji confound padding memakai kolom `real_frames` |
| Diagram arsitektur CNN | Checkpoint 1 | Gambar untuk subbab 3.5 |

Diagram arsitektur dibuat sebagai SVG orisinal, mengikuti pola diagram Bab 2,
bukan memakai keluaran `keras.utils.plot_model`. Sumber angkanya adalah
`data/models/architecture_layers.csv`.

## Validasi

- [ ] Seluruh angka pada tabel bersumber dari file metrik, bukan disalin manual
- [ ] Urutan label pada confusion matrix konsisten dengan EMOTION_LABELS
- [ ] Seluruh gambar tersimpan pada resolusi 300 dpi

## Output

- `reports/figures/`
- `data/models/summary_all_scenarios.csv`

---

# 🏗 Keputusan Arsitektur

- Bentuk input model diturunkan dari `FEATURE_SHAPE`, bukan ditulis ulang,
  agar perubahan parameter ekstraksi fitur otomatis terdeteksi.
- Bias pada Conv2D dan Dense dimatikan karena diikuti BatchNormalization.
- Kernel dibiarkan simetris 3x3; asimetri sumbu ditangani lewat ukuran
  pooling, bukan lewat bentuk kernel.
- Jumlah filter berhenti pada 128 sebagai konsekuensi batas VRAM.
- Flatten dipertahankan sebagai jembatan menuju lapisan klasifikasi, sesuai
  uraian subbab 2.4.2, bukan diganti global pooling.
- Label di-one-hot agar macro F1-score dapat dipakai sebagai metrik seleksi
  model selama training, bukan hanya dihitung setelah training.
- Konfigurasi training identik untuk seluruh skenario, sehingga perbedaan
  hasil murni berasal dari komposisi data.
- Seluruh callback memantau metrik validasi; skenario tanpa data validasi
  ditolak secara eksplisit oleh kode.
- Evaluasi menyimpan prediksi per berkas audio, bukan hanya metrik agregat,
  sehingga seluruh analisis Bab 4 dapat dihitung ulang tanpa melatih ulang.
- Model of record adalah hasil seed 42; seed lain hanya untuk pelaporan
  variansi.
- `class_weight` tidak dipakai, mengikuti mitigasi risiko R-02.

---

# 📌 Progress Saat Ini

## Checkpoint 1 — Arsitektur CNN

- [x] Rancangan layer
- [x] Justifikasi terhadap Bab 2
- [x] Verifikasi input shape
- [x] Verifikasi batasan VRAM

## Checkpoint 2 — Konfigurasi Training

- [x] Loss function
- [x] Optimizer dan learning rate
- [x] Batch size
- [x] Callback
- [x] Penanganan validation kosong pada RM2
- [x] Random seed
- [x] Verifikasi GPU

## Checkpoint 3 — RM1

- [ ] Training
- [ ] Evaluasi
- [ ] Validasi anti-kebocoran
- [ ] Penentuan jumlah seed

## Checkpoint 4 — RM2

- [ ] ValidationSplitter
- [ ] Fold 1
- [ ] Fold 2
- [ ] Fold 3
- [ ] Rekapitulasi

## Checkpoint 5 — RM3

- [ ] Mode 1
- [ ] Mode 2

## Checkpoint 6 — Analisis

- [ ] Confusion matrix
- [ ] Learning curve
- [ ] Tabel metrik Bab 4
- [ ] Diagram arsitektur

---

# Known Issues

- Data validation RM1 tidak memuat satu pun berkas SAVEE, karena SAVEE
  hanya punya 4 speaker dan seluruhnya terpakai untuk latih dan uji.
  Seleksi model karenanya buta terhadap korpus beraksen British.
- RM2 dan RM3 memiliki `validation.csv` kosong sesuai rancangan split.
  Ditangani lewat validation internal dari data latih (keputusan
  checkpoint 2), bukan lewat penghapusan early stopping.
- `data/splits/rm3/train.csv` berisi seluruh korpus berbahasa Inggris,
  namun sesuai draf Tabel 3.1 (Bab 3, belum diajukan ke dospem) RM3
  memakai model hasil RM1 tanpa pelatihan ulang. Diputuskan mengikuti
  Opsi A: file tersebut tidak dipakai untuk training apa pun.
  `SplitGenerator` tetap menghasilkannya sebagai artefak, tapi perlu
  dicatat eksplisit di 04_modeling.md dan subbab 3.4.2 bahwa
  `rm3/train.csv` sengaja tidak dipakai, supaya tidak terlihat seperti
  file yang lupa dipakai.
- Validasi kebocoran data pada `FeatureValidator` hanya mencakup RM1.
- Pemeriksaan `Feature Shape` dan `Feature Dtype` tidak memvalidasi isi
  data, hanya dimensi larik.
- Wilayah zero padding pada sumbu waktu bernilai konstan 0.0 dan
  proporsinya berkorelasi dengan identitas korpus, sehingga berpotensi
  menjadi pintasan yang dipelajari model. Diuji secara kuantitatif pada
  checkpoint 6 memakai kolom `real_frames`.
- Subbab 2.4.2 belum memuat Batch Normalization, Dropout, categorical
  cross-entropy, dan Adam. Penambahan dikerjakan setelah Modeling selesai.
- Data uji TESS tidak independen secara akustik maupun leksikal. Irisan
  pasangan speaker dan kata pembawa antara train dan test mencapai 99,6
  persen, karena TESS hanya memiliki dua speaker.
- Metrik validasi RM1 memiliki daya pisah rendah terhadap kemampuan
  lintas-speaker, sehingga titik henti early stopping bervariasi 27 sampai
  52 epoch pada tingkat skor validasi yang setara.

---

# 🎯 Target Output

Tahap Modeling menghasilkan:

- arsitektur CNN
- konfigurasi training
- model RM1
- model RM2 (tiga fold)
- metrik RM1, RM2, RM3
- confusion matrix seluruh skenario
- learning curve
- tabel metrik gabungan untuk Bab 4

---

# 🚀 Next Session

1. Implementasi prototipe Streamlit (RM4)
2. Pengujian fungsional terhadap Tabel 3.5
3. Pengukuran waktu respons sistem