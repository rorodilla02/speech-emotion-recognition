from __future__ import annotations

import io

import librosa

from ..preprocessing.common import AudioData


class AudioBytesLoader:
    """
    Memuat AudioData dari bytes di memori (unggahan atau rekaman
    pengguna di prototipe web), sebagai varian AudioLoader yang
    menerima sumber bytes, bukan path berkas di disk.

    Alasan
    ------
    AudioLoader (preprocessing/audio_loader.py) mensyaratkan Path di
    disk. Menulis audio pengguna ke berkas sementara demi memakainya
    akan melanggar KNF-05 (sistem tidak menyimpan berkas audio
    pengguna, diproses di memori). Kelas ini memakai pemanggilan
    librosa.load yang identik dengan AudioLoader (sr=None, mono=False),
    hanya sumbernya diganti io.BytesIO, sehingga konvensi bentuk dan
    sumbu kanal keluaran tetap sama persis dengan yang diharapkan
    MonoConverter pada tahap berikutnya di AudioPreprocessor.

    Catatan
    -------
    Kelas ini tidak:
    - membaca dari path/disk (lihat AudioLoader untuk itu)
    - melakukan validasi format atau durasi (tanggung jawab lapisan UI)
    - melakukan resampling, konversi mono, atau normalisasi apa pun

    Peringatan yang belum diverifikasi
    -----------------------------------
    Dukungan decode MP3 dari objek bytes (bukan path) bergantung pada
    backend yang dipakai librosa/soundfile di lingkungan WSL2 ini.
    libsndfile >= 1.1.0 mendukung MP3 langsung dari file-like object;
    versi lebih lama akan jatuh ke backend audioread, yang untuk MP3
    kerap mensyaratkan path berkas sungguhan, bukan BytesIO. WAJIB
    dites manual dengan berkas .mp3 asli sebelum Checkpoint 4.
    """

    def load(self, raw_bytes: bytes) -> AudioData:
        audio, sample_rate = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=False)

        return AudioData(audio=audio, sample_rate=sample_rate)