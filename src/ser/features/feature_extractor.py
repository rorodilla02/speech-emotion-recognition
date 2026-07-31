from __future__ import annotations
import numpy as np
import librosa
from ..preprocessing.common import AudioData
from ..preprocessing.duration_normalizer import DurationNormalizer
from ..preprocessing.constants import TARGET_SAMPLE_RATE
from .constants import(
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MFCC,
    N_MELS,
    N_CHROMA,
    PREEMPHASIS_COEF,
    DELTA_WIDTH,
    CHROMA_TUNING,
    N_FRAMES,
    FEATURE_SHAPE,
)

class FeatureExtractor:
    """
    Menghasilkan representasi feature fusion untuk satu berkas audio.

    Pipeline
    --------
    1. Center crop bila audio melebihi panjang target
    2. Pre-emphasis (khusus cabang MFCC)
    3. MFCC
    4. Delta dan Delta-Delta dari MFCC
    5. Chroma dari sinyal tanpa pre-emphasis
    6. CMVN per file pada MFCC, Delta, dan Delta-Delta
    7. Penggabungan keempat blok pada sumbu frekuensi
    8. Penyeragaman sumbu waktu ke N_FRAMES

    Keluaran berbentuk (N_FEATURES, N_FRAMES) = (51, 401).

    Catatan
    -------
    Kelas ini tidak:
    - membaca atau menulis file
    - melakukan resampling atau normalisasi amplitudo
    - melakukan augmentasi
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.duration_normalizer = DurationNormalizer(sample_rate=sample_rate)

    def extract(self, audio_data: AudioData) -> np.ndarray:
        if audio_data.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate tidak sesuai: "
                f"{audio_data.sample_rate} != {self.sample_rate}"
            )

        audio_data = self.duration_normalizer.process(audio_data)
        audio = np.asarray(audio_data.audio, dtype=np.float32)

        if audio.size < N_FFT:
            raise ValueError(
                f"Audio terlalu pendek untuk diekstraksi: "
                f"{audio_data} sampel (< {N_FFT})."
            )

        mfcc = self._mfcc(audio)
        delta = librosa.feature.delta(mfcc, width=DELTA_WIDTH)
        delta2 = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH)
        chroma = self._chroma(audio)

        self._assert_aligned(mfcc, delta, delta2, chroma)

        fused = np.vstack(
            [
                self._cmvn(mfcc),
                self._cmvn(delta),
                self._cmvn(delta2),
                chroma.astype(np.float32),
            ]
        )

        features = self._fit_time_axis(fused)

        if features.shape != FEATURE_SHAPE:
            raise RuntimeError(
                f"Bentuk fitur tidak sesuai: "
                f"{features.shape} != {FEATURE_SHAPE}"
            )

        return features

    def real_frames(self, n_samples: int) -> int:
        length = min(n_samples, self.duration_normalizer.target_length)

        return min(1 + length // HOP_LENGTH, N_FRAMES)

    def _mfcc(self, audio: np.ndarray) -> np.ndarray:
        emphasized = librosa.effects.preemphasis(
            audio, coef=PREEMPHASIS_COEF
        )

        return librosa.feature.mfcc(
            y=emphasized,
            sr=self.sample_rate,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    def _chroma(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_chroma=N_CHROMA,
            tuning=CHROMA_TUNING,
        )

    @staticmethod
    def _cmvn(block: np.ndarray) -> np.ndarray:
        mean = block.mean(axis=1, keepdims=True)
        std = block.std(axis=1, keepdims=True)

        return ((block - mean) / np.maximum(std, 1e-8)).astype(np.float32)

    @staticmethod
    def _fit_time_axis(features: np.ndarray) -> np.ndarray:
        n_frames = features.shape[1]

        if n_frames > N_FRAMES:
            start = (n_frames - N_FRAMES) // 2

            return features[:, start:start + N_FRAMES]

        if n_frames < N_FRAMES:
            total = N_FRAMES - n_frames
            left = total // 2
            right = total - left

            return np.pad(
                features,
                ((0, 0), (left, right)),
                mode="constant",
                constant_values=0.0,
            )

        return features

    @staticmethod
    def _assert_aligned(*blocks: np.ndarray):
        frame_counts = {block.shape[1] for block in blocks}

        if len(frame_counts) != 1:
            raise RuntimeError(
                f"Jumlah frame antar blok fitur tidak sama: {frame_counts}"
            )