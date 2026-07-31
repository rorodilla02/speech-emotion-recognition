from __future__ import annotations
import numpy as np
from .common import AudioData, BasePreprocessor
from .constants import TARGET_DURATION, TARGET_SAMPLE_RATE


class DurationNormalizer(BasePreprocessor):
    """
    Membatasi panjang audio pada satu panjang maksimum.

    Strategi
    --------
    - Audio lebih panjang dari target : center crop
    - Audio lebih pendek dari target  : dikembalikan apa adanya

    Audio yang lebih pendek sengaja TIDAK dipadding pada domain
    gelombang. Zero padding pada gelombang menghasilkan frame
    keheningan digital yang nilainya jatuh ke lantai skala desibel,
    sehingga merusak statistik normalisasi fitur dan menjadikan
    proporsi padding sebagai penanda identitas korpus.
    Penyeragaman panjang dilakukan pada domain fitur oleh
    FeatureExtractor (lihat subbab 3.4.5).

    Catatan
    -------
    Kelas ini tidak:
    - membaca atau menulis file
    - melakukan resampling
    - mengubah amplitudo sinyal
    - melakukan padding
    """

    def __init__(
        self,
        target_duration: float = TARGET_DURATION,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        if target_duration <= 0:
            raise ValueError("target_duration harus lebih besar dari 0.")

        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_length = int(round(target_duration * sample_rate))

    def process(self, audio_data: AudioData) -> AudioData:
        if audio_data.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate tidak sesuai: "
                f"{audio_data.sample_rate} != {self.sample_rate}"
            )

        audio = np.asarray(audio_data.audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError(
                "DurationNormalizer hanya menerima audio mono satu dimensi."
            )

        if audio.shape[0] > self.target_length:
            audio = self._center_crop(audio)

        return AudioData(
            audio=audio,
            sample_rate=self.sample_rate,
        )

    def _center_crop(self, audio: np.ndarray) -> np.ndarray:
        start = (audio.shape[0] - self.target_length) // 2

        return audio[start:start + self.target_length]