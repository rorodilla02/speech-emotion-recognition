from __future__ import annotations
from .common import AudioData, BasePreprocessor
from .constants import TARGET_RMS
import numpy as np


class RMSNormalizer(BasePreprocessor):
    def process(self, audio_data: AudioData) -> AudioData:
        audio = np.asarray(audio_data.audio, dtype=np.float32)

        current_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        if current_rms == 0.0:
            return AudioData(
                audio=audio,
                sample_rate=audio_data.sample_rate,
            )

        scale = TARGET_RMS / current_rms
        normalized_audio = np.clip(audio * scale, -1.0, 1.0)

        return AudioData(
            audio=normalized_audio.astype(np.float32),
            sample_rate=audio_data.sample_rate,
        )