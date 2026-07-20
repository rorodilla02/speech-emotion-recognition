from __future__ import annotations
from .common import AudioData, BasePreprocessor
from .constants import TARGET_RMS
import numpy as np

class RMSNormalizer(BasePreprocessor):
    def process(self, audio_data: AudioData) -> AudioData:
        current_rms = np.sqrt(np.mean(audio_data.audio**2))
        if current_rms==0:
            return audio_data
        
        scale = TARGET_RMS/current_rms
        normalized_audio = audio_data.audio*scale
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

        return AudioData(audio=normalized_audio, sample_rate=audio_data.sample_rate)