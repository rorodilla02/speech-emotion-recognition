from __future__ import annotations
from .common import BasePreprocessor, AudioData
from .constants import TARGET_SAMPLE_RATE
import librosa

class Resampler(BasePreprocessor):
    def process(self, audio_data: AudioData) -> AudioData:
        if audio_data.sample_rate == TARGET_SAMPLE_RATE:
            return audio_data
        
        resampled_audio = librosa.resample(
            y=audio_data.audio,
            orig_sr=audio_data.sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

        return AudioData(audio=resampled_audio, sample_rate=TARGET_SAMPLE_RATE)