from __future__ import annotations
from .common import BasePreprocessor, AudioData
import librosa

class MonoConverter(BasePreprocessor):
    def process(self, audio_data: AudioData) -> AudioData:
        if audio_data.audio.ndim == 1:
            return audio_data
        
        mono_audio = librosa.to_mono(audio_data.audio)

        return AudioData(audio=mono_audio, sample_rate=audio_data.sample_rate)
    