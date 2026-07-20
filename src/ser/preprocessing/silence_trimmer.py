from __future__ import annotations
from .common import AudioData, BasePreprocessor
from .constants import TRIM_TOP_DB
import librosa

class SilenceTrimmer(BasePreprocessor):
    def process(self, audio_data: AudioData) -> AudioData:
        trimmed_audio, _ = librosa.effects.trim(
            y=audio_data.audio,
            top_db=TRIM_TOP_DB,
        )

        return AudioData(audio=trimmed_audio, sample_rate=audio_data.sample_rate)