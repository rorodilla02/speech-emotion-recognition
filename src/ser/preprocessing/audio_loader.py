from __future__ import annotations
from pathlib import Path
from .common import AudioData
import librosa

class AudioLoader:
    def load(self, filepath: Path) -> AudioData:
        audio, sample_rate = librosa.load(filepath, sr=None, mono=False)

        return AudioData(audio=audio, sample_rate=sample_rate)