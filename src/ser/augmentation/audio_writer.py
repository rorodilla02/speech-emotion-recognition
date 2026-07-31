from __future__ import annotations
from pathlib import Path
import soundfile as sf
import numpy as np
from ..preprocessing.common import AudioData
from ..preprocessing.constants import PROCESSED_AUDIO_SUBTYPE

class AudioWriter:
    """
    Writes augmented audio to disk.
    """

    def write(self, audio: AudioData, outputh_path: Path):
        outputh_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            file=outputh_path,
            data=np.asarray(audio.audio, dtype=np.float32),
            samplerate=audio.sample_rate,
            subtype=PROCESSED_AUDIO_SUBTYPE,
        )