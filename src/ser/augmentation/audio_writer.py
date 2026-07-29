from __future__ import annotations
from pathlib import Path
import soundfile as sf
from ..preprocessing.common import AudioData

class AudioWriter:
    """
    Writes augmented audio to disk.
    """

    def write(self, audio: AudioData, outputh_path: Path):
        outputh_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n=== WRITER INPUT ===")
        print(audio.audio.shape)
        print(audio.sample_rate)

        sf.write(
            file=outputh_path, 
            data=audio.audio, 
            samplerate=audio.sample_rate
        )

        info = sf.info(outputh_path)

        print("\n=== WRITER OUTPUT ===")
        print(info.frames)
        print(info.samplerate)
        print(info.duration)