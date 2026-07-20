from __future__ import annotations
from .common import AudioData
from .mono_converter import MonoConverter
from .resampler import Resampler
from .silence_trimmer import SilenceTrimmer
from .rms_normalizer import RMSNormalizer

class AudioPreprocessor:
    """
    Apply the audio preprocessing pipeline.

    Pipeline
    --------
    1. Convert to mono
    2. Resample to target sample rate
    3. Trim leading and trailing silence
    4. Normalize RMS amplitude
    """

    def __init__(self):
        self.mono_converter = MonoConverter()
        self.resampler = Resampler()
        self.silence_trimmer = SilenceTrimmer()
        self.rms_normalizer = RMSNormalizer()

    def process(self, audio_data: AudioData) -> AudioData:
        audio_data = self.mono_converter.process(audio_data)
        audio_data = self.resampler.process(audio_data)
        audio_data = self.silence_trimmer.process(audio_data)
        audio_data = self.rms_normalizer.process(audio_data)

        return audio_data