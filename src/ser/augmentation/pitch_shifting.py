from __future__ import annotations
import random
import librosa
from .base import BaseAugmentor
from ..preprocessing.common import AudioData

class PitchShifting(BaseAugmentor):
    """
    Applies pitch shifting augmentation.
    """

    def __init__(self, min_steps: int=-2, max_steps: int=2, probability: float=0.5):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.probability = probability

    def apply(self, audio: AudioData) -> AudioData:
        if random.random() > self.probability:
            return audio

        n_steps = random.randint(
            self.min_steps,
            self.max_steps,
        )

        augmented_audio = librosa.effects.pitch_shift(
            y=audio.audio,
            sr=audio.sample_rate,
            n_steps=n_steps,
        )

        print(audio.audio.shape)
        print(augmented_audio.shape)

        return AudioData(
            audio=augmented_audio.astype(audio.audio.dtype),
            sample_rate=audio.sample_rate,
        )