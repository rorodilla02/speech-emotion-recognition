from __future__ import annotations
import random
import numpy as np
from .base import BaseAugmentor
from ..preprocessing.common import AudioData

class NoiseInjection(BaseAugmentor):
    """
    Applies Gaussian noise to an audio signal.
    """

    def __init__(self, noise_factor: float=0.005, probability: float=0.5):
        self.noise_factor = noise_factor
        self.probability = probability

    def apply(self, audio: AudioData) -> AudioData:
        if random.random() > self.probability:
            return audio

        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=audio.audio.shape,
        )

        augmented = (audio.audio + self.noise_factor * noise)
        augmented = np.clip(augmented, -1.0, 1.0)

        return AudioData(
            audio=augmented.astype(audio.audio.dtype),
            sample_rate=audio.sample_rate,
        )