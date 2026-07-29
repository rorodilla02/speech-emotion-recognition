from __future__ import annotations
from ..preprocessing.common import AudioData
from .base import BaseAugmentor

class AugmentationPipeline:
    """
    Sequential audio augmentation pipeline.
    """

    def __init__(self, augmentors: list[BaseAugmentor]):
        self.augmentors = augmentors

    def apply(self, audio: AudioData) -> AudioData:
        augmented_audio = audio

        for augmentor in self.augmentors:
            augmented_audio = augmentor.apply(augmented_audio)

        print("\n=== PIPELINE ===")
        print(augmented_audio.audio.shape)
        print(augmented_audio.sample_rate)

        return augmented_audio