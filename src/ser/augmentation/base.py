from __future__ import annotations
from abc import ABC, abstractmethod
from ..preprocessing.common import AudioData

class BaseAugmentor(ABC):
    """
    Base class for all audio augmentation modules.
    """

    @abstractmethod
    def apply(self, audio: AudioData) -> AudioData:
        raise NotImplementedError
