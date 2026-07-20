from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass(slots=True)
class AudioData:
    """
    Container for audio data during preprocessing.

    Attributes
    ----------
    audio : np.ndarray
        Audio signal.

    sample_rate : int
        Sampling rate of the audio signal.
    """
    audio: np.ndarray
    sample_rate: int

class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, audio_data: AudioData) -> AudioData:
        ...
    