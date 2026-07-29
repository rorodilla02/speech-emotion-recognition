from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class AudioSample:
    """
    Shared audio container used by every augmentation module.
    """

    audio: np.ndarray
    sample_rate: int
    filepath: str
    dataset: str
    emotion: str
    speaker: str