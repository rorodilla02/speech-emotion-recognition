from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
from typing import Protocol

@dataclass
class AudioMetadata:
    # Metadata dari Parser
    dataset: str
    filepath: Path
    filename: str
    speaker: str
    raw_label: str
    emotion: str
    
    # Metadata dari Audio Reader
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    
    # Status Pembacaan
    status: str = "PENDING"
    error_message: Optional[str] = None

class ParseErrorType(Enum):
    INVALID_FILENAME = "InvalidFilename"
    UNKNOWN_EMOTION = "UnknownEmotion"
    AUDIO_READ_ERROR = "AudioReadError"
    UNEXPECTED_ERROR = "UnexpectedError"

@dataclass
class FailedFile:
    dataset: str
    filepath: Path
    filename: str
    error_type: ParseErrorType
    error_message: str

@dataclass
class ParseResult:
    metadata: list[AudioMetadata]
    failed_files: list[FailedFile]

class BaseParser(Protocol):
    def parse(self) -> "ParseResult":
        ...

@dataclass
class DatasetStatistics:
    total_files: int
    speakers: list[str]
    total_speakers: int

    sample_rates: list[int]

    min_duration: float
    max_duration: float
    mean_duration: float

    emotion_distribution: dict[str, int]

# RAVDESS Emotion Label Mapping
RAVDESS_EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}

# TESS Emotion Label Mapping
TESS_EMOTION_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "ps": "Surprise",
}

# SAVEE Emotion Label Mapping
SAVEE_EMOTION_MAP = {
    "a": "Angry",
    "d": "Disgust",
    "f": "Fear",
    "h": "Happy",
    "n": "Neutral",
    "sa": "Sad",
    "su": "Surprise",
}

# INESCO Emotion Label Mapping
INESCO_EMOTION_MAP = {
    "a": "Angry",
    "h": "Happy",
    "s": "Sad",
}