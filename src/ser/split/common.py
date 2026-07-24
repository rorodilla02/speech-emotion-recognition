from dataclasses import dataclass
import pandas as pd

@dataclass(slots=True)
class SpeakerSplit:
    train: list[str]
    validation: list[str]
    test: list[str]

@dataclass(slots=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame