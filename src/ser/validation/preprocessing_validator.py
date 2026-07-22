from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import librosa
import numpy as np
import pandas as pd
from ..preprocessing.constants import TARGET_SAMPLE_RATE, TARGET_RMS, RMS_TOLERANCE

@dataclass(slots=True)
class ValidationResult:
    """
    Stores the result of a validation check.
    """

    validation: str
    status: str
    expected: str | int | float
    actual: str | int | float

class PreprocessingValidator:
    def __init__(self, metadata: pd.DataFrame, processed_root: Path):
        self.metadata = metadata.copy()
        self.processed_root = processed_root
        self.audio_files = sorted(self.processed_root.rglob("*.wav"))

    def validate(self) -> pd.DataFrame:
        results = [
            self._validate_file_count(),
            self._validate_audio_read(),
            self._validate_sample_rate(),
            self._validate_mono(),
            self._validate_rms(),
        ]

        return pd.DataFrame([asdict(result) for result in results])

    @staticmethod
    def _status(condition: bool) -> str:
        return "PASS" if condition else "FAIL"
    
    def _validate_file_count(self) -> ValidationResult:
        expected = len(self.metadata)
        actual = len(self.audio_files)
        status = self._status(expected == actual)

        return ValidationResult(
            validation = "File Count",
            status = status,
            expected = expected,
            actual = actual,
        )

    def _validate_audio_read(self) -> ValidationResult:
        expected = len(self.audio_files)
        success = 0

        for filepath in self.audio_files:
            try:
                librosa.load(
                    filepath,
                    sr=None,
                    mono=False,
                )
                success += 1
            
            except Exception:
                continue
            
        status = self._status(success == expected)

        return ValidationResult(
            validation = "Audio Read",
            status = status,
            expected = expected,
            actual = success,
        )

    def _validate_sample_rate(self) -> ValidationResult:
        expected = len(self.audio_files)
        valid = 0

        for filepath in self.audio_files:
            try:
                _, sample_rate = librosa.load(
                    filepath,
                    sr=None,
                    mono=False,
                )

                if sample_rate == TARGET_SAMPLE_RATE:
                    valid += 1
            
            except Exception:
                continue

        status = self._status(valid == expected)

        return ValidationResult(
            validation = "Sample Rate",
            status = status,
            expected = expected,
            actual = valid,
        )

    def _validate_mono(self) -> ValidationResult:
        expected = len(self.audio_files)
        valid = 0

        for filepath in self.audio_files:
            try:
                audio, _ = librosa.load(
                    filepath,
                    sr=None,
                    mono=False,
                )

                if audio.ndim == 1:
                    valid += 1

            except Exception:
                continue

        status = self._status(valid == expected)

        return ValidationResult(
            validation = "Mono Audio",
            status = status,
            expected = expected,
            actual = valid,
        )

    def _validate_rms(self) -> ValidationResult:
        expected = len(self.audio_files)
        valid = 0
        failed_files = []

        for filepath in self.audio_files:
            try:
                audio, _ = librosa.load(
                    filepath,
                    sr=None,
                    mono=False,
                )

                rms = np.sqrt(np.mean(audio**2))

                if abs(rms - TARGET_RMS) <= RMS_TOLERANCE:
                    valid += 1
                else:
                    failed_files.append(
                        {
                            "filepath": str(filepath),
                            "rms": rms,
                        }
                    )

            except Exception:
                continue

        if failed_files:
            print("\nFiles failing RMS validation:")
            for item in failed_files:
                print(f"{item['filepath']} -> RMS={item['rms']:.6f}")

        status = self._status(valid == expected)

        return ValidationResult(
            validation = "RMS Normalization",
            status = status,
            expected = expected,
            actual = valid,
        )