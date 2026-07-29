from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import soundfile as sf


@dataclass(slots=True)
class ValidationResult:
    validation: str
    status: str
    expected: int | str
    actual: int | str


class AugmentationValidator:

    def __init__(
        self,
        metadata: pd.DataFrame,
        augmented_root: Path,
    ):
        self.metadata = metadata.copy()
        self.augmented_root = augmented_root

    @staticmethod
    def _status(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    def validate(self) -> pd.DataFrame:

        results = [
            self._validate_total_files(),
            self._validate_file_exists(),
            self._validate_sample_rate(),
            self._validate_duration(),
        ]

        return pd.DataFrame(
            [asdict(r) for r in results]
        )

    def _validate_total_files(self) -> ValidationResult:

        expected = len(self.metadata)

        actual = 0

        for path in self.augmented_root.rglob("*.wav"):
            actual += 1

        return ValidationResult(
            validation="Augmented Total Files",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_file_exists(self) -> ValidationResult:
        missing_files = []

        for _, row in self.metadata.iterrows():

            audio_path = self.augmented_root / row["filepath"]

            if not audio_path.exists():
                missing_files.append(row["filepath"])

        if missing_files:
            report = pd.DataFrame(
                {"missing_file": missing_files}
            )

            report.to_csv(
                self.augmented_root / "missing_files.csv",
                index=False,
            )

        return ValidationResult(
            validation="Augmented File Exists",
            status=self._status(len(missing_files) == 0),
            expected=len(self.metadata),
            actual=len(self.metadata) - len(missing_files),
        )
    
    def _validate_sample_rate(self) -> ValidationResult:
        invalid_files = []

        expected_sample_rate = 16000

        for _, row in self.metadata.iterrows():

            filepath = self.augmented_root / row["filepath"]

            info = sf.info(filepath)

            if info.samplerate != expected_sample_rate:

                invalid_files.append({
                    "filepath": row["filepath"],
                    "expected": expected_sample_rate,
                    "actual": info.samplerate,
                })

        if invalid_files:

            pd.DataFrame(invalid_files).to_csv(
                self.augmented_root / "invalid_sample_rate.csv",
                index=False,
            )

        return ValidationResult(
            validation="Augmented Sample Rate",
            status=self._status(len(invalid_files) == 0),
            expected=expected_sample_rate,
            actual=expected_sample_rate if len(invalid_files) == 0 else "Mismatch",
        )

    def _validate_duration(self) -> ValidationResult:
        invalid_files = []

        tolerance = 1e-3

        for _, row in self.metadata.iterrows():

            filepath = self.augmented_root / row["filepath"]

            info = sf.info(filepath)

            actual_duration = info.frames / info.samplerate
            expected_duration = row["duration"]

            if abs(actual_duration - expected_duration) > tolerance:

                invalid_files.append({
                    "filepath": row["filepath"],
                    "expected_duration": expected_duration,
                    "actual_duration": actual_duration,
                })

        if invalid_files:

            pd.DataFrame(invalid_files).to_csv(
                self.augmented_root / "invalid_duration.csv",
                index=False,
            )

        return ValidationResult(
            validation="Augmented Duration",
            status=self._status(len(invalid_files) == 0),
            expected=len(self.metadata),
            actual=len(self.metadata) - len(invalid_files),
        )