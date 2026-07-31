from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from ..features.constants import (
    FEATURE_SHAPE,
    EMOTION_LABELS,
    CROSS_LINGUAL_LABELS,
    SOURCE_PROCESSED,
    SOURCE_AUGMENTED,
)
from ..features.feature_dataset import FeatureDataset


@dataclass(slots=True)
class ValidationResult:
    validation: str
    status: str
    expected: str | int | float
    actual: str | int | float


class FeatureValidator:
    """
    Memvalidasi dataset fitur hasil ekstraksi.

    Catatan
    -------
    Kelas ini tidak:
    - mengekstraksi ulang fitur
    - memperbaiki data yang tidak valid
    """

    def __init__(
        self,
        features_path: Path,
        index_path: Path,
        split_root: Path,
        sample_size: int = 500,
        random_seed: int = 42,
    ):
        self.features_path = features_path
        self.index_path = index_path
        self.split_root = split_root
        self.sample_size = sample_size
        self.random_seed = random_seed

        self.features = np.load(features_path, mmap_mode="r")
        self.index = pd.read_csv(index_path)

    def validate(self) -> pd.DataFrame:
        results = [
            self._validate_row_count(),
            self._validate_shape(),
            self._validate_dtype(),
            self._validate_finite(),
            self._validate_label_space(),
            self._validate_cross_lingual_labels(),
            self._validate_augmentation_source(),
        ]
        results.extend(self._validate_rm1_leakage())

        return pd.DataFrame([asdict(result) for result in results])

    @staticmethod
    def _status(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    # --- integritas larik --------------------------------------------------

    def _validate_row_count(self) -> ValidationResult:
        expected = len(self.index)
        actual = int(self.features.shape[0])

        return ValidationResult(
            validation="Feature Row Count",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_shape(self) -> ValidationResult:
        expected = str(FEATURE_SHAPE)
        actual = str(tuple(self.features.shape[1:]))

        return ValidationResult(
            validation="Feature Shape",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_dtype(self) -> ValidationResult:
        expected = "float32"
        actual = str(self.features.dtype)

        return ValidationResult(
            validation="Feature Dtype",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_finite(self) -> ValidationResult:
        """
        Pemeriksaan NaN dan Inf pada cuplikan acak, agar tidak perlu
        memuat seluruh larik ke memori.
        """
        rng = np.random.default_rng(self.random_seed)
        total = int(self.features.shape[0])
        size = min(self.sample_size, total)
        rows = np.sort(rng.choice(total, size=size, replace=False))

        finite = 0
        for row in rows:
            if np.isfinite(self.features[row]).all():
                finite += 1

        return ValidationResult(
            validation=f"Feature Finite (sample {size})",
            status=self._status(finite == size),
            expected=size,
            actual=finite,
        )

    # --- ruang label -------------------------------------------------------

    def _validate_label_space(self) -> ValidationResult:
        actual_labels = set(self.index["emotion"])
        unknown = actual_labels - set(EMOTION_LABELS)

        return ValidationResult(
            validation="Label Space",
            status=self._status(not unknown),
            expected=len(EMOTION_LABELS),
            actual=len(actual_labels) if not unknown else f"unknown={sorted(unknown)}",
        )

    def _validate_cross_lingual_labels(self) -> ValidationResult:
        inesco = self.index[self.index["dataset"] == "inesco"]
        actual_labels = set(inesco["emotion"])
        unexpected = actual_labels - set(CROSS_LINGUAL_LABELS)

        return ValidationResult(
            validation="INESCO Label Subset",
            status=self._status(not unexpected),
            expected=len(CROSS_LINGUAL_LABELS),
            actual=len(actual_labels) if not unexpected else f"unexpected={sorted(unexpected)}",
        )

    def _validate_augmentation_source(self) -> ValidationResult:
        """
        Setiap baris augmentasi harus punya berkas asli yang bersesuaian,
        dan INESCO tidak boleh punya baris augmentasi sama sekali.
        """
        processed = set(
            self.index.loc[
                self.index["source"] == SOURCE_PROCESSED, "filepath"
            ]
        )
        augmented = self.index[self.index["source"] == SOURCE_AUGMENTED]

        orphan = set(augmented["filepath"]) - processed
        inesco_augmented = (augmented["dataset"] == "inesco").sum()

        condition = (not orphan) and inesco_augmented == 0

        return ValidationResult(
            validation="Augmentation Source",
            status=self._status(condition),
            expected="orphan=0, inesco_augmented=0",
            actual=f"orphan={len(orphan)}, inesco_augmented={int(inesco_augmented)}",
        )

    # --- kebocoran data pada RM1 ------------------------------------------

    def _validate_rm1_leakage(self) -> list[ValidationResult]:
        dataset = FeatureDataset(self.features_path, self.index_path)
        results = []

        splits = {}
        for role in ("train", "validation", "test"):
            path = self.split_root / "rm1" / f"{role}.csv"

            if not path.exists():
                return [
                    ValidationResult(
                        validation="RM1 Split Files",
                        status="FAIL",
                        expected=str(path),
                        actual="tidak ditemukan",
                    )
                ]

            splits[role] = pd.read_csv(path)

        subsets = {
            role: dataset.build(split, role)
            for role, split in splits.items()
        }

        # 1. Augmentasi hanya boleh muncul pada train
        for role in ("validation", "test"):
            augmented = (
                subsets[role].manifest["source"] == SOURCE_AUGMENTED
            ).sum()

            results.append(
                ValidationResult(
                    validation=f"RM1 {role.capitalize()} No Augmentation",
                    status=self._status(augmented == 0),
                    expected=0,
                    actual=int(augmented),
                )
            )

        # 2. Tidak ada berkas yang muncul di lebih dari satu subset
        filepaths = {
            role: set(subset.manifest["filepath"])
            for role, subset in subsets.items()
        }

        overlap = (
            (filepaths["train"] & filepaths["validation"])
            | (filepaths["train"] & filepaths["test"])
            | (filepaths["validation"] & filepaths["test"])
        )

        results.append(
            ValidationResult(
                validation="RM1 Filepath Overlap",
                status=self._status(not overlap),
                expected=0,
                actual=len(overlap),
            )
        )

        # 3. Tidak ada speaker yang muncul di train sekaligus test.
        #    TESS dikecualikan karena hanya punya dua speaker sehingga
        #    memakai stratified split (lihat subbab 3.4.2).
        speakers = {}
        for role, subset in subsets.items():
            manifest = subset.manifest
            speakers[role] = set(
                manifest.loc[manifest["dataset"] != "tess", "speaker"]
            )

        speaker_overlap = speakers["train"] & speakers["test"]

        results.append(
            ValidationResult(
                validation="RM1 Speaker Overlap (non-TESS)",
                status=self._status(not speaker_overlap),
                expected=0,
                actual=len(speaker_overlap),
            )
        )

        # 4. Ringkasan ukuran subset
        for role, subset in subsets.items():
            results.append(
                ValidationResult(
                    validation=f"RM1 {role.capitalize()} Size",
                    status="INFO",
                    expected=len(splits[role]),
                    actual=len(subset.manifest),
                )
            )

        return results