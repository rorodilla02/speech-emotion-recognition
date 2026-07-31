from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .constants import (
    LABEL_TO_INDEX,
    SOURCE_PROCESSED,
    SOURCE_AUGMENTED,
)

TRAIN_ROLE = "train"
VALID_ROLES = ("train", "validation", "test")


@dataclass(slots=True)
class FeatureSubset:
    """
    Satu subset fitur siap pakai untuk training maupun evaluasi.
    """

    features: np.ndarray          # (N, 51, 401)
    labels: np.ndarray            # (N,) indeks kelas
    manifest: pd.DataFrame        # metadata tiap baris


class FeatureDataset:
    """
    Menyusun subset fitur untuk sebuah split berdasarkan feature_index.

    Aturan augmentasi
    -----------------
    Data augmentasi HANYA disertakan pada peran 'train', dan hanya
    untuk berkas yang sumbernya memang berada pada split tersebut.
    Aturan ini bersifat tetap dan tidak dapat dilonggarkan lewat
    parameter, karena augmentasi yang bocor ke validation atau test
    akan menggelembungkan metrik secara artifisial (risiko R-03).

    Catatan
    -------
    Kelas ini tidak:
    - mengekstraksi fitur
    - membuat split
    - melatih model
    """

    def __init__(self, features_path: Path, index_path: Path):
        self.features = np.load(features_path, mmap_mode="r")
        self.index = pd.read_csv(index_path)

        if len(self.index) != self.features.shape[0]:
            raise ValueError(
                f"Jumlah baris tidak sama: index {len(self.index)} "
                f"vs features {self.features.shape[0]}"
            )

    def build(self, split: pd.DataFrame, role: str) -> FeatureSubset:
        if role not in VALID_ROLES:
            raise ValueError(f"Peran tidak dikenal: {role}")

        if split.empty:
            return self._empty_subset()

        filepaths = set(split["filepath"].astype(str))

        selected = self.index[
            (self.index["source"] == SOURCE_PROCESSED)
            & (self.index["filepath"].astype(str).isin(filepaths))
        ]

        if role == TRAIN_ROLE:
            augmented = self.index[
                (self.index["source"] == SOURCE_AUGMENTED)
                & (self.index["filepath"].astype(str).isin(filepaths))
            ]
            selected = pd.concat([selected, augmented], ignore_index=True)

        selected = selected.sort_values("row_index").reset_index(drop=True)

        self._verify_coverage(split, selected, role)

        rows = selected["row_index"].to_numpy()
        features = np.asarray(self.features[rows], dtype=np.float32)
        labels = selected["emotion"].map(LABEL_TO_INDEX).to_numpy()

        if np.isnan(labels.astype(float)).any():
            unknown = sorted(
                set(selected["emotion"]) - set(LABEL_TO_INDEX)
            )
            raise ValueError(f"Label emosi tidak dikenal: {unknown}")

        return FeatureSubset(
            features=features,
            labels=labels.astype(np.int64),
            manifest=selected,
        )

    @staticmethod
    def _verify_coverage(
        split: pd.DataFrame,
        selected: pd.DataFrame,
        role: str,
    ):
        expected = len(split)
        actual = len(
            selected[selected["source"] == SOURCE_PROCESSED]
        )

        if actual != expected:
            raise ValueError(
                f"Peran '{role}': jumlah berkas asli tidak sesuai "
                f"({actual} ditemukan dari {expected} pada split). "
                "Periksa apakah ekstraksi fitur sudah dijalankan ulang."
            )

    def _empty_subset(self) -> FeatureSubset:
        return FeatureSubset(
            features=np.empty((0, *self.features.shape[1:]), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            manifest=self.index.iloc[0:0].copy(),
        )