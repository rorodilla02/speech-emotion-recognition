from __future__ import annotations
from .constants import (
    ALL_DATASETS,
    TRAINING_DATASETS,
    TRAINING_COMBINED_LABEL,
)
import pandas as pd
import numpy as np


class DurationAnalyzer:
    """
    Menghitung statistik deskriptif durasi audio dari metadata.

    Kelas ini dapat dijalankan pada dua basis metadata:
    - file_inventory.csv      : durasi mentah (tahap data understanding)
    - processed_inventory.csv : durasi setelah preprocessing (dasar
                                penetapan target durasi)

    Catatan
    -------
    Kelas ini tidak:
    - membaca file metadata
    - menulis file output
    - memodifikasi data audio
    """

    COLUMNS = [
        "dataset",
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p5",
        "p75",
        "p90",
        "p95",
    ]

    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata.copy()

        required_columns = {
            "dataset",
            "duration",
        }

        missing = required_columns - set(self.metadata.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def analyze(self) -> pd.DataFrame:
        rows = []
        missing_datasets = []

        # Baris per dataset, termasuk INESCO (dibutuhkan Tabel 3.14)
        for dataset in ALL_DATASETS:
            group = self.metadata[self.metadata["dataset"] == dataset]

            if group.empty:
                missing_datasets.append(dataset)
                continue

            stats = self._calculate_statistics(group["duration"])
            stats["dataset"] = dataset
            rows.append(stats)

        # Baris gabungan data latih (RAVDESS + TESS + SAVEE)
        training = self.metadata[
            self.metadata["dataset"].isin(TRAINING_DATASETS)
        ]

        if training.empty:
            raise ValueError(
                "Tidak ada dataset latih pada metadata. "
                "Periksa kembali nilai kolom 'dataset'."
            )

        combined = self._calculate_statistics(training["duration"])
        combined["dataset"] = TRAINING_COMBINED_LABEL
        rows.append(combined)

        if missing_datasets:
            print(
                "Peringatan: dataset tidak ditemukan pada metadata -> "
                f"{missing_datasets}"
            )

        summary = pd.DataFrame(rows)

        return summary[self.COLUMNS]

    def _calculate_statistics(self, durations: pd.Series) -> dict:
        durations = durations.dropna()

        if durations.empty:
            raise ValueError("Tidak ada nilai durasi yang valid.")

        values = durations.to_numpy(dtype=float)

        return {
            "count": int(durations.count()),
            "mean": round(float(durations.mean()), 3),
            "median": round(float(durations.median()), 3),
            # ddof=1 (default pandas), konsisten dengan StatisticsGenerator
            "std": round(float(durations.std()), 3),
            "min": round(float(durations.min()), 3),
            "max": round(float(durations.max()), 3),
            "p5": round(float(np.percentile(values, 5)), 3),
            "p75": round(float(np.percentile(values, 75)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
        }