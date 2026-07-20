from __future__ import annotations
from .constants import TRAINING_DATASETS
import pandas as pd
import numpy as np

class DurationAnalyzer:
    """
    Analyze audio duration statistics from dataset metadata.

    This class computes descriptive statistics for audio duration
    using metadata generated during the Dataset Audit stage.
    The statistics are used to determine the target duration
    for audio normalization.

    Notes
    -----
    This class does not:
    - read metadata files
    - write output files
    - modify audio data
    """
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
        metadata = self._filter_training_datasets()
        summary = self._build_summary(metadata)
        
        return summary

    def _filter_training_datasets(self) -> pd.DataFrame:
        return self.metadata[
            self.metadata["dataset"].isin(TRAINING_DATASETS)
        ].copy()

    def _calculate_statistics(self, durations: pd.Series) -> dict:
        durations = durations.dropna()
        
        return {
            "count": int(durations.count()), 
            "mean": round(float(durations.mean()),3),
            "median": round(float(durations.median()),3),
            "std": round(float(durations.std()),3),
            "min": round(float(durations.min()),3),
            "max": round(float(durations.max()),3),
            "p75": round(float(np.percentile(durations.to_numpy(), 75)),3),
            "p90": round(float(np.percentile(durations.to_numpy(), 90)),3),
            "p95": round(float(np.percentile(durations.to_numpy(), 95)),3),
        }

    def _build_summary(self, metadata: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for dataset in TRAINING_DATASETS:
            group = metadata[
                metadata["dataset"] == dataset
            ]

            if group.empty:
                raise ValueError(f"No samples found for dataset '{dataset}.")            
            
            stats = self._calculate_statistics(group["duration"])
            stats["dataset"] = dataset
            rows.append(stats)

        combined = self._calculate_statistics(metadata["duration"])
        combined["dataset"] = "Combined"
        rows.append(combined)

        summary = pd.DataFrame(rows)
        columns = [
            "dataset",
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "p75",
            "p90",
            "p95",
        ]

        return summary[columns]