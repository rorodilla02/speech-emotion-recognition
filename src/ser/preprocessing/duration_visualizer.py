from __future__ import annotations
from .constants  import TRAINING_DATASETS
import pandas as pd
import matplotlib.pyplot as plt

class DurationVisualizer:
    """
    Visualize audio duration distribution.

    This class is responsible for generating exploratory
    visualizations from duration metadata.

    Notes
    -----
    This class does not:
    - compute statistics
    - read metadata files
    - save output files
    """
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata.copy()

    def plot_distribution(self, summary: pd.DataFrame):
        self.metadata = self._filter_training_datasets()
        durations = self.metadata["duration"].dropna()
        combined = summary[summary["dataset"]=="Combined"].iloc[0]

        plt.figure(figsize=(10,6))
        plt.hist(durations, bins=30,)
        plt.axvline(combined["median"], color="green", linestyle="--", linewidth=2, label="Median")
        plt.axvline(combined["p75"], color="blue", linestyle="--", linewidth=2, label="P75")
        plt.axvline(combined["p90"], color="orange", linestyle="--", linewidth=2, label="P90")
        plt.axvline(combined["p95"], color="red", linestyle="--", linewidth=2, label="P95")
        
        plt.title("Training Dataset Duration Distribution")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.legend()

        plt.show()

    def _filter_training_datasets(self) -> pd.DataFrame:
        return self.metadata[
            self.metadata["dataset"].isin(TRAINING_DATASETS)
        ].copy()