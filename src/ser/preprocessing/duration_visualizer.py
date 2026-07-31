from __future__ import annotations
from pathlib import Path
from .constants import ALL_DATASETS, TRAINING_DATASETS
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # tanpa display server (WSL2)
import matplotlib.pyplot as plt


class DurationVisualizer:
    """
    Menghasilkan visualisasi sebaran durasi audio.

    Catatan
    -------
    Kelas ini tidak:
    - menghitung statistik
    - membaca file metadata
    - memodifikasi metadata yang diterima
    """

    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata.copy()

    def plot_boxplot(self, output_path: Path) -> Path:
        """
        Gambar 3.4: perbandingan sebaran durasi keempat dataset
        pada satu sumbu yang sama.
        """
        data = []
        labels = []

        for dataset in ALL_DATASETS:
            durations = self.metadata.loc[
                self.metadata["dataset"] == dataset, "duration"
            ].dropna()

            if durations.empty:
                continue

            data.append(durations.to_numpy(dtype=float))
            labels.append(dataset.upper())

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.title("Sebaran Durasi Audio per Dataset")
        plt.xlabel("Dataset")
        plt.ylabel("Durasi (detik)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def plot_histogram(self, summary: pd.DataFrame, output_path: Path) -> Path:
        """
        Histogram durasi gabungan data latih beserta garis persentil.
        Dipakai sebagai dasar penetapan target durasi.
        """
        durations = self.metadata.loc[
            self.metadata["dataset"].isin(TRAINING_DATASETS), "duration"
        ].dropna()

        combined = summary[summary["dataset"] == "training_combined"].iloc[0]

        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=30)

        for column, color, label in (
            ("median", "green", "Median"),
            ("p75", "blue", "P75"),
            ("p90", "orange", "P90"),
            ("p95", "red", "P95"),
        ):
            plt.axvline(
                combined[column],
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{label} = {combined[column]:.2f} s",
            )

        plt.title("Sebaran Durasi Gabungan Data Latih")
        plt.xlabel("Durasi (detik)")
        plt.ylabel("Jumlah Sampel")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path