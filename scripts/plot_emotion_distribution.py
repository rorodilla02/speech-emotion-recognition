"""
Visualisasi distribusi kelas emosi per dataset.

Merekonstruksi Gambar 3.3 dari processed_inventory.csv, sehingga angka
pada Tabel 3.12 dan chart yang dihasilkan bersumber dari data yang sama
dan dapat diverifikasi ulang.

    python scripts/plot_emotion_distribution.py

Basis data: data/metadata/processed_inventory.csv
    - kelas "Calm" sudah dikeluarkan (di luar ruang lingkup Bab 1)
    - speaker "OA" sudah digabung ke "OAF" (T-01)
    - file korup INESCO sudah dikeluarkan (T-03)
Ini SENGAJA berbeda dari emotion_distribution.csv hasil audit mentah,
yang belum melalui pembersihan tersebut.
"""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # tanpa display server, konsisten dengan duration_visualizer
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_inventory.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "figures" / "emotion_distribution.png"

TRAINING_DATASETS = ("ravdess", "tess", "savee")
DATASET_LABELS = {
    "ravdess": "RAVDESS",
    "tess": "TESS",
    "savee": "SAVEE",
}
COMBINED_LABEL = "Gabungan Latih"
CROSS_LINGUAL_LABEL = "INESCO"

# Urutan tampilan mengikuti Tabel 3.12 (Marah, Jijik, Takut, Senang, Sedih,
# Terkejut, Netral), BUKAN urutan alfabetis EMOTION_LABELS pada
# src/ser/features/constants.py. Urutan model tetap alfabetis; urutan di
# sini murni untuk keterbacaan pada naskah dan tidak memengaruhi indeks
# label yang dipakai CNN.
DISPLAY_ORDER = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)

# Kelas yang memang tidak tersedia pada INESCO (bukan bernilai nol, tapi
# di luar cakupan desain korpus tersebut -- lihat subbab 2.7.4).
CROSS_LINGUAL_LABELS = ("Angry", "Happy", "Sad")


def build_distribution_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Menyusun tabel jumlah sampel per (dataset, kelas emosi), termasuk
    baris Gabungan Latih dan INESCO, dengan urutan kolom dan baris
    yang sama dengan Tabel 3.12.
    """
    table = pd.DataFrame(
        0, index=DISPLAY_ORDER, columns=list(DATASET_LABELS.values())
    )

    for dataset_key, dataset_label in DATASET_LABELS.items():
        subset = metadata[metadata["dataset"] == dataset_key]
        counts = subset["emotion"].value_counts()

        for emotion in DISPLAY_ORDER:
            table.loc[emotion, dataset_label] = int(counts.get(emotion, 0))

    table[COMBINED_LABEL] = table[list(DATASET_LABELS.values())].sum(axis=1)

    inesco = metadata[metadata["dataset"] == "inesco"]
    inesco_counts = inesco["emotion"].value_counts()

    table[CROSS_LINGUAL_LABEL] = 0
    for emotion in CROSS_LINGUAL_LABELS:
        table.loc[emotion, CROSS_LINGUAL_LABEL] = int(
            inesco_counts.get(emotion, 0)
        )

    return table


def plot_distribution(table: pd.DataFrame, output_path: Path) -> Path:
    series_labels = list(table.columns)
    n_series = len(series_labels)
    n_classes = len(table.index)

    x = np.arange(n_classes)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, series in enumerate(series_labels):
        offset = (i - (n_series - 1) / 2) * width
        values = table[series].to_numpy()

        bars = ax.bar(x + offset, values, width, label=series)

        for bar, value in zip(bars, values):
            if value == 0:
                continue
            ax.annotate(
                str(value),
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("Jumlah Sampel")
    ax.set_title("Distribusi Kelas Emosi per Dataset")
    ax.legend(loc="upper right", ncol=n_series)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {METADATA_PATH}. "
            "Jalankan scripts/03_preprocessing.py terlebih dahulu."
        )

    metadata = pd.read_csv(METADATA_PATH)
    table = build_distribution_table(metadata)

    print("=== Distribusi Kelas Emosi (basis: processed_inventory.csv) ===\n")
    print(table.to_string())

    print(f"\nTotal per dataset:")
    print(table[list(DATASET_LABELS.values()) + [COMBINED_LABEL]].sum().to_string())

    # Peringatan konsistensi: bandingkan dengan total baris metadata
    training_total = int(
        metadata[metadata["dataset"].isin(TRAINING_DATASETS)].shape[0]
    )
    combined_total = int(table[COMBINED_LABEL].sum())

    if training_total != combined_total:
        print(
            f"\nPERINGATAN: total baris data latih pada metadata ({training_total}) "
            f"tidak sama dengan total Gabungan Latih pada tabel ({combined_total}). "
            "Periksa apakah ada kelas emosi di luar DISPLAY_ORDER yang lolos "
            "dari proses cleaning."
        )

    output_path = plot_distribution(table, OUTPUT_PATH)
    print(f"\nGambar disimpan: {output_path}")
    print(
        "\nCatatan: batang INESCO pada kelas Disgust, Fear, dan Surprise "
        "bernilai nol karena ketiga kelas tersebut memang tidak tersedia "
        "pada desain korpus INESCO (lihat subbab 2.7.4), bukan karena "
        "tidak ada sampel yang terdeteksi."
    )


if __name__ == "__main__":
    main()