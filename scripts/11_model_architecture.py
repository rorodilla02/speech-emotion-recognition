from pathlib import Path
import numpy as np
import pandas as pd

from ser.features.constants import FEATURE_SHAPE
from ser.models.cnn_architecture import build_cnn
from ser.models.constants import MODEL_INPUT_SHAPE, N_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "models"
SUMMARY_PATH = OUTPUT_ROOT / "architecture_summary.txt"
LAYERS_PATH = OUTPUT_ROOT / "architecture_layers.csv"

BYTES_PER_FLOAT32 = 4
MEMORY_OVERHEAD_FACTOR = 2.5   # gradien dan state optimizer
BATCH_CANDIDATES = (16, 32, 64, 128)


def verify_feature_shape():
    """
    Memastikan bentuk fitur di disk sama dengan bentuk input model,
    agar tidak terjadi ketidaksesuaian antara tahap Data Preparation
    dan tahap Modeling.
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset fitur tidak ditemukan: {FEATURES_PATH}. "
            "Jalankan scripts/09_feature_extraction.py terlebih dahulu."
        )

    features = np.load(FEATURES_PATH, mmap_mode="r")
    actual = tuple(features.shape[1:])

    if actual != FEATURE_SHAPE:
        raise ValueError(
            f"Bentuk fitur tidak sesuai: {actual} != {FEATURE_SHAPE}"
        )

    print(f"Bentuk fitur di disk : {actual}")
    print(f"Bentuk input model   : {MODEL_INPUT_SHAPE}")
    print(f"Jumlah baris fitur   : {features.shape[0]}")


def build_layer_table(model) -> pd.DataFrame:
    records = []

    for layer in model.layers:
        output_shape = tuple(layer.output.shape[1:])
        activations = int(np.prod(output_shape)) if output_shape else 0

        records.append(
            {
                "layer": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(output_shape),
                "params": int(layer.count_params()),
                "activations": activations,
            }
        )

    return pd.DataFrame(records)


def report_memory(layers_table: pd.DataFrame):
    per_sample = int(layers_table["activations"].sum())
    per_sample_mb = per_sample * BYTES_PER_FLOAT32 / 1024 ** 2

    print(f"\nAktivasi per sampel : {per_sample:,} float "
          f"({per_sample_mb:.2f} MB)")
    print("Estimasi kebutuhan memori per batch:")

    for batch_size in BATCH_CANDIDATES:
        estimate = per_sample_mb * batch_size * MEMORY_OVERHEAD_FACTOR
        print(f"  batch {batch_size:>3} : {estimate:>8.0f} MB")


def main():
    print("Verifikasi konsistensi bentuk fitur...")
    verify_feature_shape()

    print("\nMembangun arsitektur CNN...")
    model = build_cnn()
    model.summary()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    layers_table = build_layer_table(model)
    layers_table.to_csv(LAYERS_PATH, index=False)

    report_memory(layers_table)

    print(f"\nJumlah kelas    : {N_CLASSES}")
    print(f"Total parameter : {model.count_params():,}")
    print(f"Ringkasan       : {SUMMARY_PATH}")
    print(f"Tabel lapisan   : {LAYERS_PATH}")


if __name__ == "__main__":
    main()