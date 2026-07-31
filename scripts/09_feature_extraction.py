from pathlib import Path
import pandas as pd

from ser.features.dataset_feature_extractor import DatasetFeatureExtractor
from ser.features.constants import FEATURE_SHAPE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_METADATA = (
    PROJECT_ROOT / "data" / "metadata" / "processed_inventory.csv"
)
AUGMENTED_METADATA = (
    PROJECT_ROOT / "data" / "augmented" / "augmented_inventory.csv"
)
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "audio"
AUGMENTED_ROOT = PROJECT_ROOT / "data" / "augmented"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "features"


def main():
    for path in (PROCESSED_METADATA, AUGMENTED_METADATA):
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata tidak ditemukan: {path}. "
                "Jalankan tahap sebelumnya terlebih dahulu."
            )

    processed_metadata = pd.read_csv(PROCESSED_METADATA)
    augmented_metadata = pd.read_csv(AUGMENTED_METADATA)

    print("Mulai ekstraksi fitur...")
    print(f"Processed : {len(processed_metadata)} berkas")
    print(f"Augmented : {len(augmented_metadata)} berkas")
    print(f"Bentuk    : {FEATURE_SHAPE}")

    extractor = DatasetFeatureExtractor(
        processed_metadata=processed_metadata,
        augmented_metadata=augmented_metadata,
        processed_root=PROCESSED_ROOT,
        augmented_root=AUGMENTED_ROOT,
        output_root=OUTPUT_ROOT,
    )

    index = extractor.run()

    print("\n=== Ringkasan ===")
    print(index.groupby(["source", "dataset"]).size().to_string())
    print(f"\nTotal baris fitur: {len(index)}")
    print("Ekstraksi fitur selesai.")


if __name__ == "__main__":
    main()