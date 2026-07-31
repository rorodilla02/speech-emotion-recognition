from pathlib import Path

from ser.validation.feature_validator import FeatureValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "features" / "feature_index.csv"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "feature_validation.csv"


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset fitur tidak ditemukan: {FEATURES_PATH}. "
            "Jalankan scripts/09_feature_extraction.py terlebih dahulu."
        )

    print("Mulai validasi fitur...")

    validator = FeatureValidator(
        features_path=FEATURES_PATH,
        index_path=INDEX_PATH,
        split_root=SPLIT_ROOT,
    )

    summary = validator.validate()
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print(summary.to_string(index=False))
    print(f"\nLaporan disimpan: {OUTPUT_PATH}")

    failed = summary[summary["status"] == "FAIL"]

    if not failed.empty:
        raise SystemExit(
            f"\nValidasi GAGAL pada {len(failed)} pemeriksaan. "
            "Perbaiki sebelum melanjutkan ke tahap Modeling."
        )

    print("Seluruh validasi fitur lolos.")


if __name__ == "__main__":
    main()