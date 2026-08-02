from pathlib import Path
import pandas as pd
import argparse

from ser.features.feature_dataset import FeatureDataset
from ser.models.trainer import ScenarioTrainer
from ser.models.evaluator import ScenarioEvaluator
from ser.models.training_config import (
    TrainingConfig,
    set_global_seed,
    configure_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "features" / "feature_index.csv"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "rm1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "rm1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training dan evaluasi skenario RM1 (within-corpus)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Bila tidak diisi, memakai nilai baku KNF-06.",
    )

    return parser.parse_args()

def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_ROOT / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Split tidak ditemukan: {path}. "
            "Jalankan scripts/05_dataset_split.py terlebih dahulu."
        )

    return pd.read_csv(path)

def main():
    args = parse_args()

    config = (
        TrainingConfig()
        if args.seed is None
        else TrainingConfig(random_seed=args.seed)
    )

    output_dir = OUTPUT_DIR / f"seed_{config.random_seed}"

    set_global_seed(config.random_seed)
    print(f"Perangkat aktif : {configure_device()}")
    print(f"Seed            : {config.random_seed}")

    dataset = FeatureDataset(FEATURES_PATH, INDEX_PATH)

    train_split = load_split("train")
    validation_split = load_split("validation")
    test_split = load_split("test")

    print("\n=== Skema RM1 (within-corpus) ===")

    trainer = ScenarioTrainer(
        dataset=dataset,
        output_dir=output_dir,
        config=config,
    )

    result = trainer.run(train_split, validation_split)

    print("\n=== Ringkasan Training ===")
    print(f"Epoch dijalankan       : {result.epochs_run}")
    print(f"Epoch terbaik          : {result.best_epoch}")
    print(f"Macro F1 validasi      : {result.best_score:.4f}")
    print(f"Waktu per epoch        : {result.seconds_per_epoch:.1f} detik "
          "(epoch pertama dikecualikan)")

    print("\n=== Evaluasi pada Data Uji ===")

    test_subset = dataset.build(test_split, "test")

    evaluator = ScenarioEvaluator(
        model=result.model,
        output_dir=output_dir,
    )

    summary = evaluator.evaluate(test_subset)

    print(summary.to_string(index=False))
    print(f"\nArtefak tersimpan : {output_dir}")


if __name__ == "__main__":
    main()