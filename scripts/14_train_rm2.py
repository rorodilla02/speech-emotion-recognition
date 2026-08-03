from pathlib import Path
import argparse
import pandas as pd

from ser.features.feature_dataset import FeatureDataset
from ser.models.trainer import ScenarioTrainer
from ser.models.evaluator import ScenarioEvaluator
from ser.models.validation_splitter import ValidationSplitter
from ser.models.training_config import (
    TrainingConfig,
    set_global_seed,
    configure_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "features" / "feature_index.csv"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "rm2"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "models" / "rm2"

FOLDS = {
    "fold_1": ("RAVDESS + TESS", "SAVEE"),
    "fold_2": ("RAVDESS + SAVEE", "TESS"),
    "fold_3": ("TESS + SAVEE", "RAVDESS"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training dan evaluasi RM2 (Leave-One-Corpus-Out)."
    )
    parser.add_argument("--fold", required=True, choices=sorted(FOLDS))
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    config = (
        TrainingConfig()
        if args.seed is None
        else TrainingConfig(random_seed=args.seed)
    )

    output_dir = OUTPUT_ROOT / args.fold / f"seed_{config.random_seed}"

    set_global_seed(config.random_seed)
    print(f"Perangkat aktif : {configure_device()}")

    train_corpora, test_corpus = FOLDS[args.fold]
    print(f"\n=== RM2 {args.fold} ===")
    print(f"Latih : {train_corpora}")
    print(f"Uji   : {test_corpus}")
    print(f"Seed  : {config.random_seed}")

    dataset = FeatureDataset(FEATURES_PATH, INDEX_PATH)

    fold_root = SPLIT_ROOT / args.fold
    full_train = pd.read_csv(fold_root / "train.csv")
    test_split = pd.read_csv(fold_root / "test.csv")

    print("\nPemisahan validasi internal:")
    train_split, validation_split = ValidationSplitter().split(full_train)

    # Disimpan agar komposisi validasi dapat diaudit ulang.
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_split.to_csv(output_dir / "validation_split.csv", index=False)

    trainer = ScenarioTrainer(
        dataset=dataset,
        output_dir=output_dir,
        config=config,
    )

    result = trainer.run(train_split, validation_split)

    print("\n=== Ringkasan Training ===")
    print(f"Epoch dijalankan  : {result.epochs_run}")
    print(f"Epoch terbaik     : {result.best_epoch}")
    print(f"Macro F1 validasi : {result.best_score:.4f}")
    print(f"Waktu per epoch   : {result.seconds_per_epoch:.1f} detik")

    print(f"\n=== Evaluasi pada {test_corpus} ===")

    test_subset = dataset.build(test_split, "test")

    evaluator = ScenarioEvaluator(model=result.model, output_dir=output_dir)
    summary = evaluator.evaluate(test_subset)

    print(summary.to_string(index=False))
    print(f"\nArtefak tersimpan : {output_dir}")


if __name__ == "__main__":
    main()