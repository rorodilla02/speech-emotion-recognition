from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import f1_score

from ser.features.feature_dataset import FeatureDataset
from ser.features.constants import EMOTION_LABELS
from ser.models.rm3_evaluator import RM3Evaluator, TARGET_EMOTIONS
from ser.models.training_config import configure_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "features" / "feature_index.csv"
TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "rm3" / "test.csv"
RM1_ROOT = PROJECT_ROOT / "data" / "models" / "rm1"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "models" / "rm3"

SEEDS = (42, 43, 44, 45, 46)


def rm1_baseline_on_target_classes() -> pd.DataFrame:
    """
    Menghitung ulang macro F1 RM1 yang dibatasi pada tiga kelas target,
    sebagai pembanding RM3 dengan ruang kelas yang sama.
    """
    names = [n for n in EMOTION_LABELS if n.lower() in TARGET_EMOTIONS]
    records = []

    for seed in SEEDS:
        path = RM1_ROOT / f"seed_{seed}" / "predictions.csv"
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        frame = frame[frame["true_label"].isin(names)]

        # Mode 1: prediksi asli, termasuk ke kelas non-target
        mode1 = f1_score(
            frame["true_label"], frame["predicted_label"],
            labels=names, average="macro", zero_division=0,
        )

        # Mode 2: argmax dibatasi pada tiga kolom probabilitas target
        restricted = frame[[f"prob_{n}" for n in names]].to_numpy()
        pred2 = np.asarray(names)[restricted.argmax(axis=1)]
        mode2 = f1_score(
            frame["true_label"], pred2,
            labels=names, average="macro", zero_division=0,
        )

        records.append(
            {"seed": seed, "n_samples": len(frame),
             "rm1_mode1": round(mode1, 4), "rm1_mode2": round(mode2, 4)}
        )

    return pd.DataFrame(records)


def report_cropping(manifest: pd.DataFrame):
    if "real_frames" not in manifest.columns:
        return

    full = int((manifest["real_frames"] >= 401).sum())
    print(f"\nKlip mencapai batas 401 frame : {full} dari {len(manifest)} "
          f"({full / len(manifest):.1%})")
    print("Klip tersebut berdurasi minimal 4 detik sehingga mengalami "
          "center crop. Dilaporkan sebagai keterbatasan.")


def main():
    print(f"Perangkat aktif : {configure_device()}")

    dataset = FeatureDataset(FEATURES_PATH, INDEX_PATH)
    test_split = pd.read_csv(TEST_SPLIT)
    subset = dataset.build(test_split, "test")

    print(f"\n=== RM3 (cross-lingual, INESCO) ===")
    print(f"Berkas uji : {len(subset.manifest)}")
    print(f"Kelas uji  : {sorted(subset.manifest['emotion'].unique())}")
    print("Tidak ada pelatihan. Model RM1 dipakai apa adanya.")

    report_cropping(subset.manifest)

    rows = []

    for seed in SEEDS:
        model_path = RM1_ROOT / f"seed_{seed}" / "best_model.keras"

        if not model_path.exists():
            print(f"Lewat seed {seed}: model tidak ditemukan.")
            continue

        model = keras.models.load_model(model_path)
        evaluator = RM3Evaluator(model, OUTPUT_ROOT / f"seed_{seed}")
        summary, _ = evaluator.evaluate(subset)

        for record in summary.to_dict("records"):
            record["seed"] = seed
            rows.append(record)

        print(f"\nseed {seed}")
        print(summary.round(4).to_string(index=False))

    result = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_ROOT / "rm3_summary.csv", index=False)

    print("\n=== Rekapitulasi lima seed ===")
    for mode in ("mode_1", "mode_2"):
        g = result[result["mode"] == mode]
        print(f"{mode}: macro F1 rerata {g['macro_f1'].mean():.4f}  "
              f"sd {g['macro_f1'].std(ddof=1):.4f}  "
              f"rentang {g['macro_f1'].min():.4f}-{g['macro_f1'].max():.4f}  "
              f"| prediksi ke luar kelas target "
              f"{g['outside_target_ratio'].mean():.1%}")

    print("\n=== Baseline RM1 pada tiga kelas yang sama ===")
    baseline = rm1_baseline_on_target_classes()
    print(baseline.to_string(index=False))
    baseline.to_csv(OUTPUT_ROOT / "rm1_baseline_target_classes.csv", index=False)

    print(f"\nArtefak tersimpan : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()