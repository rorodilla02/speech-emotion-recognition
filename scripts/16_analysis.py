from pathlib import Path
import pandas as pd

from ser.analysis.figures import (
    plot_confusion_matrix,
    plot_learning_curve,
    emotion_labels,
)
from ser.analysis.tables import (
    rm2_restricted_to_rm1_test,
    padding_analysis,
    per_corpus_per_class,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "data" / "models"
FIGURES_ROOT = PROJECT_ROOT / "reports" / "figures"
RM1_TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "rm1" / "test.csv"

RECORD_SEED = 42
RM2_FOLDS = {"fold_1": "SAVEE", "fold_2": "TESS", "fold_3": "RAVDESS"}
RM3_TARGET = ["Angry", "Happy", "Sad"]


def build_rm1_figures(labels: list[str]):
    base = MODELS_ROOT / "rm1" / f"seed_{RECORD_SEED}"
    predictions = pd.read_csv(base / "predictions.csv")

    plot_confusion_matrix(
        predictions["true_label"], predictions["predicted_label"], labels,
        "RM1 within-corpus, seluruh korpus uji",
        FIGURES_ROOT / "cm_rm1_gabungan.png",
    )

    for dataset in sorted(predictions["dataset"].unique()):
        subset = predictions[predictions["dataset"] == dataset]
        plot_confusion_matrix(
            subset["true_label"], subset["predicted_label"], labels,
            f"RM1 within-corpus, {dataset.upper()}",
            FIGURES_ROOT / f"cm_rm1_{dataset}.png",
        )

    plot_learning_curve(
        pd.read_csv(base / "training_log.csv"),
        "RM1 within-corpus",
        FIGURES_ROOT / "lc_rm1.png",
    )

    return predictions


def build_rm2_figures(labels: list[str]):
    for fold, corpus in RM2_FOLDS.items():
        base = MODELS_ROOT / "rm2" / fold / f"seed_{RECORD_SEED}"

        if not (base / "predictions.csv").exists():
            continue

        predictions = pd.read_csv(base / "predictions.csv")

        plot_confusion_matrix(
            predictions["true_label"], predictions["predicted_label"], labels,
            f"RM2 {fold}, uji {corpus}",
            FIGURES_ROOT / f"cm_rm2_{fold}.png",
        )

        plot_learning_curve(
            pd.read_csv(base / "training_log.csv"),
            f"RM2 {fold}, uji {corpus}",
            FIGURES_ROOT / f"lc_rm2_{fold}.png",
        )


def build_rm3_figures(labels: list[str]):
    base = MODELS_ROOT / "rm3" / f"seed_{RECORD_SEED}"

    if not (base / "predictions.csv").exists():
        return

    predictions = pd.read_csv(base / "predictions.csv")

    # Mode 1 memakai ruang tujuh kelas karena model dapat memprediksi
    # kelas di luar korpus uji.
    plot_confusion_matrix(
        predictions["true_label"], predictions["pred_mode1"], labels,
        "RM3 cross-lingual, mode 1 (tujuh kelas)",
        FIGURES_ROOT / "cm_rm3_mode1.png",
    )

    plot_confusion_matrix(
        predictions["true_label"], predictions["pred_mode2"], RM3_TARGET,
        "RM3 cross-lingual, mode 2 (tiga kelas)",
        FIGURES_ROOT / "cm_rm3_mode2.png",
    )


def build_summary_table() -> pd.DataFrame:
    """Menggabungkan seluruh metrik skenario menjadi satu tabel."""
    records = []

    rm1 = pd.read_csv(MODELS_ROOT / "rm1" / "seed_summary.csv")
    for _, row in rm1.iterrows():
        records.append(
            {
                "skenario": "RM1",
                "cakupan": row["scope"],
                "rerata": row["mean"],
                "sd": row["std"],
                "min": row["min"],
                "maks": row["max"],
                "n_seed": 5,
            }
        )

    rm2 = pd.read_csv(MODELS_ROOT / "rm2" / "rm2_summary.csv")
    for fold, corpus in RM2_FOLDS.items():
        group = rm2[rm2["fold"] == fold]["macro_f1"]

        if group.empty:
            continue

        records.append(
            {
                "skenario": f"RM2 {fold}",
                "cakupan": corpus.lower(),
                "rerata": group.mean(),
                "sd": group.std(ddof=1),
                "min": group.min(),
                "maks": group.max(),
                "n_seed": len(group),
            }
        )

    rm3_path = MODELS_ROOT / "rm3" / "rm3_summary.csv"

    if rm3_path.exists():
        rm3 = pd.read_csv(rm3_path)
        for mode in ("mode_1", "mode_2"):
            group = rm3[rm3["mode"] == mode]["macro_f1"]
            records.append(
                {
                    "skenario": f"RM3 {mode}",
                    "cakupan": "inesco",
                    "rerata": group.mean(),
                    "sd": group.std(ddof=1),
                    "min": group.min(),
                    "maks": group.max(),
                    "n_seed": len(group),
                }
            )

    return pd.DataFrame(records).round(4)


def main():
    labels = emotion_labels()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    print("Membuat gambar RM1...")
    rm1_predictions = build_rm1_figures(labels)

    print("Membuat gambar RM2...")
    build_rm2_figures(labels)

    print("Membuat gambar RM3...")
    build_rm3_figures(labels)

    print("\n=== Tabel metrik gabungan ===")
    summary = build_summary_table()
    summary.to_csv(MODELS_ROOT / "summary_all_scenarios.csv", index=False)
    print(summary.to_string(index=False))

    print("\n=== RM1 per kelas per korpus ===")
    per_class = per_corpus_per_class(rm1_predictions, labels)
    per_class.round(4).to_csv(
        MODELS_ROOT / "rm1" / "per_corpus_per_class.csv", index=False
    )
    print(per_class.pivot(index="emotion", columns="dataset",
                          values="f1_score").round(4).to_string())

    print("\n=== Analisis proporsi padding (data uji RM1) ===")
    padding = padding_analysis(rm1_predictions)
    if not padding.empty:
        padding.round(4).to_csv(
            MODELS_ROOT / "padding_analysis.csv", index=False
        )
        print(padding.round(4).to_string(index=False))

    print("\n=== RM2 dibatasi pada berkas uji RM1 ===")
    restricted = rm2_restricted_to_rm1_test(
        MODELS_ROOT / "rm2", pd.read_csv(RM1_TEST_SPLIT), labels
    )
    if not restricted.empty:
        restricted.round(4).to_csv(
            MODELS_ROOT / "rm2" / "rm2_restricted.csv", index=False
        )
        print(restricted.round(4).to_string(index=False))

    print(f"\nGambar tersimpan  : {FIGURES_ROOT}")
    print(f"Tabel tersimpan   : {MODELS_ROOT}")


if __name__ == "__main__":
    main()