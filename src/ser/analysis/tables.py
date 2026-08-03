from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CORPUS_TEST_FILE = {
    "fold_1": "savee",
    "fold_2": "tess",
    "fold_3": "ravdess",
}


def rm2_restricted_to_rm1_test(
    rm2_root: Path,
    rm1_test_split: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """
    Menghitung ulang macro F1-score RM2 yang dibatasi pada berkas uji RM1.

    Data uji RM2 memakai seluruh korpus target, sedangkan data uji RM1
    hanya sebagian. Pembatasan ini menyamakan himpunan berkas sehingga
    perbandingan antara kedua skenario menjadi setara.
    """
    allowed = set(rm1_test_split["filename"])
    records = []

    for fold in sorted(CORPUS_TEST_FILE):
        for seed_dir in sorted(p for p in (rm2_root / fold).glob("seed_*") if p.is_dir()):
            path = seed_dir / "predictions.csv"

            if not path.exists():
                continue

            frame = pd.read_csv(path)
            subset = frame[frame["filename"].isin(allowed)]

            if subset.empty:
                continue

            records.append(
                {
                    "fold": fold,
                    "seed": int(seed_dir.name.split("_")[1]),
                    "n_penuh": len(frame),
                    "n_dibatasi": len(subset),
                    "macro_f1_penuh": f1_score(
                        frame["true_label"], frame["predicted_label"],
                        labels=labels, average="macro", zero_division=0,
                    ),
                    "macro_f1_dibatasi": f1_score(
                        subset["true_label"], subset["predicted_label"],
                        labels=labels, average="macro", zero_division=0,
                    ),
                }
            )

    return pd.DataFrame(records)


def padding_analysis(predictions: pd.DataFrame, total_frames: int = 401) -> pd.DataFrame:
    """
    Menguji dugaan bahwa proporsi zero padding berkorelasi dengan korpus.

    Kolom real_frames mencatat jumlah frame nyata sebelum padding. Bila
    proporsinya berbeda sistematis antar korpus, wilayah padding berpotensi
    menjadi pintasan yang dipelajari model.
    """
    if "real_frames" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["padding_ratio"] = 1 - frame["real_frames"] / total_frames

    grouped = frame.groupby("dataset").agg(
        n=("padding_ratio", "size"),
        padding_rerata=("padding_ratio", "mean"),
        padding_sd=("padding_ratio", "std"),
        padding_min=("padding_ratio", "min"),
        padding_maks=("padding_ratio", "max"),
    ).reset_index()

    if "correct" in frame.columns:
        korelasi = (
            frame.groupby("dataset")
            .apply(
                lambda g: np.corrcoef(
                    g["padding_ratio"], g["correct"].astype(float)
                )[0, 1] if g["correct"].nunique() > 1 else np.nan,
                include_groups=False,
            )
            .rename("korelasi_padding_benar")
            .reset_index()
        )
        grouped = grouped.merge(korelasi, on="dataset", how="left")

    return grouped


def per_corpus_per_class(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """
    Macro F1-score per kelas dipecah per korpus.

    Diperlukan karena metrik per kelas pada RM1 bersifat gabungan tiga
    korpus sehingga didominasi korpus yang paling mudah.
    """
    records = []

    for dataset in sorted(predictions["dataset"].unique()):
        subset = predictions[predictions["dataset"] == dataset]

        for label in labels:
            mask_true = subset["true_label"] == label

            if not mask_true.any():
                continue

            records.append(
                {
                    "dataset": dataset,
                    "emotion": label,
                    "support": int(mask_true.sum()),
                    "f1_score": f1_score(
                        subset["true_label"], subset["predicted_label"],
                        labels=[label], average="macro", zero_division=0,
                    ),
                }
            )

    return pd.DataFrame(records)