from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from ..features.feature_dataset import FeatureSubset
from ..features.constants import EMOTION_LABELS
from .data_adapter import to_model_inputs
from .constants import BATCH_SIZE

MANIFEST_COLUMNS = [
    "row_index",
    "dataset",
    "speaker",
    "filename",
    "emotion",
    "real_frames",
]


class ScenarioEvaluator:
    """
    Mengevaluasi model terlatih pada satu subset data uji.

    Keluaran utama berupa berkas prediksi per berkas audio, sehingga
    seluruh metrik, confusion matrix, dan analisis lanjutan pada Bab 4
    dapat dihitung ulang tanpa melatih model kembali.

    Catatan
    -------
    Kelas ini tidak:
    - melatih maupun menyetel model
    - menghasilkan grafik
    """

    def __init__(
        self,
        model: keras.Model,
        output_dir: Path,
        class_names: tuple[str, ...] = EMOTION_LABELS,
        batch_size: int = BATCH_SIZE,
    ):
        self.model = model
        self.output_dir = output_dir
        self.class_names = list(class_names)
        self.batch_size = batch_size

    def evaluate(self, subset: FeatureSubset) -> pd.DataFrame:
        features, _ = to_model_inputs(subset)

        probabilities = self.model.predict(
            features, batch_size=self.batch_size, verbose=0
        )

        y_true = np.asarray(subset.labels)
        y_pred = probabilities.argmax(axis=1)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self._build_predictions(
            subset.manifest, y_true, y_pred, probabilities
        )
        predictions.to_csv(self.output_dir / "predictions.csv", index=False)

        per_class = self._per_class_metrics(y_true, y_pred)
        per_class.to_csv(self.output_dir / "metrics_per_class.csv", index=False)

        summary = self._summary_metrics(predictions, y_true, y_pred)
        summary.to_csv(self.output_dir / "metrics_summary.csv", index=False)

        return summary

    def _build_predictions(
        self,
        manifest: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        available = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
        predictions = manifest[available].copy().reset_index(drop=True)

        predictions["true_label"] = [self.class_names[i] for i in y_true]
        predictions["predicted_label"] = [self.class_names[i] for i in y_pred]
        predictions["correct"] = y_true == y_pred
        predictions["confidence"] = probabilities.max(axis=1)

        for position, name in enumerate(self.class_names):
            predictions[f"prob_{name}"] = probabilities[:, position]

        return predictions

    def _per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        report = classification_report(
            y_true,
            y_pred,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        records = [
            {
                "emotion": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in self.class_names
        ]

        return pd.DataFrame(records)

    def _summary_metrics(
        self,
        predictions: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        records = [self._scope_metrics("keseluruhan", y_true, y_pred)]

        # Pelaporan terpisah per korpus merupakan mitigasi risiko R-02.
        for dataset in sorted(predictions["dataset"].unique()):
            mask = (predictions["dataset"] == dataset).to_numpy()
            records.append(
                self._scope_metrics(dataset, y_true[mask], y_pred[mask])
            )

        corpus_rows = [r for r in records if r["scope"] != "keseluruhan"]

        if len(corpus_rows) > 1:
            # Rata-rata tak berbobot antar korpus, agar angka tidak
            # didominasi korpus dengan jumlah sampel uji terbanyak.
            records.append(
                {
                    "scope": "rata-rata antar korpus",
                    "n_samples": sum(r["n_samples"] for r in corpus_rows),
                    "n_classes_present": max(
                        r["n_classes_present"] for r in corpus_rows
                    ),
                    "chance_level": sum(
                        r["chance_level"] for r in corpus_rows
                    ) / len(corpus_rows),
                    "accuracy": sum(
                        r["accuracy"] for r in corpus_rows
                    ) / len(corpus_rows),
                    "macro_f1": sum(
                        r["macro_f1"] for r in corpus_rows
                    ) / len(corpus_rows),
                    "weighted_f1": sum(
                        r["weighted_f1"] for r in corpus_rows
                    ) / len(corpus_rows),
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def _scope_metrics(
        scope: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        n_present = len(np.unique(y_true))

        return {
            "scope": scope,
            "n_samples": int(y_true.size),
            "n_classes_present": n_present,
            "chance_level": 1 / n_present if n_present else 0.0,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }