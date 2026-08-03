from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ..features.feature_dataset import FeatureSubset
from ..features.constants import EMOTION_LABELS
from .data_adapter import to_model_inputs
from .constants import BATCH_SIZE

TARGET_EMOTIONS = ("angry", "happy", "sad")


class RM3Evaluator:
    """
    Mengevaluasi model hasil RM1 pada korpus lintas-bahasa (INESCO).

    Korpus uji hanya memuat tiga kelas emosi, sedangkan model memiliki
    tujuh unit output. Evaluasi karenanya dilaporkan dalam dua mode:

    Mode 1  Tanpa pembatasan. Model tetap dapat memprediksi tujuh kelas,
            dan prediksi ke kelas di luar tiga kelas target dihitung
            sebagai kesalahan.
    Mode 2  Probabilitas dibatasi pada tiga kelas target sebelum
            pemilihan kelas dengan probabilitas tertinggi.

    Macro F1-score dihitung terhadap tiga kelas target saja. Empat kelas
    yang tidak muncul pada korpus uji memiliki support nol sehingga
    F1-nya selalu nol dan hanya akan menyeret rerata tanpa makna.
    Kesalahan berupa prediksi ke kelas non-target tetap terhitung, yaitu
    lewat penurunan recall pada ketiga kelas target.

    Catatan
    -------
    Kelas ini tidak melatih maupun menyetel model.
    """

    def __init__(
        self,
        model: keras.Model,
        output_dir: Path,
        batch_size: int = BATCH_SIZE,
    ):
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.class_names = list(EMOTION_LABELS)
        self.target_index = self._resolve_target_index()

    def _resolve_target_index(self) -> list[int]:
        lookup = {name.lower(): i for i, name in enumerate(self.class_names)}
        missing = [e for e in TARGET_EMOTIONS if e not in lookup]

        if missing:
            raise ValueError(
                f"Kelas target tidak ditemukan pada ruang label: {missing}"
            )

        return [lookup[e] for e in TARGET_EMOTIONS]

    def evaluate(self, subset: FeatureSubset) -> tuple[pd.DataFrame, pd.DataFrame]:
        features, _ = to_model_inputs(subset)
        probabilities = self.model.predict(
            features, batch_size=self.batch_size, verbose=0
        )

        y_true = np.asarray(subset.labels)

        # Mode 1: argmax atas seluruh tujuh kelas
        pred_mode1 = probabilities.argmax(axis=1)

        # Mode 2: argmax hanya di antara tiga kolom kelas target
        restricted = probabilities[:, self.target_index]
        pred_mode2 = np.asarray(self.target_index)[restricted.argmax(axis=1)]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self._build_predictions(
            subset.manifest, y_true, pred_mode1, pred_mode2, probabilities
        )
        predictions.to_csv(self.output_dir / "predictions.csv", index=False)

        summary = pd.DataFrame(
            [
                self._metrics("mode_1", y_true, pred_mode1),
                self._metrics("mode_2", y_true, pred_mode2),
            ]
        )
        summary.to_csv(self.output_dir / "metrics_summary.csv", index=False)

        per_class = pd.concat(
            [
                self._per_class("mode_1", y_true, pred_mode1),
                self._per_class("mode_2", y_true, pred_mode2),
            ],
            ignore_index=True,
        )
        per_class.to_csv(self.output_dir / "metrics_per_class.csv", index=False)

        return summary, per_class

    def _build_predictions(
        self,
        manifest: pd.DataFrame,
        y_true: np.ndarray,
        pred_mode1: np.ndarray,
        pred_mode2: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        columns = [
            c for c in
            ("row_index", "dataset", "speaker", "filename", "emotion", "real_frames")
            if c in manifest.columns
        ]
        frame = manifest[columns].copy().reset_index(drop=True)

        frame["true_label"] = [self.class_names[i] for i in y_true]
        frame["pred_mode1"] = [self.class_names[i] for i in pred_mode1]
        frame["pred_mode2"] = [self.class_names[i] for i in pred_mode2]
        frame["correct_mode1"] = y_true == pred_mode1
        frame["correct_mode2"] = y_true == pred_mode2
        frame["outside_target"] = ~np.isin(pred_mode1, self.target_index)

        for position, name in enumerate(self.class_names):
            frame[f"prob_{name}"] = probabilities[:, position]

        return frame

    def _metrics(self, mode: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        outside = float(np.mean(~np.isin(y_pred, self.target_index)))
        n_output = len(self.class_names) if mode == "mode_1" else len(self.target_index)

        return {
            "mode": mode,
            "n_samples": int(y_true.size),
            "n_classes_scored": len(self.target_index),
            "chance_accuracy": 1 / n_output,
            "chance_macro_f1": self._chance_macro_f1(y_true, n_output),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred,
                labels=self.target_index, average="macro", zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_true, y_pred,
                labels=self.target_index, average="weighted", zero_division=0,
            ),
            "outside_target_ratio": outside,
        }

    def _chance_macro_f1(self, y_true: np.ndarray, n_output: int) -> float:
        """
        Macro F1-score yang dicapai penebak acak seragam.

        Chance level untuk macro F1-score tidak sama dengan chance level
        untuk accuracy. Penebak acak atas n_output kelas memperoleh recall
        1/n_output pada tiap kelas target, sedangkan precision-nya sama
        dengan proporsi kelas tersebut pada data uji. Nilai inilah
        pembanding yang sah bagi macro F1-score yang dilaporkan.
        """
        recall = 1 / n_output
        scores = []

        for index in self.target_index:
            prior = float(np.mean(y_true == index))

            if prior + recall == 0:
                continue

            scores.append(2 * prior * recall / (prior + recall))

        return float(np.mean(scores)) if scores else 0.0

    def _per_class(
        self, mode: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        names = [self.class_names[i] for i in self.target_index]
        report = classification_report(
            y_true, y_pred,
            labels=self.target_index, target_names=names,
            output_dict=True, zero_division=0,
        )

        return pd.DataFrame(
            [
                {
                    "mode": mode,
                    "emotion": name,
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1_score": report[name]["f1-score"],
                    "support": int(report[name]["support"]),
                }
                for name in names
            ]
        )