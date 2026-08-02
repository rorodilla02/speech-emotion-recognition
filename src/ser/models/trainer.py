from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import pandas as pd
import keras
from ..features.feature_dataset import FeatureDataset
from .cnn_architecture import build_cnn
from .data_adapter import to_model_inputs
from .training_config import (
    TrainingConfig,
    compile_model,
    build_callbacks,
    set_global_seed,
)
from .constants import MONITOR_METRIC, TRAINING_LOG_FILENAME


@dataclass(slots=True)
class TrainingResult:
    """Hasil satu kali pelatihan pada satu skenario."""

    model: keras.Model
    history: pd.DataFrame
    best_epoch: int
    best_score: float
    epochs_run: int
    seconds_per_epoch: float


class ScenarioTrainer:
    """
    Melatih satu skenario penelitian (RM1, satu fold RM2, atau RM3).

    Konfigurasi training identik untuk seluruh skenario, sehingga
    perbedaan hasil antar skenario hanya berasal dari komposisi data.

    Catatan
    -------
    Kelas ini tidak:
    - membuat atau memodifikasi pembagian data
    - mengevaluasi model pada data uji
    - menghasilkan grafik maupun tabel laporan
    """

    def __init__(
        self,
        dataset: FeatureDataset,
        output_dir: Path,
        config: TrainingConfig | None = None,
    ):
        self.dataset = dataset
        self.output_dir = output_dir
        self.config = config or TrainingConfig()

    def run(
        self,
        train_split: pd.DataFrame,
        validation_split: pd.DataFrame,
    ) -> TrainingResult:
        if validation_split.empty:
            raise ValueError(
                "Data validasi kosong. Seleksi model tidak boleh memakai "
                "data uji karena akan menimbulkan kebocoran seleksi."
            )

        train = self.dataset.build(train_split, "train")
        validation = self.dataset.build(validation_split, "validation")

        x_train, y_train = to_model_inputs(train)
        x_validation, y_validation = to_model_inputs(validation)

        print(f"Data latih   : {x_train.shape[0]} sampel "
              f"({len(train_split)} asli + augmentasi)")
        print(f"Data validasi: {x_validation.shape[0]} sampel")
        self._report_composition(train.manifest, "latih")
        self._report_composition(validation.manifest, "validasi")

        set_global_seed(self.config.random_seed)

        model = compile_model(build_cnn(), self.config)
        callbacks = build_callbacks(self.output_dir, self.config)

        start = time.time()
        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        duration = time.time() - start

        history = pd.read_csv(self.output_dir / TRAINING_LOG_FILENAME)

        result = self._summarize(history, duration)
        result.model = model

        return result

    def _summarize(
        self,
        history: pd.DataFrame,
        duration: float,
    ) -> TrainingResult:
        column = MONITOR_METRIC
        if column not in history.columns:
            raise KeyError(
                f"Kolom '{column}' tidak ada pada training log. "
                f"Kolom tersedia: {list(history.columns)}."
            )
        best_position = int(history[column].idxmax())
        epochs_run = len(history)

        # Epoch pertama memuat waktu tracing dan autotuning kernel,
        # sehingga dikeluarkan dari perhitungan waktu rata-rata.
        divisor = max(epochs_run - 1, 1)

        return TrainingResult(
            model=None,
            history=history,
            best_epoch=int(history.loc[best_position, "epoch"]) + 1,
            best_score=float(history.loc[best_position, column]),
            epochs_run=epochs_run,
            seconds_per_epoch=duration / divisor,
        )

    @staticmethod
    def _report_composition(manifest: pd.DataFrame, label: str):
        counts = manifest["dataset"].value_counts().to_dict()
        print(f"  komposisi korpus data {label}: {counts}")