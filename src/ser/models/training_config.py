from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import keras
import tensorflow as tf
from ..preprocessing.constants import RANDOM_SEED
from .constants import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCH,
    METRIC_NAME,
    MONITOR_METRIC,
    MONITOR_MODE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    MIN_LEARNING_RATE,
    CHECKPOINT_FILENAME,
    TRAINING_LOG_FILENAME,
)


@dataclass(slots=True)
class TrainingConfig:
    """
    Konfigurasi training yang berlaku sama untuk seluruh skenario
    (RM1, tiga fold RM2, dan RM3), agar perbedaan hasil antar skenario
    murni berasal dari komposisi data, bukan dari perbedaan konfigurasi.
    """

    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCH
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    reduce_lr_factor: float = REDUCE_LR_FACTOR
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    min_learning_rate: float = MIN_LEARNING_RATE
    random_seed: int = RANDOM_SEED

    def as_dict(self) -> dict:
        return asdict(self)


def set_global_seed(seed: int = RANDOM_SEED):
    """Menyeragamkan seed Python, NumPy, dan backend Keras."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def configure_device() -> str:
    """
    Mengaktifkan alokasi memori bertahap pada GPU agar TensorFlow tidak
    langsung mengambil seluruh VRAM, lalu melaporkan perangkat aktif.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        return "CPU"

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return f"GPU ({len(gpus)} perangkat)"


def compile_model(
    model: keras.Model,
    config: TrainingConfig | None = None,
) -> keras.Model:
    """
    Mengompilasi model dengan categorical cross-entropy dan optimizer Adam.

    Metrik yang dipantau adalah accuracy dan macro F1-score. Macro F1-score
    dipakai sebagai metrik utama sesuai mitigasi risiko R-02, karena proporsi
    korpus dan kelas pada data latih tidak seimbang.
    """
    config = config or TrainingConfig()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.F1Score(average="macro", name=METRIC_NAME),
        ],
    )

    return model

class MonitorGuard(keras.callbacks.Callback):
    """
    Menghentikan training bila metrik yang dipantau callback lain tidak
    tersedia pada log epoch pertama.

    Keras hanya memberi peringatan ketika nama monitor tidak dikenal,
    sehingga early stopping dan model checkpoint dapat gagal berfungsi
    tanpa disadari sampai seluruh epoch selesai dijalankan.
    """

    def __init__(self, monitor: str = MONITOR_METRIC):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        if epoch > 0:
            return

        logs = logs or {}

        if self.monitor not in logs:
            raise KeyError(
                f"Metrik pantauan '{self.monitor}' tidak tersedia. "
                f"Metrik yang ada: {sorted(logs.keys())}. "
                "Training dihentikan agar early stopping dan model "
                "checkpoint tidak berjalan tanpa efek."
            )

def build_callbacks(
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> list[keras.callbacks.Callback]:
    """
    Menyusun callback training.

    Seluruh callback memantau metrik validasi. Skenario tanpa data
    validasi tidak diperbolehkan, karena seleksi model yang memakai
    data uji akan menimbulkan kebocoran seleksi pada RM2 dan RM3.
    """
    config = config or TrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        MonitorGuard(MONITOR_METRIC),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / CHECKPOINT_FILENAME,
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(
            filename=output_dir / TRAINING_LOG_FILENAME,
        ),
    ]