from pathlib import Path
import json
import time
import pandas as pd
import tensorflow as tf

from ser.features.feature_dataset import FeatureDataset
from ser.models.cnn_architecture import build_cnn
from ser.models.data_adapter import to_model_inputs
from ser.models.training_config import(
    TrainingConfig,
    set_global_seed,
    configure_device,
    compile_model,
    build_callbacks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "features.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "features" / "feature_index.csv"
RM1_SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "rm1"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "models" / "smoke_test"
CONFIG_PATH = PROJECT_ROOT / "data" / "models" / "training_config.json"

SMOKE_TRAIN_SIZE = 512
SMOKE_VALIDATION_SIZE = 256
SMOKE_EPOCHS = 3

def load_smoke_subsets():
    dataset = FeatureDataset(FEATURE_PATH, INDEX_PATH)

    train = dataset.build(pd.read_csv(RM1_SPLIT_ROOT / "train.csv"), "train")
    validation = dataset.build(pd.read_csv(RM1_SPLIT_ROOT / "validation.csv"), "validation")
    print(f"RM1 train      : {len(train.manifest)} baris")
    print(f"RM1 validation : {len(validation.manifest)} baris")

    x_train, y_train = to_model_inputs(train)
    x_validation, y_validation = to_model_inputs(validation)

    return(
        x_train[:SMOKE_TRAIN_SIZE],
        y_train[:SMOKE_TRAIN_SIZE],
        x_validation[:SMOKE_VALIDATION_SIZE],
        y_validation[:SMOKE_VALIDATION_SIZE],
    )

def report_gpu_memory():
    if not tf.config.list_physical_devices("GPU"):
        return

    info = tf.config.experimental.get_memory_info("GPU:0")
    peak_mb = info["peak"]/1024**2
    print(f"Puncak pemakaian VRAM: {peak_mb:.0f} MB dari 4096 MB")

    if peak_mb > 3000:
        print("PERINGATAN: Pemakaian VRAM mendekati batas." 
        "Turunkan BATCH_SIZE pada src/ser/models/constants.py")

def main():
    config = TrainingConfig()

    set_global_seed(config.random_seed)
    device = configure_device()

    print(f"Perangkat aktif : {device}")

    if device == "CPU":
        print("PERINGATAN: GPU tidak terdeteksi. Training akan berjalan di CPU "
              "dan tidak sejalan dengan asumsi risiko R-01 pada Bab 3.")

    print("\nMemuat data uji konfigurasi...")
    x_train, y_train, x_validation, y_validation = load_smoke_subsets()

    print(f"Bentuk x_train : {x_train.shape}")
    print(f"Bentuk y_train : {y_train.shape}")

    print("\nMembangun dan mengompilasi model...")
    model = compile_model(build_cnn(), config)
    callbacks = build_callbacks(OUTPUT_ROOT, config)

    print(f"\nMenjalankan {SMOKE_EPOCHS} epoch uji "
          f"(batch {config.batch_size})...")

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=SMOKE_EPOCHS,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - start

    steps = -(-len(x_train) // config.batch_size)
    per_step = duration / (SMOKE_EPOCHS * steps)

    print("\n=== Ringkasan ===")
    print(f"Metrik tercatat : {sorted(history.history.keys())}")
    print(f"Waktu per step  : {per_step:.2f} detik")
    print(f"Estimasi 1 epoch RM1 penuh : "
          f"{per_step * (len(x_train) and 1) * 200 / 60:.1f} menit "
          f"(asumsi ±200 step)")

    report_gpu_memory()

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8"
    )

    print(f"\nKonfigurasi disimpan : {CONFIG_PATH}")
    print(f"Artefak uji          : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()