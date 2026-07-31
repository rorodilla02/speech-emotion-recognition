TRAINING_DATASETS = (
    "ravdess",
    "tess",
    "savee",
)

EVALUATION_DATASETS = (
    "inesco",
)

ALL_DATASETS = TRAINING_DATASETS + EVALUATION_DATASETS

TRAINING_COMBINED_LABEL = "training_combined"

TARGET_SAMPLE_RATE = 16_000
TARGET_RMS = 0.1
RMS_TOLERANCE = 0.01
TRIM_TOP_DB = 30

PROCESSED_AUDIO_SUBTYPE = "FLOAT"

# NILAI SEMENTARA.
# Nilai final ditetapkan dari duration_summary_processed.csv
# (basis: durasi setelah silence trimming), bukan dari durasi mentah.
TARGET_DURATION = 4.0
TARGET_LENGTH = int(round(TARGET_DURATION * TARGET_SAMPLE_RATE))

CORRUPT_FILES = (
    ("inesco", "mbaz_h138.wav"),
)

SPEAKER_CORRECTIONS = {
    ("tess", "OA"): "OAF",
}

EXCLUDED_EMOTIONS = ("Calm",)

RANDOM_SEED = 42
RM1_TRAIN_RATIO = 0.70
RM1_VALIDATION_RATIO = 0.15
RM1_TEST_RATIO = 0.15