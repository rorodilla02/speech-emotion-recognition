from ..preprocessing.constants import(
    TARGET_SAMPLE_RATE,
    TARGET_LENGTH,
)

# Parameter Framing
# Window 25 ms dan hop 10 ms merupakan konfigurasi baku pemrosesan wicara pada sample rate 16 kHz

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10

WIN_LENGTH = int(TARGET_SAMPLE_RATE * FRAME_LENGTH_MS / 1000)  
HOP_LENGTH = int(TARGET_SAMPLE_RATE * HOP_LENGTH_MS / 1000)   
N_FFT = 512

# Parameter Fitur
N_MFCC = 13
N_MELS = 40
N_CHROMA = 12
PREEMPHASIS_COEF = 0.97

DELTA_WIDTH = 5
CHROMA_TUNING = 0.0


# Dimensi Keluaran
N_FEATURES = N_MFCC * 3 + N_CHROMA
N_FRAMES = 1 + TARGET_LENGTH // HOP_LENGTH
FEATURE_SHAPE = (N_FEATURES, N_FRAMES)

# Ruang Label
EMOTION_LABELS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
)

CROSS_LINGUAL_LABELS = ("Angry", "Happy", "Sad")
LABEL_TO_INDEX = {label: index for index, label in enumerate(EMOTION_LABELS)}

# Sumber Data
SOURCE_PROCESSED = "processed"
SOURCE_AUGMENTED = "augmented"