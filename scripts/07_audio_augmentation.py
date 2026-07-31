from pathlib import Path
import random
import numpy as np
import pandas as pd
from ser.augmentation.dataset_augmentor import DatasetAugmentor
from ser.augmentation.pipeline import AugmentationPipeline
from ser.augmentation.noise_injection import NoiseInjection
from ser.augmentation.pitch_shifting import PitchShifting
from ser.preprocessing.constants import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_inventory.csv"
AUDIO_ROOT = PROJECT_ROOT / "data" / "processed" / "audio"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "augmented"

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata[metadata["dataset"] != "inesco"]

    pipeline = AugmentationPipeline(
        augmentors=[
            PitchShifting(probability=1.0),
            NoiseInjection(probability=1.0)
        ]
    )

    augmentor = DatasetAugmentor(
        metadata=metadata,
        audio_root=AUDIO_ROOT,
        output_root=OUTPUT_ROOT,
        pipeline=pipeline,
    )

    augmentor.run()

if __name__ == "__main__":
    main()