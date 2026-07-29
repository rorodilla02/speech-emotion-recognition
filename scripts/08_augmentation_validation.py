from pathlib import Path
import pandas as pd
import soundfile as sf

from ser.augmentation.augmentation_validator import AugmentationValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = (PROJECT_ROOT/"data"/"augmented"/"augmented_inventory.csv")
AUGMENTED_ROOT = (PROJECT_ROOT/"data"/"augmented")


def main():

    metadata = pd.read_csv(METADATA_PATH)

    validator = AugmentationValidator(
        metadata=metadata,
        augmented_root=AUGMENTED_ROOT,
    )

    summary = validator.validate()

    print(summary)

if __name__ == "__main__":
    main()