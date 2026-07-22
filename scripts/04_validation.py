from pathlib import Path
import pandas as pd
from ser.validation.preprocessing_validator import PreprocessingValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT/"data"/"metadata"/"file_inventory.csv"
PROCESSED_ROOT = PROJECT_ROOT/"data"/"processed"/"audio"
OUTPUT_DIR = PROJECT_ROOT/"data"/"metadata"

def main():
    print("Starting preprocessing validation..")

    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata[metadata["filename"] != "mbaz_h138.wav"].copy()
    validator = PreprocessingValidator(
        metadata=metadata,
        processed_root=PROCESSED_ROOT,
    )
    summary = validator.validate()
    output_path = OUTPUT_DIR/"preprocessing_validation.csv"

    summary.to_csv(output_path, index=False)

    print(summary)
    print(f"Validation report saved to: \n{output_path}")
    print("Preprocessing validation completed successfully!")

if __name__ == "__main__":
    main()