from pathlib import Path
import pandas as pd
from ser.preprocessing.dataset_preprocessor import DatasetPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT/"data"/"metadata"/"file_inventory.csv"
RAW_ROOT = PROJECT_ROOT/"data"/"raw"
OUTPUT_ROOT = PROJECT_ROOT/"data"/"processed"/"audio"
PROCESSED_METADATA_PATH = (PROJECT_ROOT/"data"/"metadata"/"processed_inventory.csv")

def main():
    metadata = pd.read_csv(METADATA_PATH)

    preprocessor = DatasetPreprocessor(
        metadata=metadata,
        input_root=RAW_ROOT,
        output_root=OUTPUT_ROOT,
        metadata_output=PROCESSED_METADATA_PATH
    )

    print("Start preprocessing...")
    preprocessor.process()
    print("Finished preprocessing.")

if __name__ == "__main__":
    main()