from pathlib import Path
import pandas as pd
from ser.split.split_validator import SplitValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT/"data"/"metadata"/"processed_inventory.csv"
SPLIT_ROOT = PROJECT_ROOT/"data"/"splits"/"rm3"

def main():
    metadata = pd.read_csv(METADATA_PATH)

    validator = SplitValidator(split_root=SPLIT_ROOT, metadata=metadata)
    summary = validator.validate_rm3()

    print(summary)

if __name__ == "__main__":
    main()