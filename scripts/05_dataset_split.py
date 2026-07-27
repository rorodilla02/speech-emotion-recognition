from pathlib import Path
import pandas as pd
from ser.split.split_generator import SplitGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT/"data"/"metadata"/"processed_inventory.csv"
SPLIT_ROOT = PROJECT_ROOT/"data"/"splits"

def main():
    metadata = pd.read_csv(METADATA_PATH)
    generator = SplitGenerator(metadata=metadata, output_root=SPLIT_ROOT)

    print("RM1:")
    generator.generate_rm1()

    print("\nRM2:")
    generator.generate_rm2()

    print("\nRM3:")
    generator.generate_rm3()


if __name__ == "__main__":
    main()