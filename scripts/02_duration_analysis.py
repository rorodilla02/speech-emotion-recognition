from pathlib import Path
from ser.preprocessing.duration_analyzer import DurationAnalyzer
from ser.preprocessing.duration_visualizer import DurationVisualizer
from ser.preprocessing.duration_evaluator import DurationEvaluator
from ser.output.writer import OutputWriter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT/"data"/"metadata"/"file_inventory.csv"
METADATA_DIR = PROJECT_ROOT/"data"/"metadata"
TARGET_DURATION = 4

def main():
    metadata = pd.read_csv(METADATA_PATH)
    
    analyzer = DurationAnalyzer(metadata)
    summary = analyzer.analyze()
    writer = OutputWriter(METADATA_DIR)
    output_path = writer.write_duration_summary(summary)
    print(summary)

    visualizer = DurationVisualizer(metadata)
    visualizer.plot_distribution(summary)

    evaluator = DurationEvaluator(metadata, TARGET_DURATION)
    evaluation = evaluator.evaluate()
    evaluation_output = writer.write_duration_evaluation(evaluation)
    print(evaluation)

    print(f"Duration summary saved to: {output_path}")
    print(f"Duration evaluation saved to: {evaluation_output}")

if __name__ == "__main__":
    main()