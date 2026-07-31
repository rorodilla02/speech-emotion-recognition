from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from ser.preprocessing.duration_analyzer import DurationAnalyzer
from ser.preprocessing.duration_visualizer import DurationVisualizer
from ser.preprocessing.duration_evaluator import DurationEvaluator
from ser.preprocessing.constants import TARGET_DURATION
from ser.output.writer import OutputWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

STAGES = {
    "raw": {
        "metadata": "file_inventory.csv",
        "summary": "duration_summary_raw.csv",
        "evaluation": None,          # tidak relevan pada durasi mentah
        "boxplot": "duration_boxplot_raw.png",
        "histogram": "duration_histogram_raw.png",
    },
    "processed": {
        "metadata": "processed_inventory.csv",
        "summary": "duration_summary_processed.csv",
        "evaluation": "duration_evaluation_processed.csv",
        "boxplot": "duration_boxplot_processed.png",
        "histogram": "duration_histogram_processed.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisis durasi audio.")
    parser.add_argument(
        "stage",
        choices=sorted(STAGES.keys()),
        help="Basis metadata yang dianalisis.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=TARGET_DURATION,
        help="Target durasi (detik) yang diuji pada DurationEvaluator.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = STAGES[args.stage]

    metadata_path = METADATA_DIR / config["metadata"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {metadata_path}. "
            "Jalankan tahap sebelumnya terlebih dahulu."
        )

    print(f"Stage    : {args.stage}")
    print(f"Metadata : {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    writer = OutputWriter(METADATA_DIR)

    # 1. Statistik deskriptif durasi
    analyzer = DurationAnalyzer(metadata)
    summary = analyzer.analyze()
    summary_path = writer.write_duration_summary(summary, config["summary"])

    print("\n=== Duration Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    # 2. Visualisasi
    visualizer = DurationVisualizer(metadata)
    boxplot_path = visualizer.plot_boxplot(FIGURE_DIR / config["boxplot"])
    histogram_path = visualizer.plot_histogram(
        summary, FIGURE_DIR / config["histogram"]
    )

    print(f"Saved: {boxplot_path}")
    print(f"Saved: {histogram_path}")

    # 3. Evaluasi dampak target durasi (hanya pada basis processed)
    if config["evaluation"] is None:
        print(
            "\nCatatan: DurationEvaluator dilewati pada basis 'raw' karena "
            "keputusan target durasi harus berbasis durasi setelah trimming."
        )
        return

    evaluator = DurationEvaluator(metadata, args.target_duration)
    evaluation = evaluator.evaluate()
    evaluation_path = writer.write_duration_evaluation(
        evaluation, config["evaluation"]
    )

    print(f"\n=== Duration Evaluation (target = {args.target_duration} s) ===")
    print(evaluation.to_string(index=False))
    print(f"\nSaved: {evaluation_path}")


if __name__ == "__main__":
    main()