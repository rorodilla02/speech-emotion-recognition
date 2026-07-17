import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from ..datasets.common import (
    DatasetStatistics,
    ParseResult,
    RAVDESS_EMOTION_MAP,
    TESS_EMOTION_MAP,
    SAVEE_EMOTION_MAP,
    INESCO_EMOTION_MAP,
)

class OutputWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _output_path(self, filename: str) -> Path:
        return self.output_dir / filename
    
    def _write_csv(self, rows: list[dict], filename: str):
        df = pd.DataFrame(rows)
        df.to_csv(
            self._output_path(filename),
            index=False,
        )

    def _build_summary(self, result: ParseResult, statistics_list: list[DatasetStatistics],) -> dict:
        successful_files = len(result.metadata)
        failed_files = len(result.failed_files)
        speakers = set()

        for stats in statistics_list:
            speakers.update(stats.speakers)
        
        return {
            "total_datasets": len(statistics_list),
            "total_files": successful_files + failed_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_speakers": len(speakers),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def write_dataset_statistics(self, statistics_list: list[DatasetStatistics]):
        rows = []
        for stats in statistics_list:
            rows.append(
                {
                    "dataset": stats.dataset,
                    "total_files": stats.total_files,
                    "total_speakers": stats.total_speakers,
                    "min_duration": stats.min_duration,
                    "max_duration": stats.max_duration,
                    "mean_duration": stats.mean_duration
                }
            )

        self._write_csv(rows, "dataset_statistics.csv",)

    def write_failed_files(self, result: ParseResult,):
        rows = []
        for failed in result.failed_files:
            rows.append(
                {
                    "dataset": failed.dataset,
                    "filename": failed.filename,
                    "filepath": str(failed.filepath),
                    "error_type": failed.error_type.value,
                    "error_message": failed.error_message,
                }
            )

        self._write_csv(rows, "failed_files.csv",)

    def write_file_inventory(self, result: ParseResult,):
        rows = []
        for metadata in result.metadata:
            rows.append(
                {
                    "dataset": metadata.dataset,
                    "filename": metadata.filename,
                    "filepath": str(metadata.filepath),
                    "speaker": metadata.speaker,
                    "raw_label": metadata.raw_label,
                    "emotion": metadata.emotion,
                    "sample_rate": metadata.sample_rate,
                    "duration": metadata.duration,
                }
            )

        rows.sort(
            key=lambda row: (
                row["dataset"],
                row["speaker"],
                row["filename"],
            )
        )
        self._write_csv(rows, "file_inventory.csv",)

    def write_emotion_distribution(self, statistics_list: list[DatasetStatistics],):
        rows = []
        for stats in statistics_list:
            for emotion, total in stats.emotion_distribution.items():
                rows.append(
                    {
                        "dataset": stats.dataset,
                        "emotion": emotion,
                        "total": total,
                    }
                )

        rows.sort(
            key=lambda row:(
                row["dataset"],
                row["emotion"],
            )
        )

        self._write_csv(rows, "emotion_distribution.csv",)

    def write_label_mapping(self):
        emotion_maps = {
            "ravdess": RAVDESS_EMOTION_MAP,
            "tess": TESS_EMOTION_MAP,
            "savee": SAVEE_EMOTION_MAP,
            "inesco": INESCO_EMOTION_MAP,
        }

        rows = []
        for dataset, mapping in emotion_maps.items():
            for raw_label, emotion in mapping.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "raw_label": raw_label,
                        "emotion": emotion,
                    }
                )

        rows.sort(
            key=lambda row:(
                row["dataset"],
                row["raw_label"],
            )
        )

        self._write_csv(rows, "label_mapping.csv",)

    def write_summary(self, result: ParseResult, statistics_list: list[DatasetStatistics]):
        summary = self._build_summary(result, statistics_list,)

        rows = [
            {
                "metric": key,
                "value": value,
            }
            for key, value in summary.items()
        ]
        
        self._write_csv(rows, "audit_summary.csv",)

    def write_summary_json(self, result: ParseResult, statistics_list: list[DatasetStatistics],):
        summary = self._build_summary(result, statistics_list,)

        output_path = self._output_path("audit_summary.json")

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

    def write_report(self, result: ParseResult, statistics_list: list[DatasetStatistics]):
        summary = self._build_summary(result, statistics_list)
        
        lines: list[str] = []
        lines.append("# Dataset Audit Report")
        lines.append("")
        lines.append(f"Generated at: {summary['generated_at']}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total datasets: {summary['total_datasets']}")
        lines.append(f"- Total files: {summary['total_files']}")
        lines.append(f"- Successful files: {summary['successful_files']}")
        lines.append(f"- Failed files: {summary['failed_files']}")
        lines.append(f"- Total speakers: {summary['total_speakers']}")
        lines.append("")

        lines.append("## Dataset Statistics")
        lines.append("")
        lines.append("| Dataset | Files | Speakers | Sample Rates | Min | Mean | Max |")
        lines.append("|---------|-------|----------|--------------|-----|------|-----|")

        for stats in statistics_list:
            sample_rates = ", ".join(
                map(str, stats.sample_rates)
            )

            lines.append(
                f"| {stats.dataset} |"
                f"| {stats.total_files} |"
                f"| {stats.total_speakers} |"
                f"| {sample_rates} |"
                f"| {stats.min_duration:.2f} |"
                f"| {stats.mean_duration:.2f} |"
                f"| {stats.max_duration:.2f} |"
            )

        lines.append("## Emotion Distribution")
        lines.append("")
        for stats in statistics_list:
            lines.append(f"### {stats.dataset}")
            lines.append("")
            lines.append("| Emotion | Total |")
            lines.append("|---------|-------|")

            for emotion, total in stats.emotion_distribution.items():
                lines.append(f"| {emotion} | {total} |")
            
            lines.append("")
        
        lines.append("## Failed Files")
        lines.append("")
        if not result.failed_files:
            lines.append("No failed files.")
        else:
            lines.append("| Dataset | Filename | Error |")
            lines.append("|---------|----------|-------|")

            for failed in result.failed_files:
                lines.append(f"| {failed.dataset} | {failed.filename} | {failed.error_message} |")

        report = "\n".join(lines)
        output_path = self._output_path("audit_report.md")
        with output_path.open("w", encoding="utf-8") as file:
            file.write(report)

    def write_all(self, result: ParseResult, statistics_list: list[DatasetStatistics]):
        self.write_summary(result, statistics_list)
        self.write_summary_json(result, statistics_list)
        self.write_dataset_statistics(statistics_list)
        self.write_file_inventory(result)
        self.write_failed_files(result)
        self.write_emotion_distribution(statistics_list)
        self.write_label_mapping()
        self.write_report(result, statistics_list)