from __future__ import annotations
from .constants import TRAINING_DATASETS
import pandas as pd

class DurationEvaluator:
    """
    Evaluate the impact of a target duration.

    This class estimates how many samples require
    padding or cropping for a given target duration.

    Notes
    -----
    This class does not:
    - modify audio
    - save files
    - perform duration normalization
    """
    def __init__(self, metadata: pd.DataFrame, target_duration: float):
        self.metadata = metadata.copy()
        self.target_duration = target_duration

    def _filter_training_datasets(self) -> pd.DataFrame:
        return self.metadata[
            self.metadata["dataset"].isin(TRAINING_DATASETS)
        ].copy()
    
    def _calculate_adjustment(self, duration: float) -> dict:
        if duration < self.target_duration:
            return {
                "padding": self.target_duration - duration,
                "cropping": 0.0,
                "status": "padding",
            }
        
        if duration > self.target_duration:
            return {
                "padding": 0.0,
                "cropping": duration - self.target_duration,
                "status": "cropping",
            }
        
        return {
            "padding": 0.0,
            "cropping": 0.0,
            "status": "unchanged",
        }
    
    def _build_dataset_summary(self, dataset: str, metadata: pd.DataFrame) -> dict:
        durations = metadata["duration"]
        adjustments = [
            self._calculate_adjustment(duration)
            for duration in durations
        ]

        total_files = len(adjustments)
        
        padding_files = sum(
            item["status"] == "padding"
            for item in adjustments
        )
        cropping_files = sum(
            item["status"] == "cropping"
            for item in adjustments
        )
        unchanged_files = sum(
            item["status"] == "unchanged" 
            for item in adjustments
        )

        mean_padding = (sum(item["padding"] for item in adjustments)/total_files)
        mean_cropping = (sum(item["cropping"] for item in adjustments)/total_files)

        padding_pct = (padding_files/total_files)*100
        cropping_pct = (cropping_files/total_files)*100
        unchanged_pct = (unchanged_files/total_files)*100
        mean_padding_pct = (mean_padding/self.target_duration)*100
        mean_cropping_pct = (mean_cropping/self.target_duration)*100
        
        return {
            "dataset": dataset,
            "total_files": total_files,
            "padding_files": padding_files,
            "padding_pct": round(padding_pct,2),
            "cropping_files": cropping_files,
            "cropping_pct": round(cropping_pct,2),
            "unchanged_files": unchanged_files,
            "unchanged_pct": round(unchanged_pct,2),
            "mean_padding": round(mean_padding,3),
            "mean_padding_pct": round(mean_padding_pct,2),
            "mean_cropping": round(mean_cropping,3),
            "mean_cropping_pct": round(mean_cropping_pct,2),
        }
    
    def _build_combined_summary(self, metadata: pd.DataFrame) -> dict:
        return self._build_dataset_summary(dataset="Combined", metadata=metadata)
    
    def evaluate(self) -> pd.DataFrame:
        metadata = self._filter_training_datasets()
        
        rows = []
        for dataset in TRAINING_DATASETS:
            group = metadata[metadata["dataset"]==dataset]
            rows.append(
                self._build_dataset_summary(dataset=dataset, metadata=group,)
            )

        rows.append(self._build_combined_summary(metadata))
        summary = pd.DataFrame(rows)
        
        columns = [
            "dataset",
            "total_files",
            "padding_files",
            "padding_pct",
            "cropping_files",
            "cropping_pct",
            "unchanged_files",
            "unchanged_pct",
            "mean_padding",
            "mean_cropping",
        ]
        
        return summary[columns]