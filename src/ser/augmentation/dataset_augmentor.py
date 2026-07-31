from __future__ import annotations
from pathlib import Path
import pandas as pd
from .pipeline import AugmentationPipeline
from ..preprocessing.audio_loader import AudioLoader
from .audio_writer import AudioWriter

class DatasetAugmentor:
    """
    Applies audio augmentation to a training dataset.
    """

    def __init__(
            self,
            metadata: pd.DataFrame,
            audio_root: Path,
            output_root: Path,
            pipeline: AugmentationPipeline,
        ):
        self.metadata = metadata.copy()
        self.audio_root = audio_root
        self.output_root = output_root
        self.pipeline = pipeline
        self.audio_loader = AudioLoader()
        self.audio_writer = AudioWriter()
        self.records: list[dict] = []

    def run(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        total = len(self.metadata)

        print(f"Output directory: {self.output_root}")
        print(f"Files to augment: {total}")

        for position, (_, row) in enumerate(self.metadata.iterrows(), start=1):
            audio_path = self.audio_root / Path(row["filepath"])
            audio = self.audio_loader.load(audio_path)
            augmented_audio = self.pipeline.apply(audio)
            output_path = self.output_root / Path(row["filepath"])

            self.audio_writer.write(
                audio=augmented_audio,
                outputh_path=output_path,
            )

            self.records.append(
                self._build_record(row=row, output_path=output_path)
            )

            if position % 500 == 0 or position == total:
                print(f"Augmented {position}/{total}")

        # Ditulis sekali setelah seluruh berkas selesai diproses
        self._save_metadata()

    def _build_record(self, row: pd.Series, output_path: Path) -> dict:
        return {
            "dataset": row["dataset"],
            "filename": output_path.name,
            "filepath": output_path.relative_to(self.output_root).as_posix(),
            "speaker": row["speaker"],
            "raw_label": row["raw_label"],
            "emotion": row["emotion"],
            "sample_rate": row["sample_rate"],
            "duration": row["duration"],
            "augmentation": "pitch_noise",
        }

    def _save_metadata(self):
        metadata = pd.DataFrame(self.records)

        metadata.to_csv(
            self.output_root / "augmented_inventory.csv",
            index=False,
        )

        print(
            f"Metadata saved: "
            f"{self.output_root/'augmented_inventory.csv'}"
        )