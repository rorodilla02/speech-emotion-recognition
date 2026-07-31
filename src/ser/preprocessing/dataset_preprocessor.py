from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from .audio_loader import AudioLoader
from .audio_preprocessor import AudioPreprocessor
from .common import AudioData
from .constants import (
    CORRUPT_FILES,
    SPEAKER_CORRECTIONS,
    EXCLUDED_EMOTIONS,
    PROCESSED_AUDIO_SUBTYPE,
)


class DatasetPreprocessor:
    """
    Orkestrator preprocessing untuk seluruh dataset.

    Kelas ini memproses setiap file audio yang terdaftar pada metadata
    hasil audit, lalu menyimpan audio hasil preprocessing ke direktori
    output tanpa mengubah dataset asli.

    Catatan
    -------
    Kelas ini tidak:
    - mengimplementasikan algoritma preprocessing
    - memodifikasi dataset mentah
    - melakukan pembagian dataset
    - melakukan augmentasi data
    - melakukan normalisasi durasi
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        input_root: Path,
        output_root: Path,
        metadata_output: Path,
        failed_output: Path | None = None,
        log_every: int = 500,
    ):
        self.metadata = metadata.copy()
        self.input_root = input_root
        self.output_root = output_root
        self.metadata_output = metadata_output
        self.failed_output = failed_output
        self.log_every = log_every

        self.loader = AudioLoader()
        self.preprocessor = AudioPreprocessor()

        self._created_dirs: set[Path] = set()

    # --- alur utama --------------------------------------------------------

    def process(self) -> pd.DataFrame:
        metadata = self._clean_metadata()
        total = len(metadata)

        print(f"Files to process: {total}")

        processed_records = []
        failed_records = []

        for index, row in enumerate(metadata.itertuples(), start=1):
            try:
                processed_records.append(self._process_file(row))

            # Error handling terpusat: satu file gagal tidak menghentikan
            # keseluruhan proses (mitigasi risiko R-06).
            except Exception as error:
                failed_records.append(
                    {
                        "dataset": row.dataset,
                        "filename": row.filename,
                        "filepath": row.filepath,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

            if index % self.log_every == 0 or index == total:
                print(
                    f"Processed {index}/{total} "
                    f"({len(failed_records)} failed)"
                )

        processed_metadata = pd.DataFrame(processed_records)

        self._save_metadata(processed_metadata)
        self._save_failed(failed_records)

        return processed_metadata

    # --- data cleaning -----------------------------------------------------

    def _clean_metadata(self) -> pd.DataFrame:
        metadata = self.metadata.copy()

        # T-01: koreksi varian penulisan kode speaker
        for (dataset, wrong_speaker), correct_speaker in SPEAKER_CORRECTIONS.items():
            mask = (
                (metadata["dataset"] == dataset)
                & (metadata["speaker"] == wrong_speaker)
            )
            metadata.loc[mask, "speaker"] = correct_speaker

        # T-03: keluarkan file yang tidak terbaca
        for dataset, filename in CORRUPT_FILES:
            mask = (
                (metadata["dataset"] == dataset)
                & (metadata["filename"] == filename)
            )
            metadata = metadata[~mask]

        # Ruang lingkup Bab 1: ruang label konsisten antar korpus latih
        metadata = metadata[
            ~metadata["emotion"].isin(EXCLUDED_EMOTIONS)
        ]

        return metadata.reset_index(drop=True)

    # --- pemrosesan per file ----------------------------------------------

    def _build_output_path(self, filepath: Path) -> Path:
        output_path = self.output_root / filepath
        parent = output_path.parent

        # mkdir hanya sekali per direktori
        if parent not in self._created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            self._created_dirs.add(parent)

        return output_path

    def _save_audio(self, audio_data: AudioData, output_path: Path):
        audio = np.asarray(audio_data.audio, dtype=np.float32)

        sf.write(
            file=output_path,
            data=audio,
            samplerate=audio_data.sample_rate,
            subtype=PROCESSED_AUDIO_SUBTYPE,
        )

    def _process_file(self, row) -> dict:
        input_path = self.input_root / Path(row.filepath)

        audio_data = self.loader.load(input_path)
        processed_audio = self.preprocessor.process(audio_data)

        output_path = self._build_output_path(Path(row.filepath))
        self._save_audio(processed_audio, output_path)

        return {
            "dataset": row.dataset,
            "filename": row.filename,
            "filepath": row.filepath,
            "speaker": row.speaker,
            "raw_label": row.raw_label,
            "emotion": row.emotion,
            "sample_rate": processed_audio.sample_rate,
            "duration": (
                len(processed_audio.audio)
                / processed_audio.sample_rate
            ),
        }

    # --- output ------------------------------------------------------------

    def _save_metadata(self, metadata: pd.DataFrame):
        self.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(self.metadata_output, index=False)

        print(f"Metadata saved: {self.metadata_output}")

    def _save_failed(self, failed_records: list[dict]):
        if self.failed_output is None:
            if failed_records:
                print(f"WARNING: {len(failed_records)} file gagal diproses.")
            return

        self.failed_output.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "dataset",
            "filename",
            "filepath",
            "error_type",
            "error_message",
        ]
        pd.DataFrame(failed_records, columns=columns).to_csv(
            self.failed_output, index=False
        )

        print(
            f"Failed files: {len(failed_records)} "
            f"-> {self.failed_output}"
        )