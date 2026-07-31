from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..preprocessing.audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor
from .constants import (
    FEATURE_SHAPE,
    SOURCE_PROCESSED,
    SOURCE_AUGMENTED,
)


class DatasetFeatureExtractor:
    """
    Mengekstraksi fitur untuk seluruh berkas audio yang tersedia.

    Fitur diekstraksi satu kali per berkas unik, lalu disimpan pada
    satu larik tunggal. Setiap skenario penelitian cukup mengacu pada
    indeks baris melalui feature_index.csv, sehingga tidak ada berkas
    yang diekstraksi berulang kali (mitigasi risiko R-01).

    Catatan
    -------
    Kelas ini tidak:
    - mengimplementasikan algoritma ekstraksi fitur
    - melakukan pembagian dataset
    - melakukan augmentasi
    - melakukan normalisasi berbasis statistik data latih
    """

    INDEX_COLUMNS = [
        "row_index",
        "source",
        "dataset",
        "filename",
        "filepath",
        "speaker",
        "raw_label",
        "emotion",
        "augmentation",
        "n_samples",
        "real_frames",
    ]

    def __init__(
        self,
        processed_metadata: pd.DataFrame,
        augmented_metadata: pd.DataFrame,
        processed_root: Path,
        augmented_root: Path,
        output_root: Path,
        log_every: int = 500,
    ):
        self.processed_metadata = processed_metadata.copy()
        self.augmented_metadata = augmented_metadata.copy()
        self.processed_root = processed_root
        self.augmented_root = augmented_root
        self.output_root = output_root
        self.log_every = log_every

        self.loader = AudioLoader()
        self.extractor = FeatureExtractor()

    def run(self) -> pd.DataFrame:
        tasks = self._build_tasks()
        total = len(tasks)

        self.output_root.mkdir(parents=True, exist_ok=True)
        features_path = self.output_root / "features.npy"

        print(f"Total berkas: {total}")
        print(f"Bentuk fitur: {FEATURE_SHAPE}")
        print(f"Output       : {features_path}")

        array = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(total, *FEATURE_SHAPE),
        )

        records = []
        failed = []
        row_index = 0

        for position, task in enumerate(tasks, start=1):
            try:
                features, n_samples = self._extract_one(task)
                array[row_index] = features

                records.append(
                    {
                        "row_index": row_index,
                        "source": task["source"],
                        "dataset": task["dataset"],
                        "filename": task["filename"],
                        "filepath": task["filepath"],
                        "speaker": task["speaker"],
                        "raw_label": task["raw_label"],
                        "emotion": task["emotion"],
                        "augmentation": task["augmentation"],
                        "n_samples": n_samples,
                        "real_frames": self.extractor.real_frames(n_samples),
                    }
                )
                row_index += 1

            except Exception as error:
                failed.append(
                    {
                        "source": task["source"],
                        "filepath": task["filepath"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

            if position % self.log_every == 0 or position == total:
                print(f"Extracted {position}/{total} ({len(failed)} failed)")

        array.flush()
        del array

        if failed:
            self._truncate_features(features_path, row_index, total)

        index = pd.DataFrame(records, columns=self.INDEX_COLUMNS)
        self._save_index(index)
        self._save_failed(failed)

        return index

    def _build_tasks(self) -> list[dict]:
        tasks = []

        for row in self.processed_metadata.itertuples():
            tasks.append(
                {
                    "source": SOURCE_PROCESSED,
                    "root": self.processed_root,
                    "dataset": row.dataset,
                    "filename": row.filename,
                    "filepath": row.filepath,
                    "speaker": row.speaker,
                    "raw_label": row.raw_label,
                    "emotion": row.emotion,
                    "augmentation": "none",
                }
            )

        for row in self.augmented_metadata.itertuples():
            tasks.append(
                {
                    "source": SOURCE_AUGMENTED,
                    "root": self.augmented_root,
                    "dataset": row.dataset,
                    "filename": row.filename,
                    "filepath": row.filepath,
                    "speaker": row.speaker,
                    "raw_label": row.raw_label,
                    "emotion": row.emotion,
                    "augmentation": getattr(row, "augmentation", "unknown"),
                }
            )

        return tasks

    def _extract_one(self, task: dict) -> tuple[np.ndarray, int]:
        audio_path = task["root"] / Path(task["filepath"])
        audio_data = self.loader.load(audio_path)
        n_samples = int(np.asarray(audio_data.audio).shape[-1])

        return self.extractor.extract(audio_data), n_samples

    @staticmethod
    def _truncate_features(path: Path, valid_rows: int, total: int):
        """
        Memotong baris kosong bila ada berkas yang gagal diekstraksi,
        agar jumlah baris larik tetap sama dengan jumlah baris indeks.
        """
        print(f"Memotong larik fitur: {total} -> {valid_rows} baris")

        array = np.load(path, mmap_mode="r")
        trimmed = np.array(array[:valid_rows], dtype=np.float32)

        del array
        np.save(path, trimmed)

    def _save_index(self, index: pd.DataFrame):
        output_path = self.output_root / "feature_index.csv"
        index.to_csv(output_path, index=False)

        print(f"Index disimpan: {output_path}")

    def _save_failed(self, failed: list[dict]):
        if not failed:
            print("Tidak ada berkas yang gagal diekstraksi.")
            return

        output_path = self.output_root / "feature_failed_files.csv"
        pd.DataFrame(failed).to_csv(output_path, index=False)

        print(f"Berkas gagal: {len(failed)} -> {output_path}")