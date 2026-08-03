from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import(
    INTERNAL_VALIDATION_RATIO,
    MIN_SPEAKER_FOR_SPEAKER_SPLIT,
    VALIDATION_SPLIT_SEED,
)

class ValidationSplitter:
    """
    Menyisihkan sebagian data latih sebagai data validasi internal.

    Diperlukan pada RM2 dan RM3 yang tidak memiliki data validasi. Data
    uji tidak boleh dipakai untuk seleksi model, karena korpus target
    akan ikut menentukan kapan training berhenti dan seluruh klaim
    generalisasi kehilangan dasar.

    Kebijakan pemisahan mengikuti subbab 3.4.2. Korpus dengan speaker
    memadai dipisah berbasis speaker, sedangkan korpus dengan speaker
    terbatas dipisah secara stratified berdasarkan label emosi. Seluruh
    korpus latih pada fold ikut terwakili pada data validasi, karena
    komposisi korpus pada data validasi terbukti menentukan kualitas
    seleksi model.

    Catatan
    -------
    Kelas ini tidak:
    - mengubah berkas split di disk
    - menyentuh data uji
    - menangani data augmentasi, karena pemisahannya sudah ditegakkan
      oleh FeatureDataset lewat kolom filepath
    """

    def __init__(
        self,
        ratio: float = INTERNAL_VALIDATION_RATIO,
        seed: int = VALIDATION_SPLIT_SEED,
    ):
        if not 0.0 < ratio < 0.5:
            raise ValueError(f"Rasio validasi tidak wajar: {ratio}")

        self.ratio = ratio
        self.seed = seed

    def split(self, train_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if train_split.empty:
            raise ValueError("Data latih kosong.")

        train_parts = []
        validation_parts = []

        for dataset in sorted(train_split["dataset"].unique()):
            subset = train_split[train_split["dataset"] == dataset]
            speakers = sorted(subset["speaker"].unique())

            if len(speakers) >= MIN_SPEAKER_FOR_SPEAKER_SPLIT:
                train_part, validation_part = self._speaker_split(subset, speakers, dataset)
            else:
                train_part, validation_part = self._stratified_split(subset, dataset)

            train_parts.append(train_part)
            validation_parts.append(validation_part)

        train = pd.concat(train_parts, ignore_index=True)
        validation = pd.concat(validation_parts, ignore_index=True)

        self._verify(train_split, train, validation)

        return train, validation

    def _speaker_split(
        self,
        subset: pd.DataFrame,
        speakers: list,
        dataset: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_holdout = max(1, round(len(speakers) * self.ratio))

        rng = np.random.default_rng(self.seed)
        held_out = sorted(rng.choice(speakers, size=n_holdout, replace=False).tolist())

        mask = subset["speaker"].isin(held_out)

        print(f"{dataset:8} speaker-independent | validasi: {held_out} | {int(mask.sum())} dari {len(subset)} berkas")

        return subset[~mask].copy(), subset[mask].copy()

    def _stratified_split(
        self,
        subset: pd.DataFrame,
        dataset: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_part, validation_part = train_test_split(
            subset,
            test_size=self.ratio,
            stratify=subset["emotion"],
            random_state=self.seed,
            shuffle=True,
        )

        print(f"{dataset:8} stratified per label | speaker hanya {subset['speaker'].nunique()} | {len(validation_part)} dari {len(subset)} berkas")

        return train_part.copy(), validation_part.copy()

    @staticmethod
    def _verify(
        original: pd.DataFrame,
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ):
        if len(train) + len(validation) != len(original):
            raise ValueError(f"Jumlah berkas berubah: {len(train)} + {len(validation)} != {len(original)}")

        overlap = set(train["filepath"]) & set(validation["filepath"])

        if overlap:
            raise ValueError(f"{len(overlap)} berkas muncul di train sekaligus validasi.")

        missing =  set(original["dataset"]) - set(validation["dataset"])

        if missing:
            raise ValueError(f"Korpus tidak terwakili pada data validasi: {sorted(missing)}")