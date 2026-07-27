from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from ..preprocessing.constants import (
    TRAINING_DATASETS,
    RANDOM_SEED,
    RM1_TRAIN_RATIO,
    RM1_VALIDATION_RATIO,
)
from .common import (
    SpeakerSplit,
    DatasetSplit,
)

class SplitGenerator:
    """
    Generates dataset splits for each research scenario.
    """

    def __init__(self, metadata: pd.DataFrame, output_root: Path):
        self.metadata = metadata.copy()
        self.output_root = output_root

    def generate_rm1(self) -> pd.DataFrame:
        training_metadata = self.metadata[self.metadata["dataset"].isin(TRAINING_DATASETS)].copy()
        for dataset in TRAINING_DATASETS:
            dataset_metadata = self._get_dataset_metadata(training_metadata, dataset)

            print(f"\nPreprocessing {dataset}")
            print(f"Files: {len(dataset_metadata)}")

            if dataset == "tess":
                self._generate_stratified_split(dataset_metadata, dataset)
            else:
                self._generate_speaker_split(dataset_metadata, dataset)

        self._combined_dataset_splits(scenario="rm1")
    
    def generate_rm2(self):
        self._generate_loco_fold(
            train_datasets=["ravdess", "tess"],
            test_dataset="savee",
            fold_name="fold_1",
        )

        self._generate_loco_fold(
            train_datasets=["ravdess", "savee"],
            test_dataset="tess",
            fold_name="fold_2",
        )

        self._generate_loco_fold(
            train_datasets=["tess", "savee"],
            test_dataset="ravdess",
            fold_name="fold_3",
        )
    
    def generate_rm3(self):
        train_metadata = self.metadata[
            self.metadata["dataset"].isin(TRAINING_DATASETS)
        ].copy()

        test_metadata = self.metadata[
            self.metadata["dataset"] == "inesco"
        ].copy()

        print(f"Training files: {len(train_metadata)}")
        print(f"Testing files: {len(test_metadata)}")

        dataset_split = DatasetSplit(
            train=train_metadata,
            validation=train_metadata.iloc[0:0].copy(),
            test=test_metadata,
        )

        dataset_split = self._validate_rm3_split(dataset_split)

        self._save_dataset_split(
            split=dataset_split,
            scenario="rm3",
            dataset="",
        )

        print("\nRM3 Summary")
        print(f"Train: {len(dataset_split.train)}")
        print(f"Validation: {len(dataset_split.validation)}")
        print(f"Test: {len(dataset_split.test)}")

    def _filter_training_metadata(self):
        pass

    def _validate_rm3_split(self, split: DatasetSplit):
        total = (len(split.train) + len(split.validation) + len(split.test))

        print(f"Total files: {total}")

        return split

    def _generate_loco_fold(self, train_datasets: list[str], test_dataset: str, fold_name: str):
        train_metadata = self.metadata[
            self.metadata["dataset"].isin(train_datasets)
        ].copy()

        test_metadata = self.metadata[
            self.metadata["dataset"] == test_dataset
        ].copy()

        print(f"\n{fold_name.upper()}")
        print(f"Training datasets : {train_datasets}")
        print(f"Testing dataset   : {test_dataset}")
        print(f"Train files : {len(train_metadata)}")
        print(f"Test files  : {len(test_metadata)}")

        dataset_split = DatasetSplit(
            train=train_metadata,
            validation=train_metadata.iloc[0:0].copy(),
            test=test_metadata,
        )

        self._validate_loco_split(dataset_split)
        self._save_dataset_split(
            split=dataset_split,
            scenario="rm2",
            dataset=fold_name,
        )

        print(f"\n{fold_name.upper()} Summary")
        print(f"Train : {len(dataset_split.train)}")
        print(f"Validation : {len(dataset_split.validation)}")
        print(f"Test : {len(dataset_split.test)}")
    
    def _validate_loco_split(self, split: DatasetSplit):
        total = (
            len(split.train)
            + len(split.validation)
            + len(split.test)
        )

        print(f"Total files : {total}")

    def _get_unique_speakers(self, metadata: pd.DataFrame, dataset: str) -> list[str]:
        speakers = (
            metadata.loc[metadata["dataset"] == dataset, "speaker"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return speakers
    
    def _get_dataset_metadata(self, metadata: pd.DataFrame, dataset: str) -> pd.DataFrame:
        return metadata[metadata["dataset"] == dataset].copy()
    
    def _split_speakers(self, speakers: list[str], dataset: str) -> SpeakerSplit:
        speakers = speakers.copy()

        random.seed(RANDOM_SEED)
        random.shuffle(speakers)

        if dataset == "savee":
            return SpeakerSplit(
                train=speakers[:3],
                validation=[],
                test=speakers[3:],
            )
        
        total = len(speakers)
        
        train_size = round(total*RM1_TRAIN_RATIO)
        validation_size = round(total*RM1_VALIDATION_RATIO)
        
        train = speakers[:train_size]
        validation = speakers[train_size:train_size+validation_size]
        test = speakers[train_size+validation_size:]

        return SpeakerSplit(
            train=train,
            validation=validation,
            test=test,
        )

    def _build_metadata_split(self, metadata: pd.DataFrame, split: SpeakerSplit) -> DatasetSplit:
        train = metadata[metadata["speaker"].isin(split.train)].copy()
        validation = metadata[metadata["speaker"].isin(split.validation)].copy()
        test = metadata[metadata["speaker"].isin(split.test)].copy()

        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
        )
    
    def _validate_speaker_overlap(self, split: DatasetSplit):
        train = set(split.train["speaker"])
        validation = set(split.validation["speaker"])
        test = set(split.test["speaker"])

        if train & validation:
            raise ValueError("Speaker overlap between train and validation.")
        if train & test:
            raise ValueError("Speaker overlap between train and test.")
        if test& validation:
            raise ValueError("Speaker overlap between test and validation.")
        
        print("Speaker overlap validation passed.")

    def _save_dataset_split(self, split: DatasetSplit, scenario: str, dataset: str):
        output_dir = (self.output_root/scenario/dataset)
        output_dir.mkdir(parents=True, exist_ok=True)

        split.train.to_csv(output_dir/"train.csv", index=False)
        split.validation.to_csv(output_dir/"validation.csv", index=False)
        split.test.to_csv(output_dir/"test.csv", index=False)

    def _generate_speaker_split(self, metadata: pd.DataFrame, dataset: str):
        speakers = self._get_unique_speakers(metadata, dataset)
        speaker_split = self._split_speakers(speakers, dataset)
        dataset_split = self._build_metadata_split(metadata, speaker_split)

        print(f"Train: {len(dataset_split.train)}")
        print(f"Validation: {len(dataset_split.validation)}")
        print(f"Test: {len(dataset_split.test)}")

        self._validate_speaker_overlap(dataset_split)
        self._save_dataset_split(
            split=dataset_split,
            scenario="rm1",
            dataset=dataset,
        )

    def _generate_stratified_split(self, metadata: pd.DataFrame, dataset: str):
        train, temp = train_test_split(
            metadata,
            train_size=RM1_TRAIN_RATIO,
            random_state=RANDOM_SEED,
            stratify=metadata["emotion"],
        )

        validation_ratio = (RM1_VALIDATION_RATIO/(1-RM1_TRAIN_RATIO))
        validation, test = train_test_split(
            temp,
            train_size=validation_ratio,
            random_state=RANDOM_SEED,
            stratify=temp["emotion"],
        )

        dataset_split = DatasetSplit(
            train=train,
            validation=validation,
            test=test,
        )

        print(f"Train: {len(train)}")
        print(f"Validation: {len(validation)}")
        print(f"Test: {len(test)}")

        self._validate_label_distribution(dataset_split)
        self._save_dataset_split(
            split=dataset_split,
            scenario="rm1",
            dataset=dataset,
        )

    def _validate_label_distribution(self, split: DatasetSplit):
        train_labels = set(split.train["emotion"])
        validation_labels = set(split.validation["emotion"])
        test_labels = set(split.test["emotion"])

        if train_labels != validation_labels:
            raise ValueError("Emotion labels differ between train and validation.")
        if train_labels != test_labels:
            raise ValueError("Emotion labels differ between train and test.")
        
        print("Label distribution validation passed.")

    def _combined_dataset_splits(self, scenario: str):
        train_splits = []
        validation_splits = []
        test_splits = []

        for dataset in TRAINING_DATASETS:
            dataset_dir = self.output_root/scenario/dataset

            train = pd.read_csv(dataset_dir/"train.csv")
            validation = pd.read_csv(dataset_dir/"validation.csv")
            test = pd.read_csv(dataset_dir/"test.csv")

            train_splits.append(train)
            validation_splits.append(validation)
            test_splits.append(test)

        train = pd.concat(train_splits, ignore_index=True)
        validation = pd.concat(validation_splits, ignore_index=True)
        test = pd.concat(test_splits, ignore_index=True)

        output_dir = self.output_root/scenario
        output_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(output_dir/"train.csv", index=False)
        validation.to_csv(output_dir/"validation.csv", index=False)
        test.to_csv(output_dir/"test.csv", index=False)

        print("\nCombined RM1 Split")
        print(f"Train: {len(train)}")
        print(f"Validation: {len(validation)}")
        print(f"test: {len(test)}")