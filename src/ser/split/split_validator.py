from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from ..preprocessing.constants import TRAINING_DATASETS

@dataclass(slots=True)
class ValidationResult:
    validation: str
    status: str
    expected: str | int | float
    actual: str | int | float

class SplitValidator:
    def __init__(self, split_root: Path, metadata: pd.DataFrame):
        self.split_root = split_root
        self.metadata = metadata.copy()
        self.training_metadata = metadata[metadata["dataset"].isin(TRAINING_DATASETS)].copy()
            
    def validate(self) -> pd.DataFrame:
        results = [
            self._validate_total_files(),
            self._validate_dataset_distribution(),
            self._validate_label_distribution(),
            self._validate_speaker_overlap(),
        ]

        return pd.DataFrame([asdict(result) for result in results])

    def validate_rm2(self) -> pd.DataFrame:
        folds = [
            "fold_1",
            "fold_2",
            "fold_3",
        ]
        results = []

        for fold in folds:
            print(f"\nValidating {fold.upper()}")

            fold_root = self.split_root / fold

            train = pd.read_csv(fold_root / "train.csv")
            validation = pd.read_csv(fold_root / "validation.csv")
            test = pd.read_csv(fold_root / "test.csv")

            results.append(
                self._validate_rm2_total_files(
                    train,
                    validation,
                    test,
                    fold,
                )
            )

            results.append(
                self._validate_rm2_dataset_separation(
                    train=train,
                    test=test,
                    fold_name=fold,
                )
            )

            results.append(
                self._validate_rm2_label_distribution(
                    train=train,
                    validation=validation,
                    test=test,
                    fold_name=fold,
                )
            )

            results.append(
                self._validate_rm2_empty_validation(
                    validation=validation,
                    fold_name=fold,
                )
            )

        summary = pd.DataFrame([asdict(result) for result in results])

        self._save_report(summary, "rm2_validation_summary.csv")

        return summary

    def validate_rm3(self) -> pd.DataFrame:
        train = pd.read_csv(self.split_root/"train.csv")
        validation = pd.read_csv(self.split_root/"validation.csv")
        test = pd.read_csv(self.split_root/"test.csv")

        results = [
            self._validate_rm3_total_files(train, validation, test),
            self._validate_rm3_train_dataset(train),
            self._validate_rm3_test_dataset(test),
            self._validate_rm3_empty_validation(validation),
        ]

        summary = pd.DataFrame([asdict(r) for r in results])

        self._save_report(summary, "rm3_validation_summary.csv")
        
        return summary
    
    @staticmethod
    def _status(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    def _validate_rm3_total_files(
            self,
            train: pd.DataFrame,
            validation: pd.DataFrame,
            test: pd.DataFrame,
    ) -> ValidationResult:
        expected = len(self.metadata)
        actual = len(train) + len(validation) + len(test)

        return ValidationResult(
            validation="RM3 Total Files",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_rm3_train_dataset(self, train: pd.DataFrame) -> ValidationResult:
        expected = set(TRAINING_DATASETS)
        actual = set(train["dataset"].unique())

        return ValidationResult(
            validation="RM3 Train Dataset",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_rm3_test_dataset(self, test: pd.DataFrame) -> ValidationResult:
        expected = {"inesco"}
        actual = set(test["dataset"].unique())

        return ValidationResult(
            validation="RM3 Test Dataset",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_rm3_empty_validation(self, validation: pd.DataFrame) -> ValidationResult:
        expected = 0
        actual = len(validation)

        return ValidationResult(
            validation="RM3 Empty Validation",
            status=self._status(expected == actual),
            expected=expected,
            actual=actual,
        )

    def _validate_rm2_total_files(
            self,
            train: pd.DataFrame,
            validation: pd.DataFrame,
            test: pd.DataFrame,
            fold_name: str,
    ) -> ValidationResult:
        expected = len(self.training_metadata)

        actual = (
            len(train)
            + len(validation)
            + len(test)
        )

        status = self._status(expected == actual)

        return ValidationResult(
            validation=f"{fold_name.upper()} Total Files",
            status=status,
            expected=expected,
            actual=actual,
        )

    def _validate_rm2_dataset_separation(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        fold_name: str,
    ) -> ValidationResult:
        train_datasets = set(train["dataset"].unique())
        test_datasets = set(test["dataset"].unique())

        expected = {
            "fold_1": {
                "train": {"ravdess", "tess"},
                "test": {"savee"},
            },
            "fold_2": {
                "train": {"ravdess", "savee"},
                "test": {"tess"},
            },
            "fold_3": {
                "train": {"tess", "savee"},
                "test": {"ravdess"},
            },
        }

        expected_train = expected[fold_name]["train"]
        expected_test = expected[fold_name]["test"]

        condition = (
            train_datasets == expected_train
            and
            test_datasets == expected_test
        )

        status = self._status(condition)

        return ValidationResult(
            validation=f"{fold_name.upper()} Dataset Separation",
            status=status,
            expected=str(expected_train),
            actual=str(train_datasets),
        )

    def _validate_rm2_label_distribution(
            self,
            train: pd.DataFrame,
            validation: pd.DataFrame,
            test: pd.DataFrame,
            fold_name: str,
    ) -> ValidationResult:
        combined = pd.concat([train, validation, test], ignore_index=True)

        expected = self._label_counts(self.metadata)
        actual = self._label_counts(combined)
        labels = sorted(set(expected.keys()) | set(actual.keys()))
        condition = True

        for label in labels:
            if expected.get(label, 0) != actual.get(label, 0):
                condition = False
                break

        status = self._status(condition)

        return ValidationResult(
            validation=f"{fold_name.upper()} Label Distribution",
            status=status,
            expected=len(labels),
            actual=len(labels) if condition else 0,
        )

    def _validate_rm2_empty_validation(
                self,
                validation: pd.DataFrame,
                fold_name: str,
        ) -> ValidationResult:
        expected = 0
        actual = len(validation)

        status = self._status(actual == expected)

        return ValidationResult(
            validation=f"{fold_name.upper()} Empty Validation",
            status=status,
            expected=expected,
            actual=actual,
        )
    
    def _load_split(self, split_name: str) ->pd.DataFrame:
        return pd.read_csv(self.split_root/f"{split_name}.csv")
    
    def _load_all_splits(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self._load_split("train"),
            "validation": self._load_split("validation"),
            "test": self._load_split("test")
        }

    def _load_dataset_splits(self, dataset:str) -> dict[str, pd.DataFrame]:
        dataset_root = self.split_root/dataset

        return{
            "train": pd.read_csv(dataset_root/"train.csv"),
            "validation": pd.read_csv(dataset_root/"validation.csv"),
            "test": pd.read_csv(dataset_root/"test.csv")
        }
    
    def _save_report(self, dataframe: pd.DataFrame, filename: str):
        dataframe.to_csv(self.split_root/filename, index=False)

    def _dataset_counts(self, dataframe: pd.DataFrame) -> dict[str, int]:
        return(dataframe["dataset"].value_counts().to_dict())

    def _label_counts(self, dataframe: pd.DataFrame) -> dict[str, int]:
        return dataframe["emotion"].value_counts().to_dict()
    
    def _validate_total_files(self) -> ValidationResult:
        train = self._load_split("train")
        validation = self._load_split("validation")
        test = self._load_split("test")

        expected = len(self.metadata)
        actual = (len(train)+len(validation)+len(test))
        status = self._status(expected == actual)

        return ValidationResult(
            validation="Total Files",
            status=status,
            expected=expected,
            actual=actual,
        )

    def _validate_dataset_distribution(self) -> ValidationResult:
        splits = self._load_all_splits()
        expected = self._dataset_counts(self.metadata)

        combined = pd.concat(splits.values(), ignore_index=True)
        actual = self._dataset_counts(combined)

        records = []

        datasets = sorted(set(expected.keys()) | set(actual.keys()))

        for dataset in datasets:
            expected_count = expected.get(dataset, 0)
            actual_count = actual.get(dataset, 0)

            records.append({
                "dataset": dataset,
                "expected": expected_count,
                "actual": actual_count,
                "status": ("PASS" if expected_count == actual_count else "FAIL"),
            })

        report = pd.DataFrame(records)

        self._save_report(report, "dataset_distribution_validation.csv")
        
        status = self._status((report["status"] == "PASS").all())

        return ValidationResult(
            validation="Dataset Distribution",
            status=status,
            expected=len(report),
            actual=(report["status"] == "PASS").sum(),
        )

    def _validate_label_distribution(self) -> ValidationResult: 
        splits = self._load_all_splits()
        expected = self._label_counts(self.training_metadata)

        combined = pd.concat(splits.values(), ignore_index=True)
        actual = self._label_counts(combined)

        records = []

        labels = sorted(set(expected.keys()) | set(actual.keys()))

        for label in labels:
            expected_count = expected.get(label, 0)
            actual_count = actual.get(label, 0)

            records.append({
                "emotion": label,
                "expected": expected_count,
                "actual": actual_count,
                "status": ("PASS" if expected_count == actual_count else "FAIL"),
            })

        report = pd.DataFrame(records)

        self._save_report(report, "label_distribution_validation.csv")
        
        status = self._status((report["status"] == "PASS").all())

        return ValidationResult(
            validation="Label Distribution",
            status=status,
            expected=len(report),
            actual=(report["status"] == "PASS").sum(),
        )

    def _validate_speaker_overlap(self):
        for dataset in ("ravdess", "savee"):
            splits = self._load_dataset_splits(dataset)

        train = set(splits["train"]["speaker"])
        validation = set(splits["validation"]["speaker"])
        test = set(splits["test"]["speaker"])

        checks = [
            {
                "comparison": "train-validation",
                "overlap": len(train & validation),
            },
            {
                "comparison": "train-validation",
                "overlap": len(train & test),
            },
            {
                "comparison": "train-validation",
                "overlap": len(validation & test),
            },
        ]

        report = pd.DataFrame(checks)
        report["status"] = report["overlap"].apply(
            lambda x: "PASS" if x == 0 else "FAIL"
        )

        self._save_report(report, "speaker_overlap_validation.csv")

        status = self._status((report["status"] == "PASS").all())

        print("Train ∩ Validation :", train & validation)
        print("Train ∩ Test       :", train & test)
        print("Validation ∩ Test  :", validation & test)

        return ValidationResult(
            validation="Speaker Overlap",
            status=status,
            expected=len(report),
            actual=(report["status"] == "PASS").sum()
        )