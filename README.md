# Speech Emotion Recognition using CNN and Feature Fusion for Cross-Corpus Evaluation

This repository contains the implementation of a Speech Emotion Recognition (SER)
system developed as part of an undergraduate thesis.

The project follows the **CRISP-DM** methodology and focuses on building a
cross-corpus CNN-based emotion recognition model using feature fusion of:

- MFCC
- Delta MFCC
- Delta-Delta MFCC
- Chroma

for emotion classification across multiple public SER datasets.

---

# Project Layout

```text
app/                  Streamlit application
artifacts/            Generated models, metrics, and figures (not committed)

configs/              Reproducible experiment settings

data/
├── raw/              Original extracted datasets (not committed)
├── processed/        Cleaned datasets (not committed)
├── features/         Extracted feature arrays (not committed)
└── metadata/         Dataset audit outputs

docs/                 Technical documentation
scripts/              Runnable pipeline entry points
src/ser/              Reusable SER package
tests/                Automated tests
```

---

# Dataset Rule

Store compressed archives inside:

```
data/raw/archives/
```

Extract each dataset into its own directory under:

```
data/raw/
```

Do **not** rename or modify original filenames.

Label standardization is performed by the dataset parser during metadata generation,
not by changing the original dataset.

---

# Current Project Status

Current CRISP-DM Phase

✅ Data Understanding

Current Milestone

✅ Dataset Audit

Next Milestone

⬜ Data Preparation

---

# Implemented Components

- ✅ Dataset Parser
- ✅ Audio Reader
- ✅ Dataset Auditor
- ✅ Statistics Generator
- ✅ Output Writer

Generated outputs:

- audit_summary.csv
- audit_summary.json
- dataset_statistics.csv
- file_inventory.csv
- emotion_distribution.csv
- failed_files.csv
- label_mapping.csv
- audit_report.md

---

# Technology Stack

- Python
- TensorFlow / Keras
- SoundFile
- Librosa
- Pandas
- NumPy
- Streamlit

---

# Development Roadmap

## Data Understanding

- [x] Dataset parser
- [x] Audio reader
- [x] Dataset auditor
- [x] Statistics generator
- [x] Output writer

## Data Preparation

- [ ] Audio preprocessing
- [ ] Dataset validation
- [ ] Feature extraction

## Modeling

- [ ] CNN training
- [ ] Hyperparameter tuning

## Evaluation

- [ ] Cross-corpus evaluation

## Deployment

- [ ] Streamlit application