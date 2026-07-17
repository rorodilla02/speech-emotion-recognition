# Speech Emotion Recognition using CNN and Feature Fusion for Cross-Corpus Evaluation

This repository contains the implementation of a Speech Emotion Recognition
(SER) system developed as part of an undergraduate thesis.

The project follows the CRISP-DM methodology and focuses on building a
cross-corpus CNN-based emotion recognition model using feature fusion of
MFCC, Delta MFCC, Delta-Delta MFCC, and Chroma features.

## Project layout

```text
app/                  Streamlit application
artifacts/            Generated models, metrics, and figures (not committed)
configs/              Reproducible experiment settings
data/
  raw/                Original extracted datasets (not committed)
  processed/          Cleaned/intermediate audio data (not committed)
  features/           Extracted feature arrays (not committed)
  metadata/           Dataset inventories and file-level metadata
docs/                 Technical project notes
scripts/              Runnable pipeline entry points
src/ser/              Reusable SER Python package
tests/                Automated tests
```

## Dataset rule

Store ZIP files in `data/raw/archives/` and extract each corpus to its own
folder under `data/raw/`. Preserve all original audio filenames and directory
structures. Label standardization belongs in metadata, never in renamed files.

See `docs/dataset_inventory.md` and `data/metadata/dataset_inventory.csv`
before adding a dataset.

## Project Status

Current phase:

✔ Data Understanding

Current milestone:

Dataset Audit

## Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- SoundFile
- Pandas
- NumPy
- Streamlit

## Development Roadmap

- [x] Project initialization
- [x] Dataset parser
- [ ] Audio reader
- [ ] Dataset auditor
- [ ] Feature extraction
- [ ] CNN training
- [ ] Cross-corpus evaluation
- [ ] Streamlit deployment