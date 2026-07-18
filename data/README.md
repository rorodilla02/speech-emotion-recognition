# Dataset

This project uses four public Speech Emotion Recognition (SER) datasets.

## Datasets

- RAVDESS
- TESS
- SAVEE
- INESCO

---

## Directory Structure

```text
data/
├── raw/
│   ├── archives/
│   ├── ravdess/
│   ├── tess/
│   ├── savee/
│   └── inesco/
│
├── processed/
│
├── features/
│
└── metadata/
```

---

## Dataset Policy

The original datasets must remain unchanged.

- Do not rename files.
- Do not modify directory structures.
- Do not overwrite original audio files.

Any preprocessing results must be stored inside:

```
processed/
```

Extracted features must be stored inside:

```
features/
```

Metadata generated during the audit process is stored inside:

```
metadata/
```

---

## Current Status

- [x] Dataset downloaded
- [x] Dataset extracted
- [x] Dataset audited
- [ ] Audio preprocessing
- [ ] Feature extraction
- [ ] Model training

---

## Audit Result

Current audit covers:

- filename validation
- label parsing
- speaker inventory
- audio readability
- sample rate inspection
- duration statistics
- emotion distribution
- failed file reporting

All readable audio files successfully passed the audit except one corrupted audio file from the INESCO dataset.