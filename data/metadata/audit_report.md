# Dataset Audit Report

Generated at: 2026-07-31T21:36:11

## Summary

- Total datasets: 4
- Total files: 7119
- Successful files: 7118
- Failed files: 1
- Total speakers: 35

## Dataset Statistics

| Dataset | Files | Speakers | Sample Rates | Min | Mean | Max | Std | P5 | P95 |
|---------|-------|----------|--------------|-----|------|-----|-----|----|-----|
| ravdess | 1440 | 24 | 16000 | 2.94 | 3.70 | 5.27 | 0.34 | 3.20 | 4.31 |
| tess | 2800 | 3 | 24414, 96000 | 1.25 | 2.06 | 2.98 | 0.32 | 1.53 | 2.59 |
| savee | 480 | 4 | 44100 | 1.63 | 3.84 | 7.14 | 1.08 | 2.31 | 5.86 |
| inesco | 2398 | 4 | 16000, 44100 | 0.86 | 4.54 | 11.84 | 1.55 | 2.37 | 7.39 |

## Emotion Distribution

### ravdess

| Emotion | Total |
|---------|-------|
| Fear | 192 |
| Calm | 192 |
| Angry | 192 |
| Sad | 192 |
| Disgust | 192 |
| Surprise | 192 |
| Happy | 192 |
| Neutral | 96 |

### tess

| Emotion | Total |
|---------|-------|
| Angry | 400 |
| Sad | 400 |
| Surprise | 400 |
| Happy | 400 |
| Fear | 400 |
| Neutral | 400 |
| Disgust | 400 |

### savee

| Emotion | Total |
|---------|-------|
| Surprise | 60 |
| Happy | 60 |
| Neutral | 120 |
| Angry | 60 |
| Fear | 60 |
| Disgust | 60 |
| Sad | 60 |

### inesco

| Emotion | Total |
|---------|-------|
| Sad | 796 |
| Happy | 843 |
| Angry | 759 |

## Failed Files

| Dataset | Filename | Error |
|---------|----------|-------|
| inesco | mbaz_h138.wav | Failed to read audio metadata: /home/dilla/projects/tensorflow-gpu/speech-emotion-recognition/data/raw/inesco/INESCO Dataset Indonesian Expressive Speech Corpus/INESCO Dataset/Male/mbaz/mbaz_h138.wav (Error opening '/home/dilla/projects/tensorflow-gpu/speech-emotion-recognition/data/raw/inesco/INESCO Dataset Indonesian Expressive Speech Corpus/INESCO Dataset/Male/mbaz/mbaz_h138.wav': Format not recognised.) |