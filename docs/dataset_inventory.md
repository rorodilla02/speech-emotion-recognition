# Dataset inventory

`data/metadata/dataset_inventory.csv` is the single inventory for every
corpus used in this project. It is intentionally tracked by Git; raw audio and
ZIP archives are not.

## Folder and archive names

| Dataset | Archive | Extraction folder |
| --- | --- | --- |
| RAVDESS | `ravdess_speech.zip` | `data/raw/ravdess/` |
| TESS | `tess.zip` | `data/raw/tess/` |
| SAVEE | `savee.zip` | `data/raw/savee/` |
| INESCO | `inesco.zip` | `data/raw/inesco/` |

Do not rename the original audio files. Several corpora encode the actor,
emotion, intensity, or utterance in their names.

## Inventory procedure after each extraction

1. Set `download_status` and `extract_status` to `complete`.
2. Record the actual archive name if it differs from the suggested name.
3. Count audio files, speakers, sample rates, and duration range.
4. Verify the source labels against the directory and filename conventions.
5. Record the source URL, licence, version, and access date.
6. Leave `usable_for_training` as `no` until the label mapping and files have
   been verified by the preparation script.

## Canonical labels

The English training label space is:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

RAVDESS `calm` is excluded. TESS `pleasant_surprise` is mapped to `surprise`.
INESCO is evaluated only for `angry`, `happy`, and `sad`; its exact source
label spelling must be confirmed after extraction.
