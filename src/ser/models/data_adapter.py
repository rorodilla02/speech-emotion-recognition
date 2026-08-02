from __future__ import annotations
import numpy as np
import keras
from ..features.feature_dataset import FeatureSubset
from .constants import MODEL_INPUT_SHAPE, N_CLASSES


def to_model_inputs(
    subset: FeatureSubset,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengubah FeatureSubset menjadi pasangan larik siap pakai untuk Keras.

    Dua penyesuaian yang dilakukan:
    1. Penambahan sumbu kanal, (N, 51, 401) menjadi (N, 51, 401, 1)
    2. One-hot encoding label, agar metrik macro F1-score dapat dihitung
       selama training

    Catatan
    -------
    Fungsi ini tidak melakukan normalisasi apa pun. Penyeragaman skala
    fitur sudah dilakukan lewat CMVN per berkas pada tahap ekstraksi.
    """
    features = np.asarray(subset.features, dtype=np.float32)

    if features.ndim != 3:
        raise ValueError(
            f"Fitur harus tiga dimensi (N, frekuensi, waktu), "
            f"diterima: {features.shape}"
        )

    features = np.expand_dims(features, axis=-1)

    if features.shape[1:] != MODEL_INPUT_SHAPE:
        raise ValueError(
            f"Bentuk input tidak sesuai: "
            f"{features.shape[1:]} != {MODEL_INPUT_SHAPE}"
        )

    labels = np.asarray(subset.labels)

    if labels.size and labels.max() >= n_classes:
        raise ValueError(
            f"Indeks label melebihi jumlah kelas: "
            f"{int(labels.max())} >= {n_classes}"
        )

    labels = keras.utils.to_categorical(labels, num_classes=n_classes)

    return features, labels.astype(np.float32)