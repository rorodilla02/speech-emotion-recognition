from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np

from ..features.constants import EMOTION_LABELS
from ..features.feature_extractor import FeatureExtractor
from ..preprocessing.audio_preprocessor import AudioPreprocessor
from ..preprocessing.common import AudioData


@dataclass(slots=True)
class PredictionResult:
    """Hasil satu kali inferensi. Label dan kunci probabilities memakai
    label Inggris asli model (EMOTION_LABELS) - penerjemahan ke Bahasa
    Indonesia (KF-09, Tabel 2.1) sengaja dilakukan di lapisan UI (app/),
    bukan di sini, supaya paket src/ser tetap dapat dipakai ulang di
    luar konteks prototipe web."""

    label: str
    probabilities: dict[str, float]


class EmotionPredictor:
    """
    Menjalankan inferensi emosi ujung ke ujung untuk satu berkas audio:
    preprocessing -> ekstraksi fitur -> klasifikasi CNN.

    Memakai ulang AudioPreprocessor dan FeatureExtractor apa adanya
    (mitigasi risiko R-05 pada Tabel 3.4, train-serve mismatch) - kelas
    ini hanya mengorkestrasi pemanggilan keduanya dan menerjemahkan
    keluaran model mentah menjadi label serta distribusi probabilitas.

    Catatan
    -------
    Kelas ini tidak:
    - memuat audio dari bytes atau path (lihat AudioBytesLoader/AudioLoader)
    - melakukan validasi format atau durasi (tanggung jawab lapisan UI)
    - mengubah, melatih ulang, atau mengkalibrasi model
    """

    def __init__(self, model_path: str):
        # Keras 3 native format (.keras) - pemuatan lewat `keras.models`,
        # bukan `tf.keras.models`, karena stack proyek memakai Keras 3.15
        # sebagai paket mandiri di atas backend TensorFlow.
        self._model = keras.models.load_model(model_path)
        self._preprocessor = AudioPreprocessor()
        self._feature_extractor = FeatureExtractor()

    def predict(self, audio_data: AudioData) -> PredictionResult:
        processed = self._preprocessor.process(audio_data)
        features = self._feature_extractor.extract(processed)  # (51, 401)

        # (51, 401) -> (1, 51, 401, 1): tambah sumbu batch dan sumbu kanal
        model_input = features[np.newaxis, ..., np.newaxis]

        raw_output = self._model.predict(model_input, verbose=0)[0]

        if len(raw_output) != len(EMOTION_LABELS):
            raise RuntimeError(
                f"Jumlah keluaran model ({len(raw_output)}) tidak sesuai "
                f"jumlah kelas EMOTION_LABELS ({len(EMOTION_LABELS)}). "
                "Kemungkinan model_path menunjuk ke model yang salah."
            )

        probabilities = {
            label: float(score) for label, score in zip(EMOTION_LABELS, raw_output)
        }
        predicted_label = max(probabilities, key=probabilities.get)

        return PredictionResult(label=predicted_label, probabilities=probabilities)