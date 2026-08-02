from __future__ import annotations
import keras
from keras import layers
from .constants import(
    MODEL_INPUT_SHAPE,
    N_CLASSES,
    CONV_BLOCKS,
    KERNEL_SIZE,
    CONV_PADDING,
    ACTIVATION,
    DENSE_UNITS,
    DENSE_DROPOUT_RATE,
    OUTPUT_ACTIVATION,
    USE_BATCH_NORM,
    MODEL_NAME,
)

class CNNArchitecture:
    """
    Membangun arsitektur CNN untuk klasifikasi emosi berbasis suara.

    Susunan lapisan mengikuti subbab 2.4.2, yaitu Convolution, ReLU,
    Pooling, Flatten, Dense, dan Softmax. Batch Normalization dan Dropout
    ditambahkan sebagai komponen regularisasi.

    Struktur
    --------
    Empat blok konvolusi, masing-masing:
        Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout (opsional)
    diikuti Flatten, satu lapisan Dense tersembunyi, dan lapisan output
    Softmax sebanyak jumlah kelas emosi.

    Catatan
    -------
    Kelas ini tidak:
    - melakukan kompilasi model (loss, optimizer, metrik)
    - melatih maupun mengevaluasi model
    - memuat atau menyiapkan data
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = MODEL_INPUT_SHAPE,
        n_classes: int=N_CLASSES,
        use_batch_norm: bool=USE_BATCH_NORM,
        name: str=MODEL_NAME
    ):
        if len(input_shape) != 3:
            raise ValueError(
                f"Bentuk input harus tiga dimensi (frekuensi, waktu, kanal), "
                f"diterima: {input_shape}"
            )

        if n_classes < 2:
            raise ValueError(f"Jumah kelas tidak valid: {n_classes}")

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.name = name

    def build(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_shape, name="feature_fusion")
        x = inputs

        for position, block in enumerate(CONV_BLOCKS, start=1):
            filters, pool_freq, pool_time, dropout_rate = block
            x = self._conv_block(
                x,
                filters=filters,
                pool_size=(pool_freq, pool_time),
                dropout_rate=dropout_rate,
                position=position,
            )

        x = layers.Flatten(name="flatten")(x)
        x =  self._dense_block(x)

        outputs = layers.Dense(
            self.n_classes,
            activation=OUTPUT_ACTIVATION,
            name="output",
        )(x)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def _conv_block(
        self,
        x,
        filters: int,
        pool_size: tuple[int, int],
        dropout_rate: float,
        position: int,
    ):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            padding=CONV_PADDING,
            use_bias=not self.use_batch_norm,
            name=f"conv{position}",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"bn{position}")(x)

        x = layers.Activation(ACTIVATION, name=f"relu{position}")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=f"pool{position}")(x)

        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"dropout{position}")(x)

        return x

    def _dense_block(self, x):
        x = layers.Dense(
            DENSE_UNITS,
            use_bias=not self.use_batch_norm,
            name="dense",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(name="bn_dense")(x)

        x = layers.Activation(ACTIVATION, name="relu_dense")(x)
        x = layers.Dropout(DENSE_DROPOUT_RATE, name="dropout_dense")(x)

        return x


def build_cnn(
    input_shape: tuple[int, int, int] = MODEL_INPUT_SHAPE,
    n_classes: int = N_CLASSES,
) -> keras.Model:
    """Pintasan pembangunan model dengan konfigurasi baku penelitian."""
    return CNNArchitecture(
        input_shape=input_shape,
        n_classes=n_classes,
    ).build()