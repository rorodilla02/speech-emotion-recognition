from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ..features.constants import EMOTION_LABELS

DPI = 300


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: bool = True,
):
    """
    Menyimpan confusion matrix sebagai gambar beresolusi 300 dpi.

    Urutan label mengikuti EMOTION_LABELS agar konsisten pada seluruh
    gambar yang dihasilkan.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = matrix.astype(float)

    if normalize:
        row_sum = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display, row_sum, out=np.zeros_like(display), where=row_sum != 0
        )

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 2))
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() or 1)

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(title)

    threshold = (display.max() or 1) / 2

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{display[i, j]:.2f}" if normalize else f"{matrix[i, j]}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if display[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(history: pd.DataFrame, title: str, output_path: Path):
    """
    Menyimpan kurva macro F1-score dan loss selama training.

    Epoch terbaik ditandai agar titik henti early stopping terlihat.
    """
    best = int(history["val_macro_f1"].idxmax())
    epochs = history["epoch"] + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["macro_f1"], label="latih")
    axes[0].plot(epochs, history["val_macro_f1"], label="validasi")
    axes[0].axvline(best + 1, linestyle="--", linewidth=1, color="gray")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro F1-score")
    axes[0].set_title("Macro F1-score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["loss"], label="latih")
    axes[1].plot(epochs, history["val_loss"], label="validasi")
    axes[1].axvline(best + 1, linestyle="--", linewidth=1, color="gray")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{title} (epoch terbaik {best + 1})")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def emotion_labels() -> list[str]:
    return list(EMOTION_LABELS)