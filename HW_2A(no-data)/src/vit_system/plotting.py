from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .analysis import MisclassifiedSample
from .utils import denormalize_image, ensure_dir


def plot_history(history: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    ax.plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    ax.plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    ax.set_xlabel("epoch")
    ax.set_title("Training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    size = max(6, min(14, len(class_names) * 0.6))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_class_accuracy(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    classes = [row["class"] for row in rows]
    values = [row["accuracy"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 5))
    ax.bar(classes, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-class accuracy")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_misclassified(samples: list[MisclassifiedSample], class_names: list[str], path: str | Path, max_items: int = 16) -> None:
    if not samples:
        return
    path = Path(path)
    ensure_dir(path.parent)
    items = samples[:max_items]
    cols = min(4, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, sample in zip(axes.ravel(), items):
        image = denormalize_image(sample.image.cpu())
        image = image.permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(
            f"true: {class_names[sample.target]}\npred: {class_names[sample.prediction]}\nconf: {sample.confidence:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sweep_accuracy(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    labels = [row["run_name"] for row in rows]
    values = [row["best_val_acc"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("best_val_acc")
    ax.set_title("Validation accuracy by configuration")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sweep_tradeoff(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    x = [row["params_total"] for row in rows]
    y = [row["best_val_acc"] for row in rows]
    labels = [row["run_name"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("parameters")
    ax.set_ylabel("best_val_acc")
    ax.set_title("Accuracy / capacity trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)