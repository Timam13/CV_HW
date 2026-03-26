from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .config import ModelConfig
from .utils import count_parameters


@dataclass(slots=True)
class MisclassifiedSample:
    image: torch.Tensor
    target: int
    prediction: int
    confidence: float


@dataclass(slots=True)
class ErrorAnalysis:
    confusion: np.ndarray
    per_class_accuracy: list[dict[str, float | str]]
    hardest_pairs: list[dict[str, float | str | int]]
    hardest_classes: list[dict[str, float | str | int]]

    def to_dict(self) -> dict:
        return {
            "per_class_accuracy": self.per_class_accuracy,
            "hardest_pairs": self.hardest_pairs,
            "hardest_classes": self.hardest_classes,
        }


@dataclass(slots=True)
class ModelProfile:
    params_total: int
    num_patches: int
    sequence_length: int
    token_dim: int
    attn_scores_per_block: int
    attn_memory_mb_fp32_per_block: float
    attn_flops_per_block: int
    attn_flops_total: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_model_profile(model: torch.nn.Module, cfg: ModelConfig) -> ModelProfile:
    params_total = count_parameters(model)
    n = cfg.sequence_length
    d = cfg.embed_dim
    attn_scores_per_block = cfg.num_heads * (n**2)
    attn_memory_mb = attn_scores_per_block * 4 / (1024**2)
    attn_flops_per_block = int(4 * n * (d**2) + 2 * (n**2) * d)
    attn_flops_total = attn_flops_per_block * cfg.depth
    return ModelProfile(
        params_total=params_total,
        num_patches=cfg.num_patches,
        sequence_length=n,
        token_dim=cfg.token_dim,
        attn_scores_per_block=attn_scores_per_block,
        attn_memory_mb_fp32_per_block=attn_memory_mb,
        attn_flops_per_block=attn_flops_per_block,
        attn_flops_total=attn_flops_total,
    )


def build_error_analysis(y_true: list[int], y_pred: list[int], class_names: list[str]) -> ErrorAnalysis:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = []
    for idx, name in enumerate(class_names):
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        accuracy = float(correct / total) if total else 0.0
        per_class.append({"class": name, "correct": correct, "total": total, "accuracy": accuracy})

    hardest_classes = sorted(per_class, key=lambda x: x["accuracy"])[: min(10, len(per_class))]

    pairs = []
    for i, true_name in enumerate(class_names):
        row = cm[i].copy()
        row[i] = 0
        if row.sum() == 0:
            continue
        j = int(row.argmax())
        pairs.append(
            {
                "true_class": true_name,
                "predicted_as": class_names[j],
                "count": int(row[j]),
                "error_rate": float(row[j] / cm[i].sum()),
            }
        )
    hardest_pairs = sorted(pairs, key=lambda x: (x["count"], x["error_rate"]), reverse=True)[: min(10, len(pairs))]

    return ErrorAnalysis(
        confusion=cm,
        per_class_accuracy=per_class,
        hardest_pairs=hardest_pairs,
        hardest_classes=hardest_classes,
    )