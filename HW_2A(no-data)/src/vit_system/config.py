from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path: float = 0.0
    cls_token: bool = True
    pooling: str = "cls"

    @property
    def patch_grid(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_grid**2

    @property
    def sequence_length(self) -> int:
        return self.num_patches + int(self.cls_token)

    @property
    def token_dim(self) -> int:
        return self.embed_dim


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    num_workers: int = 4
    grad_clip_norm: float | None = 1.0
    amp: bool = True
    max_misclassified: int = 25


@dataclass(slots=True)
class DataConfig:
    dataset: str = "imagefolder"
    data_root: str = "data"
    train_dir: str | None = None
    val_dir: str | None = None
    download: bool = False
    val_ratio: float = 0.2
    image_size: int = 32
    seed: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str = "vit_run"
    output_dir: str = "outputs"
    device: str = "auto"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, ModelConfig] = {
    "patch4_tiny_cls": ModelConfig(patch_size=4, embed_dim=192, depth=4, num_heads=3, cls_token=True, pooling="cls"),
    "patch8_tiny_cls": ModelConfig(patch_size=8, embed_dim=192, depth=4, num_heads=3, cls_token=True, pooling="cls"),
    "patch8_tiny_mean": ModelConfig(patch_size=8, embed_dim=192, depth=4, num_heads=3, cls_token=False, pooling="mean"),
    "patch8_deep_cls": ModelConfig(patch_size=8, embed_dim=192, depth=6, num_heads=3, cls_token=True, pooling="cls"),
    "patch8_wide_cls": ModelConfig(patch_size=8, embed_dim=256, depth=4, num_heads=4, cls_token=True, pooling="cls"),
    "patch4_wide_cls": ModelConfig(patch_size=4, embed_dim=256, depth=6, num_heads=4, cls_token=True, pooling="cls"),

    "pets_patch16_tiny_cls": ModelConfig(
        image_size=128,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=True,
        pooling="cls",
    ),
    "pets_patch16_tiny_mean": ModelConfig(
        image_size=128,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=False,
        pooling="mean",
    ),
    "pets_patch8_tiny_cls": ModelConfig(
        image_size=128,
        patch_size=8,
        embed_dim=192,
        depth=4,
        num_heads=3,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=True,
        pooling="cls",
    ),
    "pets_patch16_deep_cls": ModelConfig(
        image_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=True,
        pooling="cls",
    ),
    "cifar_patch4_strong_cls": ModelConfig(
        image_size=32,
        patch_size=4,
        embed_dim=256,
        depth=6,
        num_heads=4,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=True,
        pooling="cls",
    ),
    "cifar_patch4_strong_mean": ModelConfig(
        image_size=32,
        patch_size=4,
        embed_dim=256,
        depth=6,
        num_heads=4,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path=0.1,
        cls_token=False,
        pooling="mean",
    ),
}


SWEEPS: dict[str, list[str]] = {
    "patch_size": ["patch4_tiny_cls", "patch8_tiny_cls"],
    "pooling": ["patch8_tiny_cls", "patch8_tiny_mean"],
    "depth": ["patch8_tiny_cls", "patch8_deep_cls"],
    "embedding": ["patch8_tiny_cls", "patch8_wide_cls"],
    "cifar_pooling": ["cifar_patch4_strong_cls", "cifar_patch4_strong_mean"],
    "pets_patch_size": ["pets_patch16_tiny_cls", "pets_patch8_tiny_cls"],
    "pets_pooling": ["pets_patch16_tiny_cls", "pets_patch16_tiny_mean"],
    "pets_depth": ["pets_patch16_tiny_cls", "pets_patch16_deep_cls"],
}


def build_model_config(preset: str, **overrides: Any) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset: {preset}")
    data = asdict(PRESETS[preset]).copy()
    data.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**data)