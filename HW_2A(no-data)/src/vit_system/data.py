from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, random_split

from .config import DataConfig, TrainingConfig
from .utils import DEFAULT_MEAN, DEFAULT_STD


@dataclass(slots=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]
    num_classes: int
    train_size: int
    val_size: int
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable | None = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


class IndexedSubset(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        return image, label, index


class_names_cache: dict[str, list[str]] = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
}


def _torchvision_import():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for data loading. Install matching torch/torchvision versions from requirements.txt"
        ) from exc
    return datasets, transforms


def _build_transforms(dataset: str, image_size: int):
    _, transforms = _torchvision_import()

    if dataset in {"cifar10", "cifar100"}:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_transform, eval_transform

    interpolation = transforms.InterpolationMode.BICUBIC

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=interpolation),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15), interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    return train_transform, eval_transform

def _split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _resolve_class_names(dataset_name: str, base_dataset: Dataset) -> list[str]:
    if hasattr(base_dataset, "classes"):
        return [str(x) for x in base_dataset.classes]
    if dataset_name == "flowers102":
        return [f"class_{i}" for i in range(102)]
    if dataset_name == "oxfordiiitpet":
        labels = sorted(set(int(base_dataset[i][1]) for i in range(len(base_dataset))))
        return [f"pet_{i}" for i in labels]
    if dataset_name in class_names_cache:
        return class_names_cache[dataset_name]
    labels = sorted(set(int(base_dataset[i][1]) for i in range(len(base_dataset))))
    return [str(i) for i in labels]


def _build_imagefolder(cfg: DataConfig, train_transform, eval_transform):
    datasets, _ = _torchvision_import()
    if cfg.train_dir is None:
        raise ValueError("train_dir is required for dataset=imagefolder")
    train_root = datasets.ImageFolder(cfg.train_dir)
    class_names = [str(x) for x in train_root.classes]
    if cfg.val_dir:
        train_dataset = IndexedDataset(TransformDataset(train_root, train_transform))
        val_dataset = IndexedDataset(TransformDataset(datasets.ImageFolder(cfg.val_dir), eval_transform))
        return train_dataset, val_dataset, class_names
    train_subset, val_subset = _split_train_val(train_root, cfg.val_ratio, cfg.seed)
    train_dataset = IndexedSubset(Subset(TransformDataset(train_root, train_transform), train_subset.indices))
    val_dataset = IndexedSubset(Subset(TransformDataset(train_root, eval_transform), val_subset.indices))
    return train_dataset, val_dataset, class_names


def _build_cifar(name: str, cfg: DataConfig, train_transform, eval_transform):
    datasets, _ = _torchvision_import()
    dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    base_train = dataset_cls(root=cfg.data_root, train=True, transform=train_transform, download=cfg.download)
    base_eval = dataset_cls(root=cfg.data_root, train=True, transform=eval_transform, download=cfg.download)
    train_subset, val_subset = _split_train_val(base_train, cfg.val_ratio, cfg.seed)
    train_dataset = IndexedSubset(train_subset)
    val_dataset = IndexedSubset(Subset(base_eval, val_subset.indices))
    class_names = [str(x) for x in base_train.classes]
    return train_dataset, val_dataset, class_names


def _build_flowers(cfg: DataConfig, train_transform, eval_transform):
    datasets, _ = _torchvision_import()
    train_dataset = IndexedDataset(
        datasets.Flowers102(root=cfg.data_root, split="train", transform=train_transform, download=cfg.download)
    )
    val_dataset = IndexedDataset(
        datasets.Flowers102(root=cfg.data_root, split="val", transform=eval_transform, download=cfg.download)
    )
    class_names = [f"flower_{i}" for i in range(102)]
    return train_dataset, val_dataset, class_names


def _build_pets(cfg: DataConfig, train_transform, eval_transform):
    datasets, _ = _torchvision_import()
    train_dataset = IndexedDataset(
        datasets.OxfordIIITPet(root=cfg.data_root, split="trainval", target_types="category", transform=train_transform, download=cfg.download)
    )
    val_dataset = IndexedDataset(
        datasets.OxfordIIITPet(root=cfg.data_root, split="test", target_types="category", transform=eval_transform, download=cfg.download)
    )
    base = datasets.OxfordIIITPet(root=cfg.data_root, split="trainval", target_types="category", download=cfg.download)
    class_names = _resolve_class_names("oxfordiiitpet", base)
    return train_dataset, val_dataset, class_names


def build_dataloaders(cfg: DataConfig, training: TrainingConfig) -> DatasetBundle:
    train_transform, eval_transform = _build_transforms(cfg.dataset, cfg.image_size)

    if cfg.dataset == "imagefolder":
        train_dataset, val_dataset, class_names = _build_imagefolder(cfg, train_transform, eval_transform)
    elif cfg.dataset in {"cifar10", "cifar100"}:
        train_dataset, val_dataset, class_names = _build_cifar(cfg.dataset, cfg, train_transform, eval_transform)
    elif cfg.dataset == "flowers102":
        train_dataset, val_dataset, class_names = _build_flowers(cfg, train_transform, eval_transform)
    elif cfg.dataset == "oxfordiiitpet":
        train_dataset, val_dataset, class_names = _build_pets(cfg, train_transform, eval_transform)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=training.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        num_classes=len(class_names),
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )