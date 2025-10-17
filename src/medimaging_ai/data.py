"""Utilitários de carregamento de dados."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import ExperimentConfig


def _build_transforms(cfg: ExperimentConfig) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std)

    train_transforms = [
        transforms.Resize((cfg.train.image_size, cfg.train.image_size)),
        transforms.ToTensor(),
        normalize,
    ]

    if cfg.transforms.horizontal_flip:
        train_transforms.insert(1, transforms.RandomHorizontalFlip())
    if cfg.transforms.rotation_degrees:
        train_transforms.insert(1, transforms.RandomRotation(cfg.transforms.rotation_degrees))

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((cfg.train.image_size, cfg.train.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {
        "train": transforms.Compose(train_transforms),
        "eval": eval_transforms,
    }


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """Cria dataloaders de treino, validação e teste."""

    tfms = _build_transforms(cfg)

    def _dataset(split: str, transform: transforms.Compose):
        path = Path(getattr(cfg.paths, split))
        if not path.exists():
            raise FileNotFoundError(
                f"Diretório de dados '{path}' não encontrado. Crie a estrutura antes de treinar."
            )
        return datasets.ImageFolder(path, transform=transform)

    train_ds = _dataset("train", tfms["train"])
    val_ds = _dataset("val", tfms["eval"])

    test_loader = None
    if Path(cfg.paths.test).exists():
        test_ds = _dataset("test", tfms["eval"])
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.paths.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.paths.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.paths.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
