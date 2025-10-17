"""Carregamento e validação de configurações de experimento."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class PathsConfig:
    """Configurações relacionadas a caminhos."""

    train: Path
    val: Path
    test: Path
    num_workers: int = 4
    output_dir: Path = Path("artifacts")

    def ensure(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Hiperparâmetros do treinamento."""

    seed: int = 42
    image_size: int = 224
    batch_size: int = 16
    num_epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5


@dataclass
class TransformsConfig:
    """Configuração de transformações de dados."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    horizontal_flip: bool = True
    rotation_degrees: int = 15


@dataclass
class CheckpointConfig:
    """Configuração de salvamento de checkpoints."""

    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class ExperimentConfig:
    """Configuração completa do experimento."""

    paths: PathsConfig
    classes: List[str]
    train: TrainConfig = field(default_factory=TrainConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def _build_config(data: dict) -> ExperimentConfig:
    paths = PathsConfig(**data["paths"])
    classes = data["classes"]
    train_cfg = TrainConfig(**data.get("train", {}))
    transforms_cfg = TransformsConfig(**data.get("transforms", {}))
    checkpoint_cfg = CheckpointConfig(**data.get("checkpoint", {}))

    config = ExperimentConfig(
        paths=paths,
        classes=classes,
        train=train_cfg,
        transforms=transforms_cfg,
        checkpoint=checkpoint_cfg,
    )

    config.paths.ensure()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Carrega configuração YAML e converte em dataclasses."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _build_config(data)
