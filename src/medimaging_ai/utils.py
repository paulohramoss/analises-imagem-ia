"""Funções utilitárias gerais."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Define sementes para reprodutibilidade."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Salva métricas em arquivo CSV append-only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_training_status(path: Path, status: Dict[str, Any]) -> None:
    """Grava o estado atual do treinamento em um arquivo JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def load_training_status(path: Path) -> Optional[Dict[str, Any]]:
    """Carrega o arquivo de status do treinamento, se existir."""

    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
