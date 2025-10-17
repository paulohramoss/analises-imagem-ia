"""Ferramentas para comparar exames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass
class ComparisonResult:
    ssim: float
    mean_abs_diff: float


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.float32)


def compare_images(reference_path: Path, target_path: Path) -> ComparisonResult:
    """Compara duas imagens e retorna métricas simples de similaridade."""

    reference = _load_image(reference_path)
    target = _load_image(target_path)

    if reference.shape != target.shape:
        raise ValueError("As imagens devem ter o mesmo tamanho para comparação direta.")

    ssim_value = structural_similarity(reference, target, data_range=target.max() - target.min())
    mean_abs_diff = float(np.mean(np.abs(reference - target)))

    return ComparisonResult(ssim=ssim_value, mean_abs_diff=mean_abs_diff)
