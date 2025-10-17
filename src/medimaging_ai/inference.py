"""Rotinas de inferência."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from .config import ExperimentConfig
from .models import build_model
from .utils import get_device


class Predictor:
    """Carrega um modelo treinado e realiza previsões em imagens individuais."""

    def __init__(self, cfg: ExperimentConfig, checkpoint_path: Path) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.model, _, _ = build_model(cfg.num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.train.image_size, cfg.train.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
            ]
        )

    def predict(self, image_path: Path) -> Dict[str, float]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {label: float(prob) for label, prob in zip(self.cfg.classes, probs)}
