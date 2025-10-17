"""Definições de modelos."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Cria modelo ResNet18 com cabeça personalizada."""

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    return model, optimizer, scheduler
