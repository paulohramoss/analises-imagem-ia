"""Loop de treinamento e validação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .utils import get_device, save_metrics, set_seed


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class Trainer:
    """Executa treinamento supervisionado padrão."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device()
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        set_seed(self.cfg.train.seed)
        self.model.to(self.device)

        best_metric = float("inf")
        patience_counter = 0
        best_path = Path(self.cfg.paths.output_dir) / "checkpoints" / "best.pt"

        for epoch in range(1, self.cfg.train.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            self.scheduler.step(val_loss)

            metrics = TrainState(epoch, train_loss, train_acc, val_loss, val_acc)
            save_metrics(metrics.__dict__, Path(self.cfg.paths.output_dir) / "logs" / "metrics.csv")

            if self.cfg.checkpoint.save_best_only:
                improved = (
                    val_loss < best_metric if self.cfg.checkpoint.mode == "min" else val_acc > best_metric
                )
                if improved:
                    best_metric = val_loss if self.cfg.checkpoint.mode == "min" else val_acc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.train.patience:
                        break
            else:
                torch.save(self.model.state_dict(), best_path)

        return best_path

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc="train" if train else "eval"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss /= total
        accuracy = correct / total if total > 0 else 0.0
        return epoch_loss, accuracy
