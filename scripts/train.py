"""Script de treinamento."""

from __future__ import annotations

import argparse
from pathlib import Path

from medimaging_ai import Trainer, build_dataloaders, build_model, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar modelo de classificação de exames")
    parser.add_argument("--config", type=Path, required=True, help="Caminho para arquivo de configuração YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model, optimizer, scheduler = build_model(cfg.num_classes)

    trainer = Trainer(cfg, model, optimizer, scheduler)
    best_path = trainer.fit(train_loader, val_loader)
    print(f"Treinamento concluído. Melhor modelo salvo em: {best_path}")


if __name__ == "__main__":
    main()
