"""Script de inferência para uma imagem."""

from __future__ import annotations

import argparse
from pathlib import Path

from medimaging_ai import Predictor, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realizar predição em um exame")
    parser.add_argument("--config", type=Path, required=True, help="Arquivo de configuração")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Arquivo .pt com pesos do modelo")
    parser.add_argument("--image", type=Path, required=True, help="Imagem do exame a ser analisado")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    predictor = Predictor(cfg, args.checkpoint)
    probs = predictor.predict(args.image)

    print("Probabilidades por classe:")
    for label, prob in probs.items():
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
