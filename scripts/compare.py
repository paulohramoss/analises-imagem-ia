"""Script para comparar exames."""

from __future__ import annotations

import argparse
from pathlib import Path

from medimaging_ai.compare import compare_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparar exame do paciente com imagem de referência")
    parser.add_argument("--reference", type=Path, required=True, help="Imagem de referência considerada normal")
    parser.add_argument("--target", type=Path, required=True, help="Imagem do paciente")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_images(args.reference, args.target)
    print("Métricas de similaridade:")
    print(f"  SSIM: {result.ssim:.4f}")
    print(f"  Diferença absoluta média: {result.mean_abs_diff:.4f}")


if __name__ == "__main__":
    main()
