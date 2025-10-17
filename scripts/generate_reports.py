"""Geracao automatica de relatorios a partir das probabilidades do modelo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from medimaging_ai import Predictor, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerar relatorios (medico e paciente) a partir das probabilidades do modelo."
    )
    parser.add_argument("--config", type=Path, required=True, help="Arquivo de configuracao YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pt treinado.")
    parser.add_argument("--image", type=Path, required=True, help="Imagem do exame para avaliacao.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("relatorios"),
        help="Diretorio onde os relatorios serao salvos (padrao: relatorios/).",
    )
    parser.add_argument(
        "--lesao-label",
        type=str,
        default="lesao",
        help="Nome da classe que representa lesao para rotular o risco.",
    )
    parser.add_argument(
        "--alto-risco-threshold",
        type=float,
        default=0.7,
        help="Probabilidade a partir da qual o risco e considerado alto (padrao: 0.7).",
    )
    return parser.parse_args()


def construir_relatorio_medico(
    probs: Dict[str, float], lesao_label: str, alto_risco_threshold: float
) -> str:
    lesao_prob = probs.get(lesao_label, 0.0)
    top_label = max(probs, key=probs.get)
    risco = "alto" if lesao_prob >= alto_risco_threshold else "moderado" if lesao_prob >= 0.4 else "baixo"

    linhas = [
        "Relatorio tecnico - avaliacao automatizada",
        f"Classe com maior probabilidade: {top_label} ({probs[top_label]:.1%}).",
        f"Probabilidade de lesao: {lesao_prob:.1%}. Risco estimado: {risco}.",
        "Recomendacoes:",
        "- Correlacionar com quadro clinico e historia pregressa do paciente.",
        "- Comparar com exames anteriores (se disponiveis).",
        "- Considerar exames complementares caso o quadro clinico nao esteja explicado.",
        "",
        "Observacao: resultado gerado por algoritmo supervisionado. Confirmar achados em discussao clinica.",
    ]
    return "\n".join(linhas)


def construir_relatorio_paciente(
    probs: Dict[str, float], lesao_label: str, alto_risco_threshold: float
) -> str:
    lesao_prob = probs.get(lesao_label, 0.0)
    risco = "alto" if lesao_prob >= alto_risco_threshold else "moderado" if lesao_prob >= 0.4 else "baixo"

    linhas = [
        "Relatorio para paciente - linguagem simplificada",
        f"As imagens analisadas indicam {lesao_prob:.1%} de chance de haver lesao.",
        f"Isto significa que o nivel de atencao sugerido e {risco}.",
        "Proximo passo recomendado:",
        "- Agende uma consulta com o(a) seu(sua) medico(a) para discutir o resultado.",
        "- Leve este relatorio e informe qualquer sintoma ou mudanca recente.",
        "",
        "Importante: este resultado complementa, mas nao substitui, a avaliacao medica presencial.",
    ]
    return "\n".join(linhas)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(cfg, args.checkpoint)
    probs = predictor.predict(args.image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    relatorio_medico = construir_relatorio_medico(probs, args.lesao_label, args.alto_risco_threshold)
    relatorio_paciente = construir_relatorio_paciente(probs, args.lesao_label, args.alto_risco_threshold)

    medico_path = args.output_dir / "relatorio_medico.txt"
    paciente_path = args.output_dir / "relatorio_paciente.txt"

    medico_path.write_text(relatorio_medico, encoding="utf-8")
    paciente_path.write_text(relatorio_paciente, encoding="utf-8")

    print("Probabilidades por classe:")
    for label, prob in probs.items():
        print(f"  {label}: {prob:.4f}")
    print(f"\nArquivos gerados: {medico_path} e {paciente_path}")


if __name__ == "__main__":
    main()
