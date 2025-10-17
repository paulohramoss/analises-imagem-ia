#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${project_root}/.env" ]]; then
  echo "Arquivo .env não encontrado. Copie .env.example e ajuste as variáveis de ambiente." >&2
  exit 1
fi

echo "Iniciando deploy utilizando docker compose..."
(cd "${project_root}" && docker compose up -d --build)

echo "Deploy concluído. Serviços em execução:"
(cd "${project_root}" && docker compose ps)

