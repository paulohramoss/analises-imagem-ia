"""Camada de persistência baseada em SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS analises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    whatsapp_number TEXT NOT NULL,
    body TEXT,
    media_url TEXT,
    media_content_type TEXT,
    metadata TEXT,
    scores TEXT,
    status TEXT NOT NULL,
    storage_uri TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: Path) -> None:
    """Garante que o banco e a tabela principal existam."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(SCHEMA)
        connection.commit()


def insert_analysis(
    db_path: Path,
    *,
    message_id: str,
    whatsapp_number: str,
    body: Optional[str],
    media_url: Optional[str],
    media_content_type: Optional[str],
    metadata: Dict[str, Any],
    scores: Dict[str, float],
    status: str,
    storage_uri: Optional[str],
    error_message: Optional[str] = None,
) -> int:
    """Insere ou atualiza um registro na tabela ``analises``.

    Parameters
    ----------
    db_path:
        Caminho do arquivo SQLite.
    message_id:
        Identificador único fornecido pelo provedor (ex.: ``MessageSid`` da Twilio).
    whatsapp_number:
        Número de origem do WhatsApp (incluindo código do país).
    body:
        Texto enviado pelo usuário.
    media_url:
        URL original do arquivo de mídia recebida.
    media_content_type:
        MIME type informado pelo provedor.
    metadata:
        Demais campos recebidos na requisição.
    scores:
        Probabilidades retornadas pelo modelo.
    status:
        Estado do processamento (``processed``, ``ignored``, ``error`` etc.).
    storage_uri:
        Caminho/URI de retenção do arquivo processado, quando configurado.
    error_message:
        Detalhes adicionais em caso de falha.

    Returns
    -------
    int
        ``rowid`` do registro inserido/atualizado.
    """

    metadata_json = json.dumps(metadata, ensure_ascii=False)
    scores_json = json.dumps(scores, ensure_ascii=False)

    sql = """
    INSERT INTO analises (
        message_id,
        whatsapp_number,
        body,
        media_url,
        media_content_type,
        metadata,
        scores,
        status,
        storage_uri,
        error_message
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(message_id) DO UPDATE SET
        whatsapp_number=excluded.whatsapp_number,
        body=excluded.body,
        media_url=excluded.media_url,
        media_content_type=excluded.media_content_type,
        metadata=excluded.metadata,
        scores=excluded.scores,
        status=excluded.status,
        storage_uri=excluded.storage_uri,
        error_message=excluded.error_message;
    """

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            sql,
            (
                message_id,
                whatsapp_number,
                body,
                media_url,
                media_content_type,
                metadata_json,
                scores_json,
                status,
                storage_uri,
                error_message,
            ),
        )
        connection.commit()
        return cursor.lastrowid

