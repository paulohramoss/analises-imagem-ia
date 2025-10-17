"""Processamento de requisições provenientes do WhatsApp."""

from __future__ import annotations

import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from .config import ExperimentConfig, load_config
from .inference import Predictor
from .persistence import init_db, insert_analysis
from .storage import TemporaryFileManager

LOGGER = logging.getLogger(__name__)


@dataclass
class WhatsAppMessage:
    """Representa a carga relevante recebida do provedor de WhatsApp."""

    provider_message_id: str
    from_number: str
    body: str
    media_url: Optional[str]
    media_content_type: Optional[str]
    metadata: Dict[str, Any]


class WhatsAppProcessor:
    """Orquestra o download, inferência e registro de mensagens de WhatsApp."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        db_path: Path,
        *,
        retention_dir: Optional[Path] = None,
    ) -> None:
        self.cfg: ExperimentConfig = load_config(config_path)
        self.predictor = Predictor(self.cfg, checkpoint_path)
        self.db_path = db_path
        init_db(db_path)

        self.media_token = os.getenv("WHATSAPP_PROVIDER_TOKEN", "")
        self.media_base_url = os.getenv("WHATSAPP_PROVIDER_BASE_URL", "")
        self.retain_media = os.getenv("WHATSAPP_RETAIN_MEDIA", "false").lower() in {"1", "true", "yes"}
        self.temp_manager = TemporaryFileManager(retention_dir=retention_dir)

    def process_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Processa e registra uma requisição de webhook do WhatsApp."""

        message = self._parse_payload(payload)
        status = "received"
        scores: Dict[str, float] = {}
        storage_uri: Optional[str] = None
        error_message: Optional[str] = None

        if not message.media_url:
            status = "ignored"
            LOGGER.info("Mensagem %s sem mídia foi registrada, porém não analisada.", message.provider_message_id)
            self._persist(message, scores, status, storage_uri, error_message)
            return {"status": status, "scores": scores}

        try:
            media_bytes = self._download_media(message.media_url)
            suffix = self._guess_suffix(message.media_content_type)
            with self.temp_manager.reserve_path(suffix=suffix) as temp_handle:
                temp_handle.path.write_bytes(media_bytes)
                scores = self.predictor.predict(temp_handle.path)
                if self.retain_media:
                    safe_name = f"{message.provider_message_id}{suffix}"
                    try:
                        temp_handle.persist(safe_name)
                    except RuntimeError:
                        LOGGER.warning(
                            "Retenção de mídia solicitada, mas nenhum backend foi configurado."
                        )
            storage_uri = temp_handle.persisted_uri
            status = "processed"
        except Exception as exc:  # pragma: no cover - integrações externas
            status = "error"
            error_message = str(exc)
            LOGGER.exception("Falha ao processar mensagem %s", message.provider_message_id)

        self._persist(message, scores, status, storage_uri, error_message)
        response: Dict[str, Any] = {"status": status, "scores": scores}
        if storage_uri:
            response["storage_uri"] = storage_uri
        if error_message:
            response["error"] = error_message
        return response

    def _parse_payload(self, payload: Dict[str, Any]) -> WhatsAppMessage:
        message_id = (
            payload.get("MessageSid")
            or payload.get("message_id")
            or payload.get("Id")
        )
        from_number = payload.get("From") or payload.get("from")
        if not message_id or not from_number:
            raise ValueError("Payload inválido: campos obrigatórios ausentes.")

        media_url = payload.get("MediaUrl0") or payload.get("media_url")
        media_content_type = payload.get("MediaContentType0") or payload.get("media_content_type")
        body = payload.get("Body") or payload.get("body") or ""
        ignored_keys = {
            "MessageSid",
            "message_id",
            "Id",
            "From",
            "from",
            "Body",
            "body",
            "MediaUrl0",
            "media_url",
            "MediaContentType0",
            "media_content_type",
        }
        metadata = {key: str(value) for key, value in payload.items() if key not in ignored_keys}
        return WhatsAppMessage(
            provider_message_id=str(message_id),
            from_number=str(from_number),
            body=str(body),
            media_url=str(media_url) if media_url else None,
            media_content_type=str(media_content_type) if media_content_type else None,
            metadata=metadata,
        )

    def _download_media(self, media_url: str) -> bytes:
        if self.media_base_url and not media_url.startswith("http"):
            url = urljoin(self.media_base_url.rstrip("/"), media_url.lstrip("/"))
        else:
            url = media_url

        headers = {"Authorization": f"Bearer {self.media_token}"} if self.media_token else {}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _guess_suffix(content_type: Optional[str]) -> str:
        if not content_type:
            return ".bin"
        suffix = mimetypes.guess_extension(content_type)
        return suffix or ".bin"

    def _persist(
        self,
        message: WhatsAppMessage,
        scores: Dict[str, float],
        status: str,
        storage_uri: Optional[str],
        error_message: Optional[str],
    ) -> None:
        insert_analysis(
            self.db_path,
            message_id=message.provider_message_id,
            whatsapp_number=message.from_number,
            body=message.body,
            media_url=message.media_url,
            media_content_type=message.media_content_type,
            metadata=message.metadata,
            scores=scores,
            status=status,
            storage_uri=storage_uri,
            error_message=error_message,
        )

