"""API FastAPI para lidar com webhooks de provedores de mensagens."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Iterable, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, status

from medimaging_ai.config import ExperimentConfig, load_config
from medimaging_ai.inference import Predictor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MedImaging AI Webhook")

_META_SIGNATURE_HEADER = "x-hub-signature-256"
_MEDIA_TEMP_DIR = Path(os.getenv("WEBHOOK_MEDIA_DIR", Path(gettempdir()) / "medimaging_ai_webhook"))
if not _MEDIA_TEMP_DIR.exists():
    _MEDIA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(_MEDIA_TEMP_DIR, 0o700)
    except OSError:
        logger.debug("Não foi possível ajustar permissões do diretório temporário.")


class WebhookError(HTTPException):
    """Erro especializado para respostas HTTP do webhook."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error("Variável de ambiente %s não configurada.", var_name)
        raise WebhookError(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Variável {var_name} ausente")
    return value


@lru_cache(maxsize=1)
def _load_predictor() -> Predictor:
    config_path = Path(os.getenv("MODEL_CONFIG_PATH", "configs/default.yaml"))
    checkpoint_env = os.getenv("MODEL_CHECKPOINT_PATH")
    if not checkpoint_env:
        raise RuntimeError("MODEL_CHECKPOINT_PATH não configurado.")
    checkpoint_path = Path(checkpoint_env)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint {checkpoint_path} não encontrado.")

    cfg: ExperimentConfig = load_config(config_path)
    return Predictor(cfg, checkpoint_path)


def _verify_meta_signature(headers: Dict[str, str], body: bytes) -> None:
    signature_header = headers.get(_META_SIGNATURE_HEADER)
    if not signature_header:
        raise WebhookError(status.HTTP_401_UNAUTHORIZED, "Assinatura ausente.")

    try:
        scheme, received_signature = signature_header.split("=", 1)
    except ValueError as exc:
        raise WebhookError(status.HTTP_400_BAD_REQUEST, "Formato de assinatura inválido.") from exc

    if scheme != "sha256":
        raise WebhookError(status.HTTP_400_BAD_REQUEST, "Algoritmo de assinatura não suportado.")

    app_secret = _require_env("META_APP_SECRET").encode()
    expected_signature = hmac.new(app_secret, body, hashlib.sha256).hexdigest()

    if not secrets.compare_digest(received_signature, expected_signature):
        raise WebhookError(status.HTTP_401_UNAUTHORIZED, "Assinatura inválida.")


def _extract_first_image_message(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]] | None:
    entries: Iterable[Dict[str, Any]] = payload.get("entry", [])
    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            if not messages:
                continue
            for message in messages:
                if message.get("type") == "image" and "image" in message:
                    return value.get("metadata", {}).get("phone_number_id", ""), message
    return None


async def _resolve_media_url(media: Dict[str, Any], access_token: str, api_version: str) -> str:
    url = media.get("url") or media.get("link")
    media_id = media.get("id")
    if url:
        return url
    if not media_id:
        raise WebhookError(status.HTTP_400_BAD_REQUEST, "Mídia sem URL ou ID disponível.")

    graph_url = f"https://graph.facebook.com/{api_version}/{media_id}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(graph_url, params={"access_token": access_token})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception("Falha ao resolver URL da mídia: %s", exc)
            raise WebhookError(status.HTTP_502_BAD_GATEWAY, "Falha ao consultar API da Meta.") from exc
        data = response.json()
    resolved_url = data.get("url")
    if not resolved_url:
        raise WebhookError(status.HTTP_400_BAD_REQUEST, "Resposta da API sem URL de mídia.")
    return resolved_url


async def _download_media(media_url: str, access_token: str) -> Path:
    suffix = Path(media_url.split("?")[0]).suffix or ".jpg"
    file_path = _MEDIA_TEMP_DIR / f"{secrets.token_hex(16)}{suffix}"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(media_url, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception("Falha ao baixar mídia: %s", exc)
            raise WebhookError(status.HTTP_502_BAD_GATEWAY, "Não foi possível baixar a imagem.") from exc
    file_path.write_bytes(response.content)
    return file_path


def _format_medical_summary(scores: Dict[str, float]) -> str:
    if not scores:
        return (
            "Não foi possível gerar uma análise automática. "
            "Reenvie a imagem ou consulte um especialista."
        )

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    linhas = [f"- {label}: {prob * 100:.1f}%" for label, prob in ordered]
    texto = (
        "Análise automática da imagem recebida:\n"
        f"Probabilidade mais elevada para {top_label} ({top_score * 100:.1f}%).\n\n"
        "Distribuição estimada por classe:\n"
        + "\n".join(linhas)
        + "\n\nEste laudo é assistivo e não substitui avaliação médica presencial."
    )
    return texto


async def _send_meta_response(
    access_token: str,
    phone_number_id: str,
    destination: str,
    message_text: str,
    api_version: str,
) -> None:
    if not phone_number_id:
        phone_number_id = _require_env("META_PHONE_NUMBER_ID")

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": destination,
        "type": "text",
        "text": {"preview_url": False, "body": message_text},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception("Falha ao enviar mensagem de resposta: %s", exc)
            raise WebhookError(status.HTTP_502_BAD_GATEWAY, "Não foi possível enviar a resposta ao paciente.") from exc


@app.post("/webhook/{provider}")
async def receive_webhook(provider: str, request: Request) -> Dict[str, str]:
    if provider.lower() != "meta":
        raise WebhookError(status.HTTP_404_NOT_FOUND, "Provedor não suportado.")

    body = await request.body()
    _verify_meta_signature({k.lower(): v for k, v in request.headers.items()}, body)

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WebhookError(status.HTTP_400_BAD_REQUEST, "Payload JSON inválido.") from exc

    metadata = _extract_first_image_message(payload)
    if not metadata:
        logger.info("Nenhuma mensagem de imagem encontrada no payload.")
        return {"status": "ignorado"}

    phone_number_id, message = metadata
    image_info = message.get("image", {})
    access_token = _require_env("META_ACCESS_TOKEN")
    api_version = os.getenv("META_API_VERSION", "v17.0")

    media_url = await _resolve_media_url(image_info, access_token, api_version)
    image_path = await _download_media(media_url, access_token)

    try:
        predictor = _load_predictor()
        scores = predictor.predict(image_path)
    except Exception as exc:  # pragma: no cover - depende de modelo pré-treinado
        logger.exception("Falha na etapa de inferência: %s", exc)
        raise WebhookError(status.HTTP_500_INTERNAL_SERVER_ERROR, "Não foi possível processar a imagem.") from exc
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except TypeError:
            # Compatibilidade Python < 3.8
            if image_path.exists():
                image_path.unlink()

    response_text = _format_medical_summary(scores)

    await _send_meta_response(
        access_token=access_token,
        phone_number_id=phone_number_id,
        destination=message.get("from", ""),
        message_text=response_text,
        api_version=api_version,
    )

    return {"status": "processado"}


__all__ = ["app"]
