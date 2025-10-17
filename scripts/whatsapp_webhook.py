"""Webhook FastAPI para processar mensagens do WhatsApp."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException
from fastapi.responses import PlainTextResponse

from medimaging_ai.whatsapp import WhatsAppProcessor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="MedImaging AI - WhatsApp Webhook")
processor: WhatsAppProcessor | None = None


def _build_payload(**data: Any) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


@app.on_event("startup")
def startup_event() -> None:
    global processor
    config_path = os.getenv("WHATSAPP_CONFIG_PATH")
    checkpoint_path = os.getenv("WHATSAPP_CHECKPOINT_PATH")
    if not config_path or not checkpoint_path:
        LOGGER.warning("Variáveis WHATSAPP_CONFIG_PATH e WHATSAPP_CHECKPOINT_PATH são obrigatórias.")
        processor = None
        return

    db_path = Path(os.getenv("WHATSAPP_DB_PATH", "artifacts/whatsapp.sqlite"))
    retention_dir_env = os.getenv("WHATSAPP_RETENTION_DIR")
    retention_dir = Path(retention_dir_env) if retention_dir_env else None

    processor = WhatsAppProcessor(
        Path(config_path),
        Path(checkpoint_path),
        db_path,
        retention_dir=retention_dir,
    )
    LOGGER.info("Webhook inicializado com banco %s", db_path)


@app.get("/health", response_class=PlainTextResponse)
def healthcheck() -> str:
    return "ok"


@app.post("/webhook/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    MessageSid: str = Form(...),
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: str | None = Form(None),
    MediaContentType0: str | None = Form(None),
    **extras: str,
) -> str:
    if processor is None:
        raise HTTPException(status_code=503, detail="Processador não configurado")

    payload = _build_payload(
        MessageSid=MessageSid,
        From=From,
        Body=Body,
        NumMedia=NumMedia,
        MediaUrl0=MediaUrl0,
        MediaContentType0=MediaContentType0,
        **extras,
    )

    def _task() -> None:
        try:
            result = processor.process_request(payload)
            LOGGER.info(
                "Mensagem %s processada com status %s",
                payload.get("MessageSid"),
                result.get("status"),
            )
        except Exception:
            LOGGER.exception("Falha ao processar mensagem %s", payload.get("MessageSid"))

    background_tasks.add_task(_task)
    return "ACK"


if __name__ == "__main__":  # pragma: no cover - execução manual
    import uvicorn

    uvicorn.run("scripts.whatsapp_webhook:app", host="0.0.0.0", port=8000, reload=False)

