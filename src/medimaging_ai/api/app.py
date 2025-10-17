"""Aplicação FastAPI para monitoramento e inferência."""

from __future__ import annotations

import logging
import logging.config
import os
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger

from ..config import ExperimentConfig, load_config
from ..inference import Predictor
from ..settings import Settings, get_settings


def _configure_logging(level: str) -> None:
    level = level.upper()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": jsonlogger.JsonFormatter,
                    "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s",
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json",
                }
            },
            "loggers": {
                "medimaging_ai": {
                    "handlers": ["stdout"],
                    "level": level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["stdout"],
                    "level": level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["stdout"],
                    "level": level,
                    "propagate": False,
                },
            },
        }
    )


def _load_predictor(settings: Settings, config: ExperimentConfig | None) -> Predictor | None:
    if not config:
        return None

    checkpoint_path = settings.default_checkpoint_path()
    if checkpoint_path is None:
        return None

    if not checkpoint_path.exists():
        logging.getLogger("medimaging_ai.api").warning(
            "Checkpoint não encontrado",
            extra={"checkpoint_path": str(checkpoint_path)},
        )
        return None

    logging.getLogger("medimaging_ai.api").info(
        "Carregando modelo a partir do checkpoint",
        extra={"checkpoint_path": str(checkpoint_path)},
    )
    return Predictor(config, checkpoint_path)


def _ensure_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("medimaging_ai.api")

    app = FastAPI(title="MedImaging AI API", version=os.getenv("APP_VERSION", "0.1.0"))
    _ensure_checkpoint_dir(settings.model_checkpoint_dir)

    app.state.settings = settings
    app.state.config: ExperimentConfig | None = None
    app.state.predictor: Predictor | None = None

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "client": request.client.host if request.client else None,
            },
        )
        return response

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info(
            "Iniciando serviço",
            extra={
                "environment": settings.environment,
                "whatsapp_configured": bool(settings.whatsapp_token),
                "checkpoint_dir": str(settings.model_checkpoint_dir),
            },
        )

        config: ExperimentConfig | None = None
        if settings.model_config_path:
            try:
                config = load_config(settings.model_config_path)
                app.state.config = config
                logger.info(
                    "Configuração de modelo carregada",
                    extra={"config_path": str(settings.model_config_path)},
                )
            except FileNotFoundError:
                logger.warning(
                    "Arquivo de configuração não encontrado",
                    extra={"config_path": str(settings.model_config_path)},
                )
            except Exception as exc:  # pragma: no cover - logs para diagnóstico em produção
                logger.exception(
                    "Falha ao carregar configuração",
                    extra={"config_path": str(settings.model_config_path), "error": str(exc)},
                )

        if settings.load_model_on_startup:
            predictor = _load_predictor(settings, config)
            if predictor:
                app.state.predictor = predictor

    @app.get("/health", response_class=JSONResponse, tags=["monitoramento"])
    async def health(settings: Settings = Depends(get_settings)) -> Any:
        config_loaded = app.state.config is not None
        checkpoint_loaded = app.state.predictor is not None
        whatsapp_ready = bool(settings.whatsapp_token and settings.whatsapp_phone_number_id)

        return {
            "status": "ok",
            "environment": settings.environment,
            "whatsapp": {
                "configured": whatsapp_ready,
                "has_verify_token": bool(settings.whatsapp_verify_token),
            },
            "model": {
                "config_loaded": config_loaded,
                "checkpoint_loaded": checkpoint_loaded,
                "checkpoint_dir": str(settings.model_checkpoint_dir),
            },
        }

    return app


__all__ = ["create_app"]

