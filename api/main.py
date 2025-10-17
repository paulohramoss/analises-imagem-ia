"""API FastAPI para inferência de imagens médicas."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request

from medimaging_ai.config import load_config
from medimaging_ai.inference import Predictor

logger = logging.getLogger("medimaging_ai.api")
logging.basicConfig(level=logging.INFO)


class PredictionResponse(BaseModel):
    """Representa o retorno da análise de imagem."""

    classes: List[str]
    probabilities: Dict[str, float]


class ErrorResponse(BaseModel):
    """Resposta padronizada para erros."""

    error: str
    message: str


def get_predictor(request: Request) -> Predictor:
    """Obtém o predictor inicializado durante o startup."""

    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
    return predictor


def get_classes(request: Request) -> List[str]:
    """Retorna as classes definidas na configuração carregada."""

    classes: List[str] | None = getattr(request.app.state, "classes", None)
    if classes is None:
        raise HTTPException(status_code=503, detail="Configuração não carregada")
    return classes


app = FastAPI(title="MedImaging AI API", version="1.0.0")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Erro ao processar requisição"
    logger.warning("Erro HTTP %s: %s", exc.status_code, message)
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erro não tratado durante o processamento da requisição")
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Erro interno do servidor"})


@app.on_event("startup")
async def startup_event() -> None:
    """Carrega configuração e inicializa o modelo para inferência."""

    config_path = Path(os.getenv("MEDIMAGING_CONFIG", "configs/default.yaml"))
    if not config_path.exists():
        message = f"Arquivo de configuração não encontrado: {config_path}"
        logger.error(message)
        raise RuntimeError(message)

    checkpoint_env = os.getenv("MEDIMAGING_CHECKPOINT")
    if not checkpoint_env:
        message = "Variável de ambiente MEDIMAGING_CHECKPOINT não definida"
        logger.error(message)
        raise RuntimeError(message)

    checkpoint_path = Path(checkpoint_env)
    if not checkpoint_path.exists():
        message = f"Checkpoint não encontrado em: {checkpoint_path}"
        logger.error(message)
        raise RuntimeError(message)

    logger.info("Carregando configuração a partir de %s", config_path)
    cfg = load_config(config_path)
    logger.info("Inicializando Predictor com checkpoint %s", checkpoint_path)
    predictor = Predictor(cfg, checkpoint_path)

    app.state.predictor = predictor
    app.state.classes = cfg.classes
    logger.info("API pronta para receber requisições")


@app.post(
    "/analyze",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_image(
    file: UploadFile = File(..., description="Imagem a ser analisada"),
    predictor: Predictor = Depends(get_predictor),
    classes: List[str] = Depends(get_classes),
) -> PredictionResponse:
    """Recebe um arquivo de imagem e retorna as probabilidades previstas.

    **Request**: `multipart/form-data`

    - `file`: imagem compatível com PIL (`.png`, `.jpg`, `.jpeg`, etc.).

    **Response**: JSON contendo as chaves

    - `classes`: lista de rótulos retornados na mesma ordem em que foram configurados;
    - `probabilities`: dicionário com a probabilidade associada a cada classe.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido")

    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        try:
            shutil.copyfileobj(file.file, tmp)
        finally:
            file.file.close()

    try:
        predictions = predictor.predict(temp_path)
        ordered_probabilities = {label: float(predictions.get(label, 0.0)) for label in classes}
        logger.info("Análise concluída para arquivo %s", file.filename)
        return PredictionResponse(classes=classes, probabilities=ordered_probabilities)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha ao processar a imagem %s", file.filename)
        raise HTTPException(status_code=500, detail="Falha ao processar a imagem") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Não foi possível remover o arquivo temporário %s", temp_path)
