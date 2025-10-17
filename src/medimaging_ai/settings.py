"""Definições e utilidades para configuração baseada em variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(slots=True)
class Settings:
    """Configurações carregadas a partir de variáveis de ambiente."""

    environment: str
    whatsapp_token: str
    whatsapp_phone_number_id: str
    whatsapp_verify_token: str
    model_config_path: Optional[Path]
    model_checkpoint_dir: Path
    model_default_checkpoint: str
    model_device: str
    log_level: str
    load_model_on_startup: bool

    @classmethod
    def from_env(cls) -> "Settings":
        config_path = os.getenv("MODEL_CONFIG_PATH")
        checkpoint_dir = Path(os.getenv("MODEL_CHECKPOINT_DIR", "/checkpoints"))

        return cls(
            environment=os.getenv("ENVIRONMENT", "production"),
            whatsapp_token=os.getenv("WHATSAPP_TOKEN", ""),
            whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
            whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", ""),
            model_config_path=Path(config_path) if config_path else None,
            model_checkpoint_dir=checkpoint_dir,
            model_default_checkpoint=os.getenv("MODEL_DEFAULT_CHECKPOINT", ""),
            model_device=os.getenv("MODEL_DEVICE", "cpu"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            load_model_on_startup=_as_bool(os.getenv("MODEL_LOAD_ON_STARTUP"), default=True),
        )

    def default_checkpoint_path(self) -> Optional[Path]:
        """Retorna o caminho absoluto para o checkpoint padrão, se configurado."""

        if not self.model_default_checkpoint:
            return None

        checkpoint_path = Path(self.model_default_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.model_checkpoint_dir / checkpoint_path
        return checkpoint_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Obtém uma instância cacheada das configurações da aplicação."""

    return Settings.from_env()


__all__ = ["Settings", "get_settings"]

