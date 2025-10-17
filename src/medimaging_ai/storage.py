"""Gerenciamento centralizado de arquivos temporários."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional


class TemporaryFileHandle:
    """Representa um arquivo temporário controlado pelo ``TemporaryFileManager``."""

    def __init__(self, path: Path, manager: "TemporaryFileManager") -> None:
        self.path = path
        self._manager = manager
        self._persist_requested = False
        self._custom_name: Optional[str] = None
        self._persisted_uri: Optional[str] = None

    def persist(self, name: Optional[str] = None) -> None:
        """Solicita a retenção do arquivo ao final do contexto.

        Parameters
        ----------
        name:
            Nome opcional para o arquivo persistido. Caso não seja informado,
            o nome temporário será reutilizado.
        """

        if not self._manager.persistence_available:
            raise RuntimeError("Nenhum backend de retenção configurado.")
        self._persist_requested = True
        self._custom_name = name

    @property
    def retention_path(self) -> Optional[Path]:
        if not self._persist_requested or self._manager.retention_dir is None:
            return None
        name = self._custom_name or self.path.name
        return self._manager.retention_dir / name

    @property
    def persisted_uri(self) -> Optional[str]:
        return self._persisted_uri

    def _set_persisted_uri(self, uri: Optional[str]) -> None:
        self._persisted_uri = uri

    def _should_persist(self) -> bool:
        return self._persist_requested

    def _custom_filename(self) -> Optional[str]:
        return self._custom_name


class TemporaryFileManager:
    """Cria e limpa arquivos temporários de forma automática."""

    def __init__(
        self,
        *,
        base_dir: Optional[Path] = None,
        retention_dir: Optional[Path] = None,
        persist_callback: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "medimaging_ai"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._retention_dir = retention_dir
        if self._retention_dir is not None:
            self._retention_dir.mkdir(parents=True, exist_ok=True)
        self._persist_callback = persist_callback

    @property
    def retention_dir(self) -> Optional[Path]:
        return self._retention_dir

    @property
    def persistence_available(self) -> bool:
        return self._persist_callback is not None or self._retention_dir is not None

    @contextmanager
    def reserve_path(self, *, suffix: str = "") -> Iterator[TemporaryFileHandle]:
        """Fornece um arquivo temporário e garante a limpeza ao final do bloco."""

        fd, name = tempfile.mkstemp(prefix="medimg_", suffix=suffix, dir=str(self.base_dir))
        os.close(fd)
        handle = TemporaryFileHandle(Path(name), self)
        try:
            yield handle
        finally:
            uri = self._finalize(handle.path, handle._should_persist(), handle._custom_filename())
            handle._set_persisted_uri(uri)

    def _finalize(self, path: Path, persist: bool, custom_name: Optional[str]) -> Optional[str]:
        if persist:
            if self._persist_callback is not None:
                uri = self._persist_callback(path)
                if uri:
                    return uri
            if self._retention_dir is not None:
                dest_name = custom_name or path.name
                destination = self._retention_dir / dest_name
                shutil.move(str(path), destination)
                return str(destination)
        path.unlink(missing_ok=True)
        return None

