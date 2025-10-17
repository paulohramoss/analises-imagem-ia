"""Interface Streamlit para treinamento e análise de modelos de imagens médicas."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from medimaging_ai.compare import compare_images
from medimaging_ai.config import ExperimentConfig, load_config
from medimaging_ai.data import build_dataloaders
from medimaging_ai.inference import Predictor
from medimaging_ai.models import build_model
from medimaging_ai.trainer import Trainer
from medimaging_ai.utils import load_training_status
from medimaging_ai.storage import TemporaryFileManager


st.set_page_config(page_title="MedImaging AI", layout="wide")


temp_manager = TemporaryFileManager()


def _to_serializable(data: Any) -> Any:
    """Converte dataclasses e objetos Path em dicionários serializáveis."""

    # is_dataclass() returns True for both dataclass instances and dataclass types;
    # ensure we only call asdict on instances (not on the class/type itself).
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    if isinstance(data, dict):
        return {key: _to_serializable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_serializable(item) for item in data]
    if isinstance(data, Path):
        return str(data)
    return data


def _ensure_session_defaults() -> None:
    if "config_path" not in st.session_state:
        default_cfg = Path("configs/default.yaml")
        st.session_state["config_path"] = str(default_cfg) if default_cfg.exists() else ""
    st.session_state.setdefault("cfg", None)
    st.session_state.setdefault("last_checkpoint", "")


def _load_cfg(path: str) -> ExperimentConfig:
    cfg = load_config(path)
    cfg.paths.ensure()
    return cfg


def _training_status(cfg: ExperimentConfig) -> Dict[str, Any] | None:
    status_path = Path(cfg.paths.output_dir) / "logs" / "status.json"
    return load_training_status(status_path)


def _run_training(cfg: ExperimentConfig) -> Path:
    train_loader, val_loader, _ = build_dataloaders(cfg)
    model, optimizer, scheduler = build_model(cfg.num_classes)
    trainer = Trainer(cfg, model, optimizer, scheduler)
    return trainer.fit(train_loader, val_loader)


@contextmanager
def _temporary_upload(upload, suffix: str):
    with temp_manager.reserve_path(suffix=suffix) as handle:
        handle.path.write_bytes(upload.getvalue())
        yield handle


def _checkpoint_options(cfg: ExperimentConfig) -> list[str]:
    ckpt_dir = Path(cfg.paths.output_dir) / "checkpoints"
    return sorted(str(path) for path in ckpt_dir.glob("*.pt"))


def _render_config_tab() -> None:
    st.header("Configuração do experimento")
    config_path = st.text_input("Arquivo de configuração", value=st.session_state.get("config_path", ""))
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Carregar configuração"):
            if not config_path:
                st.warning("Informe o caminho para um arquivo YAML de configuração.")
            else:
                try:
                    cfg = _load_cfg(config_path)
                    st.session_state["cfg"] = cfg
                    st.session_state["config_path"] = config_path
                    st.success("Configuração carregada com sucesso.")
                except Exception as exc:  # pragma: no cover - interface interativa
                    st.error(f"Falha ao carregar configuração: {exc}")
    with col2:
        if st.button("Limpar configuração"):
            st.session_state["cfg"] = None
            st.session_state["config_path"] = ""

    cfg = st.session_state.get("cfg")
    if cfg:
        st.subheader("Resumo")
        st.json(_to_serializable(cfg))
    else:
        st.info("Carregue uma configuração para habilitar as demais abas.")


def _render_training_tab() -> None:
    st.header("Treinamento")
    cfg: ExperimentConfig | None = st.session_state.get("cfg")
    if not cfg:
        st.info("Nenhuma configuração carregada.")
        return

    status = _training_status(cfg)
    running = status is not None and status.get("state") == "running"

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Iniciar treinamento", disabled=running):
            try:
                with st.spinner("Executando treinamento..."):
                    best_path = _run_training(cfg)
                st.session_state["last_checkpoint"] = str(best_path)
                st.success(f"Treinamento concluído. Checkpoint salvo em {best_path}")
            except Exception as exc:  # pragma: no cover - interface interativa
                st.error(f"Falha durante o treinamento: {exc}")
    with col2:
        if st.button("Atualizar status"):
            try:
                rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
                if callable(rerun):
                    rerun()
                else:
                    st.info("Atualize a página para ver o status mais recente.")
            except Exception:
                st.info("Atualize a página para ver o status mais recente.")

    if status:
        st.subheader("Status do treinamento")
        st.json(status)
    else:
        st.info("Nenhum histórico de treinamento disponível.")

    metrics_path = Path(cfg.paths.output_dir) / "logs" / "metrics.csv"
    if metrics_path.exists():
        st.subheader("Histórico de métricas")
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, width="stretch")


def _render_inference_tab() -> None:
    st.header("Inferência")
    cfg: ExperimentConfig | None = st.session_state.get("cfg")
    if not cfg:
        st.info("Carregue uma configuração antes de realizar inferência.")
        return

    checkpoints = _checkpoint_options(cfg)
    last_ckpt = st.session_state.get("last_checkpoint", "")
    default_index = checkpoints.index(last_ckpt) if last_ckpt in checkpoints else (len(checkpoints) - 1 if checkpoints else 0)
    checkpoint = st.selectbox("Checkpoint", checkpoints, index=default_index if checkpoints else 0)

    uploaded_image = st.file_uploader("Imagem para análise", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if st.button("Executar inferência", disabled=not checkpoint or uploaded_image is None):
        if not checkpoint:
            st.warning("Nenhum checkpoint encontrado. Treine o modelo primeiro.")
            return
        if uploaded_image is None:
            st.warning("Envie uma imagem para análise.")
            return

        with _temporary_upload(uploaded_image, suffix=Path(uploaded_image.name).suffix or ".png") as temp_handle:
            temp_path = temp_handle.path
            try:
                predictor = Predictor(cfg, Path(checkpoint))
                predictions = predictor.predict(temp_path)
            except Exception as exc:  # pragma: no cover - interface interativa
                st.error(f"Erro ao executar a inferência: {exc}")
                return

        st.session_state["last_checkpoint"] = checkpoint
        st.subheader("Probabilidades por classe")
        df = pd.DataFrame(
            {
                "Classe": list(predictions.keys()),
                "Probabilidade": list(predictions.values()),
            }
        )
        st.dataframe(df, width="stretch")
        try:
            fig = px.bar(df, x="Classe", y="Probabilidade", range_y=[0, 1])
            st.plotly_chart(fig, width="stretch")
        except Exception:  # pragma: no cover - visualização opcional
            pass


def _render_comparison_tab() -> None:
    st.header("Comparação de exames")
    reference_file = st.file_uploader("Imagem de referência", key="reference", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    target_file = st.file_uploader("Imagem do paciente", key="target", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

    if st.button("Comparar imagens", disabled=reference_file is None or target_file is None):
        if not reference_file or not target_file:
            st.warning("Envie as duas imagens para realizar a comparação.")
            return

        with _temporary_upload(
            reference_file, suffix=Path(reference_file.name).suffix or ".png"
        ) as ref_handle, _temporary_upload(
            target_file, suffix=Path(target_file.name).suffix or ".png"
        ) as tgt_handle:
            try:
                result = compare_images(ref_handle.path, tgt_handle.path)
            except Exception as exc:  # pragma: no cover - interface interativa
                st.error(f"Erro ao comparar imagens: {exc}")
                return

        st.metric("SSIM", f"{result.ssim:.4f}")
        st.metric("Diferença absoluta média", f"{result.mean_abs_diff:.4f}")


def main() -> None:
    st.title("Painel MedImaging AI")
    st.write("Acompanhe o treinamento, execute inferências e compare exames em um único lugar.")

    _ensure_session_defaults()

    tabs = st.tabs(["Configuração", "Treinamento", "Inferência", "Comparação"])

    with tabs[0]:
        _render_config_tab()
    with tabs[1]:
        _render_training_tab()
    with tabs[2]:
        _render_inference_tab()
    with tabs[3]:
        _render_comparison_tab()


if __name__ == "__main__":
    main()
