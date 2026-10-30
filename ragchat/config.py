"""Single source of configuration. Override via .env / environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Privacy: never let libraries phone home.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    ollama_host: str
    llm_model: str
    embedding_model: str
    embedding_download_hint: str
    chunk_size: int  # characters
    chunk_overlap: int  # characters
    top_k: int
    min_similarity: float  # cosine similarity, 0..1
    max_history_turns: int
    num_ctx: int
    temperature: float
    max_upload_mb: int
    tesseract_cmd: str  # "" = look on PATH
    ocr_language: str

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ.get
    s = Settings(
        data_dir=Path(env("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))),
        ollama_host=env("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=env("LLM_MODEL", "llama3.2:3b"),
        embedding_model=env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_download_hint=env("EMBEDDING_DOWNLOAD_HINT", "about 90 MB"),
        chunk_size=int(env("CHUNK_SIZE", "1000")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "150")),
        top_k=int(env("TOP_K", "5")),
        min_similarity=float(env("MIN_SIMILARITY", "0.25")),
        max_history_turns=int(env("MAX_HISTORY_TURNS", "4")),
        num_ctx=int(env("NUM_CTX", "4096")),
        temperature=float(env("TEMPERATURE", "0.1")),
        max_upload_mb=int(env("MAX_UPLOAD_MB", "100")),
        tesseract_cmd=env("TESSERACT_CMD", ""),
        ocr_language=env("OCR_LANGUAGE", "eng"),
    )
    return replace(s, **overrides) if overrides else s
