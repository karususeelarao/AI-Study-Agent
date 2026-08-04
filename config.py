"""
config.py
Configuration for Gemini-based AI Study Assistant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


class ConfigError(Exception):
    pass


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer")


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a float")


@dataclass(frozen=True)
class Settings:
    # Base
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_temperature: float = 0.4
    gemini_max_tokens: int = 2048

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector DB
    vector_db_dir: Path = field(default_factory=lambda: Path("vector_db"))
    faiss_index_name: str = "study_assistant_index"

    # SQLite
    sqlite_db_path: Path = field(default_factory=lambda: Path("database/study_assistant.db"))

    # Storage
    upload_dir: Path = field(default_factory=lambda: Path("uploads"))
    notes_dir: Path = field(default_factory=lambda: Path("notes"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Logging
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("data/app.log"))

    # App
    app_name: str = "AI Personal Study & Interview Assistant"
    max_upload_size_mb: int = 20

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4

    def validate(self):

        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is missing in .env"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            )

        if not (0 <= self.gemini_temperature <= 2):
            raise ConfigError(
                "GEMINI_TEMPERATURE must be between 0 and 2"
            )

        if self.retrieval_top_k <= 0:
            raise ConfigError(
                "RETRIEVAL_TOP_K must be greater than zero"
            )

    def ensure_directories(self):

        dirs = [
            self.vector_db_dir,
            self.upload_dir,
            self.notes_dir,
            self.data_dir,
            self.sqlite_db_path.parent,
            self.log_file.parent,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:

    settings = Settings(

        gemini_api_key=_get_env(
            "GEMINI_API_KEY",
            required=True,
        ),

        gemini_model=_get_env(
            "GEMINI_MODEL",
            default="gemini-3.6-flash",
        ),

        gemini_temperature=_get_env_float(
            "GEMINI_TEMPERATURE",
            0.4,
        ),

        gemini_max_tokens=_get_env_int(
            "GEMINI_MAX_TOKENS",
            2048,
        ),

        embedding_model_name=_get_env(
            "EMBEDDING_MODEL_NAME",
            "all-MiniLM-L6-v2",
        ),

        vector_db_dir=Path(
            _get_env("VECTOR_DB_DIR", "vector_db")
        ),

        faiss_index_name=_get_env(
            "FAISS_INDEX_NAME",
            "study_assistant_index",
        ),

        sqlite_db_path=Path(
            _get_env(
                "SQLITE_DB_PATH",
                "database/study_assistant.db",
            )
        ),

        upload_dir=Path(
            _get_env("UPLOAD_DIR", "uploads")
        ),

        notes_dir=Path(
            _get_env("NOTES_DIR", "notes")
        ),

        data_dir=Path(
            _get_env("DATA_DIR", "data")
        ),

        log_level=_get_env(
            "LOG_LEVEL",
            "INFO",
        ),

        log_file=Path(
            _get_env(
                "LOG_FILE",
                "data/app.log",
            )
        ),

        app_name=_get_env(
            "APP_NAME",
            "AI Personal Study & Interview Assistant",
        ),

        max_upload_size_mb=_get_env_int(
            "MAX_UPLOAD_SIZE_MB",
            20,
        ),

        chunk_size=_get_env_int(
            "CHUNK_SIZE",
            1000,
        ),

        chunk_overlap=_get_env_int(
            "CHUNK_OVERLAP",
            150,
        ),

        retrieval_top_k=_get_env_int(
            "RETRIEVAL_TOP_K",
            4,
        ),
    )

    settings.validate()
    settings.ensure_directories()

    return settings


settings = load_settings()