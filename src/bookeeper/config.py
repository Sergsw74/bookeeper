"""
Configuration management for bookeeper using pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for bookeeper."""

    model_config = SettingsConfigDict(
        env_prefix="BOOKEEPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Calibre integration
    calibre_library_path: str = Field(
        default="~/Calibre Library",
        description="Path to local library directory (with metadata.db) or remote URL (http://host:port/#library).",
    )
    calibre_user: Optional[str] = Field(
        default=None,
        description="Optional username for remote Calibre content server.",
    )
    calibre_password: Optional[str] = Field(
        default=None,
        description="Optional password for remote Calibre content server.",
    )

    # Local Ollama LLM & Embeddings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server endpoint URL.",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for metadata cleaning and concept extraction.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for semantic chunking and entity deduplication.",
    )

    # Processing & Deduplication
    similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold for merging duplicate concept nodes.",
    )

    # Output directory for graphs & exports
    output_dir: str = Field(
        default="./output",
        description="Base output directory for generated graphs and Obsidian vaults.",
    )

    @property
    def resolved_output_dir(self) -> Path:
        """Return expanded Path for output_dir."""
        return Path(self.output_dir).expanduser().resolve()

    @property
    def is_remote_calibre(self) -> bool:
        """Check if calibre_library_path is an HTTP/HTTPS URL."""
        path_str = self.calibre_library_path.strip().lower()
        return path_str.startswith("http://") or path_str.startswith("https://")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Load settings from a YAML file with path expansion."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file not found at {resolved}")

        with open(resolved, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle nested or flat YAML
        flat_data = {}
        for key, val in data.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flat_data[f"{key}_{sub_key}"] = sub_val
                    flat_data[sub_key] = sub_val
            else:
                flat_data[key] = val

        return cls(**flat_data)


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Retrieve application settings. Checks:
    1. Explicit config_path if provided
    2. ./config.yaml
    3. ~/.config/bookeeper/config.yaml
    4. Default settings with env vars
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    candidates.extend(
        [
            Path("config.yaml"),
            Path("~/.config/bookeeper/config.yaml").expanduser(),
        ]
    )

    for p in candidates:
        if p.is_file():
            try:
                return Settings.from_yaml(p)
            except Exception:
                pass

    return Settings()
