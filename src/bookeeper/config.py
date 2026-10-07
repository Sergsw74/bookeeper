"""
Configuration management for bookeeper using pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaServerConfig(BaseModel):
    """Configuration for an individual Ollama server node."""

    url: str = Field(description="Base URL for the Ollama server (e.g. http://192.168.50.15:11434).")
    priority: int = Field(default=1, description="Priority integer (1 is highest priority, 2, 3...).")
    name: Optional[str] = Field(default=None, description="Optional label for the server.")

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        return v.strip().rstrip("/")


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
    stage_metadata_db: bool = Field(
        default=True,
        description="Download metadata.db locally for microsecond SSD transactions and sync back on completion.",
    )
    backup_metadata_db: bool = Field(
        default=True,
        description="Create remote metadata.db.bak backup before syncing staged DB.",
    )
    staged_db_path: Optional[str] = Field(
        default=None,
        description="Optional custom local path for staged metadata.db (default: output_dir/.staged_metadata.db).",
    )

    # Local / Remote Ollama LLM & Embeddings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Primary or default Ollama server endpoint URL.",
    )
    ollama_servers: List[OllamaServerConfig] = Field(
        default_factory=list,
        description="Prioritized list of Ollama servers for multi-server failover client.",
    )
    failover_cooldown_seconds: int = Field(
        default=600,
        description="Cooldown duration in seconds (default 600s = 10 min) before retrying a failed server.",
    )

    llm_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for metadata cleaning and concept extraction.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for semantic chunking and entity deduplication.",
    )
    state_file: Optional[str] = Field(
        default=None,
        description="Optional custom path for progress checkpoint state file (default: output_dir/.bookeeper_state.json).",
    )

    @field_validator("ollama_servers", mode="before")
    @classmethod
    def parse_servers(cls, v: Any) -> List[Any]:
        if not v:
            return []
        if isinstance(v, str):
            return [
                {"url": u.strip(), "priority": idx + 1}
                for idx, u in enumerate(v.split(","))
                if u.strip()
            ]
        if isinstance(v, list):
            res = []
            for idx, item in enumerate(v):
                if isinstance(item, str):
                    res.append({"url": item.strip(), "priority": idx + 1})
                elif isinstance(item, dict):
                    if "priority" not in item:
                        item["priority"] = idx + 1
                    res.append(item)
                else:
                    res.append(item)
            return res
        return v

    @property
    def resolved_ollama_servers(self) -> List[OllamaServerConfig]:
        """Return list of servers sorted by priority ascending (1 = highest priority)."""
        if self.ollama_servers:
            return sorted(self.ollama_servers, key=lambda s: s.priority)
        return [OllamaServerConfig(url=self.ollama_base_url, priority=1, name="default")]

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
    def resolved_state_file(self) -> Path:
        """Return expanded Path for state checkpoint file."""
        if self.state_file:
            p = Path(self.state_file).expanduser().resolve()
        else:
            p = self.resolved_output_dir / ".bookeeper_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def resolved_staged_db_path(self) -> Path:
        """Return expanded Path for staged metadata.db."""
        if self.staged_db_path:
            p = Path(self.staged_db_path).expanduser().resolve()
        else:
            p = self.resolved_output_dir / ".staged_metadata.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

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
