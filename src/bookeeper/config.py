"""
Configuration management for bookeeper using Pydantic v2 and YAML.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CalibreSettings(BaseModel):
    """Calibre library integration settings."""

    library_path: Path = Field(
        default=Path("~/Calibre Library"),
        description="Path to local Calibre library containing metadata.db.",
    )
    calibredb_path: Optional[str] = Field(
        default=None,
        description="Explicit path to calibredb CLI binary, or null to search PATH.",
    )
    preferred_formats: List[str] = Field(
        default=["EPUB", "PDF"],
        description="Ordered list of preferred book formats to ingest.",
    )


class OllamaSettings(BaseModel):
    """Local Ollama instance configuration."""

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama daemon.",
    )
    model: str = Field(
        default="llama3:8b",
        description="LLM model name for metadata extraction & reasoning.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for vector representation and entity resolution.",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature for structured extraction (0.0 for deterministic).",
    )
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds.",
    )


class ProcessingSettings(BaseModel):
    """Text processing and chunking settings."""

    chunk_size: int = Field(
        default=2000,
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=300,
        description="Character overlap between consecutive chunks.",
    )
    min_chunk_size: int = Field(
        default=200,
        description="Minimum character length threshold to process a chunk.",
    )
    toc_aware: bool = Field(
        default=True,
        description="Whether to segment by Table-of-Contents / chapter boundaries first.",
    )
    dedup_similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold for embedding-based entity resolution.",
    )


class ObsidianExportSettings(BaseModel):
    """Obsidian Markdown vault export settings."""

    enabled: bool = True
    vault_path: Path = Path("./obsidian_vault")
    create_moc: bool = True
    embed_metadata: bool = True


class GraphMLExportSettings(BaseModel):
    """GraphML export settings."""

    enabled: bool = True
    output_path: Path = Path("./graph_exports/knowledge_graph.graphml")


class CytoscapeExportSettings(BaseModel):
    """Cytoscape JSON export settings."""

    enabled: bool = True
    output_path: Path = Path("./graph_exports/cytoscape_elements.json")


class ExportSettings(BaseModel):
    """Container for graph exporter settings."""

    obsidian: ObsidianExportSettings = Field(default_factory=ObsidianExportSettings)
    graphml: GraphMLExportSettings = Field(default_factory=GraphMLExportSettings)
    cytoscape: CytoscapeExportSettings = Field(default_factory=CytoscapeExportSettings)


class GraphSettings(BaseModel):
    """Knowledge Graph storage and export settings."""

    backend: Literal["networkx", "sqlite", "neo4j"] = Field(
        default="networkx",
        description="Graph storage backend.",
    )
    storage_path: Path = Field(
        default=Path("./data/bookeeper_graph.json"),
        description="Storage location for serialized graph data.",
    )
    exports: ExportSettings = Field(default_factory=ExportSettings)


class Settings(BaseSettings):
    """Root configuration object for bookeeper."""

    model_config = SettingsConfigDict(
        env_prefix="BOOKEEPER_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    calibre: CalibreSettings = Field(default_factory=CalibreSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Load settings from a YAML file with path expansion."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file not found at {resolved}")

        with open(resolved, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Resolve tildes in paths before initialization
        if "calibre" in data and "library_path" in data["calibre"]:
            data["calibre"]["library_path"] = str(
                Path(data["calibre"]["library_path"]).expanduser()
            )
        return cls(**data)


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Retrieve application settings. Looks in:
    1. Explicit config_path if provided
    2. ./config.yaml
    3. ~/.config/bookeeper/config.yaml
    4. Default settings instance
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
            return Settings.from_yaml(p)

    return Settings()
