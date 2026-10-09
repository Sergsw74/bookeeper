"""
Configuration management for bookeeper using pydantic-settings.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


VALID_CAPABILITIES = {"llm", "embedding", "verification"}
DEFAULT_CAPABILITIES = ["llm", "embedding", "verification"]


class OllamaServerConfig(BaseModel):
    """Configuration for an individual Ollama server node."""

    url: str = Field(description="Base URL for the Ollama server (e.g. http://192.168.50.15:11434).")
    priority: int = Field(default=1, description="Priority integer (1 is highest priority, 2, 3...).")
    name: Optional[str] = Field(default=None, description="Optional label for the server.")
    capability: List[str] = Field(
        default_factory=lambda: list(DEFAULT_CAPABILITIES),
        description="List of capabilities supported by this node: 'llm', 'embedding', 'verification'.",
    )
    mac: Optional[str] = Field(
        default=None,
        description="MAC address for Wake-on-LAN (e.g. 'D8:43:AE:FA:44:95').",
    )
    wol_broadcast: str = Field(
        default="255.255.255.255",
        description="Broadcast IP address for WOL magic packet (default: 255.255.255.255).",
    )
    wol_port: int = Field(
        default=9,
        description="UDP port for WOL magic packet (default: 9).",
    )
    wol_secret: Optional[str] = Field(
        default=None,
        description="Optional Wake-on-LAN password/secret (e.g. 6-byte hex '01:02:03:04:05:06' or 4-byte/ASCII string).",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_capabilities_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Accept 'capabilities' as plural alias for 'capability'
            if "capabilities" in data and "capability" not in data:
                data["capability"] = data.pop("capabilities")
            # If capability is passed as string (e.g. "embedding" or "llm, embedding")
            if "capability" in data and isinstance(data["capability"], str):
                data["capability"] = [c.strip() for c in data["capability"].split(",") if c.strip()]
            # Accept 'wol_mac' as alias for 'mac'
            if "wol_mac" in data and "mac" not in data:
                data["mac"] = data.pop("wol_mac")
            # Accept 'secret' or 'wol_password' as alias for 'wol_secret'
            if "secret" in data and "wol_secret" not in data:
                data["wol_secret"] = data.pop("secret")
            if "wol_password" in data and "wol_secret" not in data:
                data["wol_secret"] = data.pop("wol_password")
        return data

    @field_validator("mac")
    @classmethod
    def clean_mac(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v_clean = v.strip()
        hex_only = re.sub(r"[^0-9a-fA-F]", "", v_clean)
        if len(hex_only) != 12:
            raise ValueError(f"Invalid MAC address '{v}': must contain 12 hexadecimal characters.")
        return ":".join(hex_only[i : i + 2].upper() for i in range(0, 12, 2))

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        return v.strip().rstrip("/")

    @field_validator("capability", mode="after")
    @classmethod
    def normalize_capabilities(cls, v: List[str]) -> List[str]:
        if not v:
            return list(DEFAULT_CAPABILITIES)
        normalized = []
        for c in v:
            c_str = str(c).strip().lower()
            if c_str in ("llm", "llms", "chat", "generate"):
                normalized.append("llm")
            elif c_str in ("embedding", "embeddings", "embed", "embeds"):
                normalized.append("embedding")
            elif c_str in ("verification", "verify", "verifier"):
                normalized.append("verification")
            elif c_str:
                normalized.append(c_str)
        seen = set()
        res = []
        for c in normalized:
            if c not in seen:
                seen.add(c)
                res.append(c)
        return res or list(DEFAULT_CAPABILITIES)

    def has_capability(self, cap: str) -> bool:
        """Check if server supports the given capability (llm, embedding, verification)."""
        cap_clean = cap.strip().lower()
        if cap_clean in ("embedding", "embeddings", "embed", "embeds"):
            target = "embedding"
        elif cap_clean in ("verification", "verify", "verifier"):
            target = "verification"
        elif cap_clean in ("llm", "llms", "chat", "generate"):
            target = "llm"
        else:
            target = cap_clean
        return target in self.capability

    @property
    def supports_llm(self) -> bool:
        return self.has_capability("llm")

    @property
    def supports_embedding(self) -> bool:
        return self.has_capability("embedding")

    @property
    def supports_verification(self) -> bool:
        return self.has_capability("verification")

    @property
    def capability_str(self) -> str:
        return ", ".join(self.capability)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j knowledge graph export and upsert."""

    enabled: bool = Field(
        default=False,
        description="Whether to automatically export/upsert graph to Neo4j during build-graph.",
    )
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI (e.g. bolt://localhost:7687, neo4j://..., neo4j+s://...).",
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j authentication username.",
    )
    password: str = Field(
        default="password",
        description="Neo4j authentication password.",
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j target database name (default: 'neo4j').",
    )
    batch_size: int = Field(
        default=500,
        description="Batch size for UNWIND Cypher upsert transactions.",
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds.",
    )
    clean_export: bool = Field(
        default=False,
        description="Whether to perform a clean start (delete all existing content in database before export).",
    )


class VerificationConfig(BaseModel):
    """Configuration for Knowledge Graph factual verification."""

    enabled: bool = Field(
        default=True,
        description="Whether verification is enabled.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Dedicated LLM model for verification. Defaults to llm_model if None.",
    )
    percent: float = Field(
        default=1.0,
        description="Default sampling percentage of ideas to verify (e.g. 1.0 = 1%).",
    )
    mode: str = Field(
        default="ideas",
        description="Verification mode (default: 'ideas').",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for verification LLM calls (lower is more deterministic).",
    )
    max_examples: int = Field(
        default=20,
        description="Maximum discrepancy examples to display in report.",
    )
    distance_threshold: float = Field(
        default=0.20,
        description="Cosine distance threshold for semantic chunk expansion and coherence verification.",
    )


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
    wol_enabled: bool = Field(
        default=True,
        description="Enable Wake-on-LAN for unreachable Ollama servers during warmup.",
    )
    wol_wait_seconds: int = Field(
        default=120,
        description="Max seconds to wait for server to boot after sending WOL magic packet (default: 120s / 2 mins).",
    )
    wol_probe_interval: float = Field(
        default=5.0,
        description="Seconds between connectivity probes while waiting for woken server (default: 5.0s).",
    )
    wol_broadcast: str = Field(
        default="255.255.255.255",
        description="Default broadcast IP address for Wake-on-LAN (default: 255.255.255.255).",
    )
    wol_port: int = Field(
        default=9,
        description="Default UDP port for Wake-on-LAN (default: 9).",
    )
    max_active_tasks: Optional[int] = Field(
        default=None,
        description="Max concurrent active tasks across Ollama servers. If None, auto-calculated from alive servers.",
    )
    tasks_per_server: int = Field(
        default=3,
        description="Active task pipeline depth per alive Ollama server for fast invocation (server execution is strictly sequential, no mutual execution).",
    )
    max_active_tasks_cap: int = Field(
        default=10,
        description="Upper ceiling for auto-calculated active tasks across pool (default: 10).",
    )

    llm_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for metadata cleaning and concept extraction.",
    )
    llm_model_fallback: Optional[str] = Field(
        default=None,
        description="Optional fallback LLM model name used after 50% chunk retry attempts (e.g. 'llama3.1:8b').",
    )
    fallback_model: Optional[str] = Field(
        default=None,
        description="Alias for llm_model_fallback.",
    )
    llm_fallback_model: Optional[str] = Field(
        default=None,
        description="Alias for llm_model_fallback.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for semantic chunking and entity deduplication.",
    )
    embedding_base_url: Optional[str] = Field(
        default=None,
        description="Optional dedicated Ollama endpoint for embeddings (e.g. http://localhost:11434 for local Mac embeddings). Set to 'pool' or None to share the LLM server pool.",
    )
    embedding_servers: List[OllamaServerConfig] = Field(
        default_factory=list,
        description="Optional dedicated prioritized list of Ollama servers for embeddings.",
    )
    state_file: Optional[str] = Field(
        default=None,
        description="Optional custom path for progress checkpoint state file (default: output_dir/.bookeeper_state.json).",
    )
    request_timeout: int = Field(
        default=60,
        description="HTTP request timeout in seconds for Ollama LLM requests (default: 60s).",
    )
    max_retries: int = Field(
        default=1,
        description="Number of retries for an attempt on an Ollama server before failover/skipping (default: 1).",
    )
    retries_per_attempt: Optional[int] = Field(
        default=None,
        description="Alias for max_retries: retry count for an attempt on an Ollama server.",
    )
    attempt_retries: Optional[int] = Field(
        default=None,
        description="Alias for max_retries: retry count for an attempt on an Ollama server.",
    )
    max_chunk_attempts: int = Field(
        default=6,
        description="Maximum total attempts across pool for a stalled/poison chunk before marking as failed (default: 6).",
    )

    @model_validator(mode="after")
    def validate_retries_alias(self) -> "Settings":
        if self.retries_per_attempt is not None:
            self.max_retries = self.retries_per_attempt
        elif self.attempt_retries is not None:
            self.max_retries = self.attempt_retries
        if not self.llm_model_fallback:
            if self.fallback_model:
                self.llm_model_fallback = self.fallback_model
            elif self.llm_fallback_model:
                self.llm_model_fallback = self.llm_fallback_model
        return self

    @field_validator("ollama_servers", "embedding_servers", mode="before")
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
        """Return list of all configured servers sorted by priority ascending (1 = highest priority)."""
        if self.ollama_servers:
            return sorted(self.ollama_servers, key=lambda s: s.priority)
        return [OllamaServerConfig(url=self.ollama_base_url, priority=1, name="default")]

    @property
    def resolved_llm_servers(self) -> List[OllamaServerConfig]:
        """Return list of servers supporting LLM tasks, or fallback to all servers."""
        all_servers = self.resolved_ollama_servers
        llm_nodes = [s for s in all_servers if s.supports_llm]
        return llm_nodes or all_servers

    @property
    def resolved_embedding_servers(self) -> List[OllamaServerConfig]:
        """Return dedicated list of servers for embeddings, or pool servers supporting embedding tasks."""
        if self.embedding_servers:
            valid_servers = [s for s in self.embedding_servers if s.url.strip().lower() != "pool"]
            if valid_servers:
                return sorted(valid_servers, key=lambda s: s.priority)
        if self.embedding_base_url and self.embedding_base_url.strip().lower() != "pool":
            return [OllamaServerConfig(url=self.embedding_base_url, priority=1, name="local-embeddings", capability=["embedding"])]
        # Filter Ollama pool nodes that support embedding tasks
        all_servers = self.resolved_ollama_servers
        embed_nodes = [s for s in all_servers if s.supports_embedding]
        return embed_nodes or all_servers

    @property
    def resolved_verification_servers(self) -> List[OllamaServerConfig]:
        """Return list of servers supporting verification tasks, or fallback to LLM servers."""
        all_servers = self.resolved_ollama_servers
        verify_nodes = [s for s in all_servers if s.supports_verification]
        return verify_nodes or self.resolved_llm_servers

    @property
    def uses_llm_pool_for_embeddings(self) -> bool:
        """Return True if embeddings share the LLM server pool rather than using a dedicated endpoint."""
        if self.embedding_servers and any(s.url.strip().lower() != "pool" for s in self.embedding_servers):
            return False
        if self.embedding_base_url and self.embedding_base_url.strip().lower() != "pool":
            return False
        return True

    def calculate_pool_concurrency(self, num_servers: int) -> int:
        """
        Calculate active concurrent tasks limit in the task pipeline:
        - default: 3 * num_servers (for fast invocation prefetch, capped at max_active_tasks_cap)
        - custom override: max_active_tasks if explicitly configured
        Server execution is strictly sequential per Ollama instance (1 task at a time, no mutual execution).
        """
        if self.max_active_tasks is not None:
            return max(1, self.max_active_tasks)
        n = max(1, num_servers)
        return min(self.max_active_tasks_cap, max(n, self.tasks_per_server * n))

    # Processing & Deduplication
    similarity_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for merging duplicate concept nodes (recommended: 0.92-0.95).",
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

    chunks_dir: Optional[str] = Field(
        default=None,
        description="Optional custom directory for persistent book chunks storage (default: output_dir/chunks).",
    )

    @property
    def resolved_chunks_dir(self) -> Path:
        """Return expanded Path for persistent book chunks storage."""
        if self.chunks_dir:
            p = Path(self.chunks_dir).expanduser().resolve()
        else:
            p = self.resolved_output_dir / "chunks"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # LightRAG Engine settings
    enable_lightrag: bool = Field(
        default=False,
        description="Enable LightRAG engine for fast incremental indexing and dual-level retrieval.",
    )
    lightrag_dir: Optional[str] = Field(
        default=None,
        description="Working directory for LightRAG graph and vector indices (default: output_dir/lightrag).",
    )
    lightrag_embedding_dim: int = Field(
        default=768,
        description="Embedding dimension for LightRAG (768 for nomic-embed-text).",
    )
    lightrag_mode: str = Field(
        default="hybrid",
        description="Default retrieval mode for LightRAG queries ('hybrid', 'local', 'global', 'naive', or 'mix').",
    )

    # Neo4j Graph Database Export
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j graph database export configuration.",
    )

    @field_validator("neo4j", mode="before")
    @classmethod
    def parse_neo4j_config(cls, v: Any) -> Any:
        if v is None:
            return Neo4jConfig()
        if isinstance(v, dict):
            return Neo4jConfig(**v)
        return v

    # Knowledge Graph Factual Verification
    verification: VerificationConfig = Field(
        default_factory=VerificationConfig,
        description="Knowledge Graph factual verification configuration.",
    )

    @field_validator("verification", mode="before")
    @classmethod
    def parse_verification_config(cls, v: Any) -> Any:
        if v is None:
            return VerificationConfig()
        if isinstance(v, dict):
            return VerificationConfig(**v)
        return v

    @property
    def resolved_verifier_model(self) -> str:
        """Return dedicated verification model or fallback to llm_model."""
        return self.verification.model or self.llm_model

    clean_export: bool = Field(
        default=False,
        description="Whether to perform a clean start on export targets (clearing existing notes or database content before export).",
    )

    @property
    def resolved_lightrag_dir(self) -> Path:
        """Return expanded Path for LightRAG working directory."""
        if self.lightrag_dir:
            p = Path(self.lightrag_dir).expanduser().resolve()
        else:
            p = self.resolved_output_dir / "lightrag"
        p.mkdir(parents=True, exist_ok=True)
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
            if key == "neo4j" and isinstance(val, dict):
                flat_data["neo4j"] = val
            elif key == "verification" and isinstance(val, dict):
                flat_data["verification"] = val
            elif key == "lightrag" and isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if sub_key == "enabled":
                        flat_data["enable_lightrag"] = sub_val
                    else:
                        flat_data[f"lightrag_{sub_key}"] = sub_val
            elif isinstance(val, dict):
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
