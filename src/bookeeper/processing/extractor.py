"""
LangChain structured output schemas and Ollama extraction chains for book metadata and concepts.
"""

import json
import logging
import shutil
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator

from bookeeper.config import Settings
from bookeeper.processing.ollama_pool import OllamaPool

logger = logging.getLogger(__name__)


# ==============================================================================
# Strict Pydantic Output Schemas
# ==============================================================================


class BookMetadata(BaseModel):
    """Cleaned and normalized book catalog metadata."""

    title: str = Field(
        description="Clean canonical book title in original/canonical language (e.g. 'Метро 2033', 'Designing Data-Intensive Applications')."
    )
    author: str = Field(
        description="Canonical author name or comma-separated authors (e.g. 'Дмитрий Глуховский', 'Сергей Зубков', 'Martin Kleppmann')."
    )
    summary: str = Field(
        description="Concise catalog blurb (2-4 sentences) summarizing the book's core premise, subject, or technical themes."
    )


class Concept(BaseModel):
    """Canonical domain idea, architectural pattern, principle, or technical theme."""

    name: str = Field(
        description="Canonical concise title of the idea or concept (2-4 words, e.g. 'Event Sourcing', 'Leader Election', 'Write-Ahead Logging')."
    )
    brief_description: str = Field(
        default="",
        description="Brief 1-2 sentence description summarizing the core idea.",
    )
    detailed_explanation: str = Field(
        default="",
        description="Detailed multi-sentence explanation of the idea, how it works, mechanisms, and key nuances.",
    )
    category: str = Field(
        default="Idea",
        description="Category classification (e.g. 'Architectural Pattern', 'System Design Principle', 'Theory', 'Tradeoff', 'Data Structure').",
    )
    supporting_quote: Optional[str] = Field(
        default=None,
        description="Direct concise quote or excerpt from the text that directly introduces or supports this idea.",
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="List of related idea/concept names discussed in connection with this idea.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_descriptions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If 'summary' was provided instead of brief/detailed descriptions
            if "summary" in data and not data.get("brief_description"):
                data["brief_description"] = data["summary"]
            if not data.get("detailed_explanation") and data.get("brief_description"):
                data["detailed_explanation"] = data["brief_description"]
            if not data.get("brief_description") and data.get("detailed_explanation"):
                data["brief_description"] = data["detailed_explanation"]
        return data

    @property
    def summary(self) -> str:
        """Backward-compatible summary property."""
        return self.brief_description or self.detailed_explanation


class SectionExtraction(BaseModel):
    """Extraction output containing all concepts discovered in a section."""

    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical ideas, concepts, and architectural patterns discussed.",
    )


# ==============================================================================
# KnowledgeExtractor Chain with Multi-Server Failover
# ==============================================================================


class KnowledgeExtractor:
    """Extracts structured metadata and concept graphs using Ollama with multi-server failover."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        pool: Optional[OllamaPool] = None,
        cooldown_seconds: int = 600,
    ):
        self.model_name = model
        self.temperature = temperature
        self.cooldown_seconds = cooldown_seconds

        if pool is not None:
            self.pool = pool
        elif base_url:
            urls = [u.strip() for u in base_url.split(",") if u.strip()]
            self.pool = OllamaPool.from_urls(urls, cooldown_seconds=cooldown_seconds)
        else:
            self.pool = OllamaPool.from_urls(["http://localhost:11434"], cooldown_seconds=cooldown_seconds)

        self.base_url = self.pool.primary_server.url

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "KnowledgeExtractor":
        if base_url:
            urls = [u.strip() for u in base_url.split(",") if u.strip()]
            pool = OllamaPool.from_urls(urls, cooldown_seconds=settings.failover_cooldown_seconds)
        else:
            pool = OllamaPool(
                servers=settings.resolved_ollama_servers,
                cooldown_seconds=settings.failover_cooldown_seconds,
            )
        return cls(
            pool=pool,
            model=model or settings.llm_model,
            cooldown_seconds=settings.failover_cooldown_seconds,
        )

    def _execute_structured_invoke(self, schema_cls: Any, messages: Any) -> Any:
        """Execute structured output invocation with multi-server failover."""
        def _invoke(url: str):
            llm = ChatOllama(
                base_url=url,
                model=self.model_name,
                temperature=self.temperature,
            )
            chain = llm.with_structured_output(schema_cls)
            return chain.invoke(messages)

        return self.pool.execute_with_failover(_invoke)

    def clean_metadata(
        self,
        raw_title: str,
        raw_authors: List[str],
        raw_comments: Optional[str] = None,
        content_sample: Optional[str] = None,
        file_hint: Optional[str] = None,
    ) -> BookMetadata:
        """Clean and normalize title, authors, and summary blurb using content sampling."""
        from bookeeper.calibre.parser import BookParser

        raw_title = BookParser.repair_mojibake(raw_title)
        raw_authors = [BookParser.repair_mojibake(a) for a in raw_authors]
        if raw_comments:
            raw_comments = BookParser.repair_mojibake(raw_comments)
        if content_sample:
            content_sample = BookParser.repair_mojibake(content_sample)

        authors_str = ", ".join(raw_authors) if raw_authors else "Unknown"
        comments_str = (raw_comments or "").strip()

        system_instruction = (
            "You are an expert multilingual library cataloguer and metadata specialist.\n"
            "Your task is to identify the canonical book title, author(s), and generate an accurate "
            "executive summary blurb (2-4 sentences) describing the book's premise, themes, and content.\n"
            "Important guidelines:\n"
            "1. Multilingual Support: Preserve the book's canonical language and script (e.g. Russian Cyrillic, "
            "English, French, etc.). If the book is in Russian, produce the title, author, and summary in Russian.\n"
            "2. Identify Corrupted or Missing Metadata: If the raw Calibre title/author has encoding errors "
            "(e.g., mojibake like 'Ñóïåðìåí Ïðèêëþ÷åíèÿ' which is Windows-1251 decoded as Latin-1 for 'Супермен Приключения', "
            "or ' 2033'), missing data, generic placeholders ('Unknown'), or is just a filename ('zubkov'), identify "
            "the actual canonical title and author from the content excerpt, RTF/OPF headers, and file hints.\n"
            "   - ALWAYS output clean, natural UTF-8 text in the book's proper script. NEVER echo garbled mojibake characters like 'Ñóïåðìåí' or 'Ð.Ð.Ð.'.\n"
            "   - E.g. ' 2033' / '2033 - .rtf' is 'Метро 2033' by 'Дмитрий Глуховский'.\n"
            "   - E.g. 'zubkov' / 'zubkov.djvu' in Russian computer science refers to 'Сергей Зубков' (С. В. Зубков), "
            "author of 'Ассемблер. Для DOS, Windows и UNIX'.\n"
            "3. Comprehensive Summary: ALWAYS generate a rich, informative 2-4 sentence summary blurb describing "
            "what the book is about based on the excerpt, premise, or subject. NEVER output 'No summary available', "
            "'None', or placeholder text."
        )

        user_content = (
            f"Please clean and standardize this book's metadata:\n\n"
            f"Raw Calibre Title: {raw_title}\n"
            f"Raw Calibre Authors: {authors_str}\n"
            f"Existing Comments / Blurb: {comments_str or '(None)'}\n"
        )
        if file_hint:
            user_content += f"File / Format Hint: {file_hint}\n"
        if content_sample:
            user_content += f"\nBook Content Excerpt (Title page / Preface / Beginning):\n\"\"\"\n{content_sample[:3500]}\n\"\"\"\n"

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content),
        ]

        try:
            result = self._execute_structured_invoke(BookMetadata, messages)
            if isinstance(result, BookMetadata):
                result.title = BookParser.repair_mojibake(result.title)
                result.author = BookParser.repair_mojibake(result.author)
                result.summary = BookParser.repair_mojibake(result.summary)
                return result
            data = dict(result)
            return BookMetadata(
                title=BookParser.repair_mojibake(data.get("title", raw_title)),
                author=BookParser.repair_mojibake(data.get("author", authors_str)),
                summary=BookParser.repair_mojibake(data.get("summary", "No summary available.")),
            )
        except Exception as e:
            logger.warning(f"Metadata cleaning LLM call failed across all pool servers: {e}. Falling back to raw.")
            return BookMetadata(
                title=raw_title,
                author=authors_str,
                summary=comments_str[:300] if comments_str else "No summary available.",
            )

    def extract_section(
        self,
        text: str,
        book_title: str,
        section_title: str,
        subtitle: Optional[str] = None,
        parent_context: Optional[str] = None,
    ) -> SectionExtraction:
        """
        Extract canonical domain ideas, principles, patterns, and their relationships
        from a section or chunk with multi-server failover.
        Supports parent macro context for hierarchical RAG understanding.
        """
        system_text = (
            "You are an expert technical knowledge extractor and concept ontologist. "
            "Extract canonical ideas, principles, patterns, and theoretical mechanisms discussed in the text.\n"
            "For each idea:\n"
            "- 'name': concise canonical title (2-4 words, e.g. 'Consistent Hashing', 'Two-Phase Commit', 'Event Sourcing').\n"
            "- 'brief_description': 1-2 sentence high-level definition of the idea.\n"
            "- 'detailed_explanation': thorough multi-sentence technical explanation of how the idea functions, its mechanisms, and tradeoffs.\n"
            "- 'category': classification (e.g. 'Architectural Pattern', 'System Design Principle', 'Data Structure', 'Algorithm', 'Design Tradeoff').\n"
            "- 'supporting_quote': direct concise sentence from the text directly stating or supporting this idea.\n"
            "- 'related_concepts': list of related idea names discussed in relation to this idea."
        )

        loc = f"Book: '{book_title}'\nSection/Chapter: '{section_title}'"
        if subtitle and subtitle != section_title:
            loc += f"\nSubsection/Subtitle: '{subtitle}'"

        human_text = f"{loc}\n\n"
        if parent_context and parent_context != text:
            human_text += f"Parent Context (Macro Passage):\n\"\"\"\n{parent_context[:2000]}\n\"\"\"\n\n"
        human_text += f"Target Text for Idea Extraction:\n\"\"\"\n{text[:4000]}\n\"\"\"\n"

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]

        try:
            result = self._execute_structured_invoke(SectionExtraction, messages)
            if isinstance(result, SectionExtraction):
                return result
            return SectionExtraction(**dict(result))
        except Exception as e:
            logger.warning(f"Section extraction LLM call failed across all pool servers for '{section_title}': {e}")
            return SectionExtraction(concepts=[])

    def warmup_and_check_device(self, timeout: int = 15) -> Dict[str, Any]:
        """
        Trigger warmup requests and query /api/ps across all configured Ollama servers in the pool.
        Returns metrics for each server, identifying GPU (VRAM) vs CPU execution and failover priority.
        """
        def _fetch_server_json(base_url: str, endpoint: str, post_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            url = f"{base_url.rstrip('/')}{endpoint}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(post_body).encode("utf-8") if post_body else None,
                    headers={"Content-Type": "application/json"} if post_body else {},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                if shutil.which("curl"):
                    cmd = ["curl", "-s", "--max-time", str(timeout)]
                    if post_body:
                        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(post_body)])
                    cmd.append(url)
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        return json.loads(res.stdout)
                raise

        def _inspect_node(node) -> Dict[str, Any]:
            # 1. Warmup ping
            try:
                _fetch_server_json(
                    node.url,
                    "/api/generate",
                    {"model": self.model_name, "prompt": "warmup ping", "options": {"num_predict": 1}, "stream": False},
                )
            except Exception as e:
                logger.debug(f"Warmup ping error for {node.url}: {e}")

            # 2. Query /api/ps
            try:
                ps_data = _fetch_server_json(node.url, "/api/ps")
                models = ps_data.get("models", [])

                matched = None
                target_norm = self.model_name.lower()
                for m in models:
                    m_name = (m.get("name") or m.get("model") or "").lower()
                    if m_name == target_norm or m_name.startswith(target_norm) or target_norm.startswith(m_name.split(":")[0]):
                        matched = m
                        break

                if not matched and models:
                    matched = models[0]

                if matched:
                    size = matched.get("size", 0)
                    size_vram = matched.get("size_vram", 0)
                    runner = matched.get("runner", "llamacpp")
                    model_active = matched.get("name") or self.model_name
                    node.mark_success()

                    if size_vram == 0:
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": False,
                            "device": "CPU",
                            "size": size,
                            "size_vram": 0,
                            "vram_pct": 0.0,
                            "runner": runner,
                            "model": model_active,
                            "warning": f"Running on CPU (0% VRAM offload).",
                        }
                    elif size_vram >= size:
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": True,
                            "device": "GPU (100% VRAM offload)",
                            "size": size,
                            "size_vram": size_vram,
                            "vram_pct": 100.0,
                            "runner": runner,
                            "model": model_active,
                            "warning": None,
                        }
                    else:
                        pct = round((size_vram / size) * 100, 1)
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": True,
                            "device": f"Partial GPU ({pct}% VRAM offload)",
                            "size": size,
                            "size_vram": size_vram,
                            "vram_pct": pct,
                            "runner": runner,
                            "model": model_active,
                            "warning": f"Partially offloaded ({pct}% VRAM, remainder on CPU).",
                        }

                node.mark_success()
                return {
                    "url": node.url,
                    "priority": node.priority,
                    "name": node.label,
                    "status": "idle",
                    "is_gpu": False,
                    "device": "Idle/Ready",
                    "size": 0,
                    "size_vram": 0,
                    "vram_pct": 0.0,
                    "runner": "unknown",
                    "model": self.model_name,
                    "warning": None,
                }
            except Exception as e:
                node.mark_failure(str(e))
                return {
                    "url": node.url,
                    "priority": node.priority,
                    "name": node.label,
                    "status": "unreachable",
                    "is_gpu": False,
                    "device": "Unreachable",
                    "size": 0,
                    "size_vram": 0,
                    "vram_pct": 0.0,
                    "runner": "unknown",
                    "model": self.model_name,
                    "warning": f"Unreachable: {e}",
                    "error": str(e),
                }

        server_reports = [_inspect_node(n) for n in self.pool.nodes]
        primary_node = self.pool.primary_server
        primary_report = next((r for r in server_reports if r["url"] == primary_node.url), server_reports[0])

        return {
            "status": "ok",
            "model": self.model_name,
            "servers": server_reports,
            "primary": primary_report,
            "pool_status": self.pool.get_status(),
            "device": primary_report.get("device", "Unknown"),
            "is_gpu": primary_report.get("is_gpu", False),
            "size": primary_report.get("size", 0),
            "size_vram": primary_report.get("size_vram", 0),
            "vram_pct": primary_report.get("vram_pct", 0.0),
            "runner": primary_report.get("runner", "llamacpp"),
            "warning": primary_report.get("warning"),
        }
