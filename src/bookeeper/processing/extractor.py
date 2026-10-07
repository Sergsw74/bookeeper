"""
LangChain structured output schemas and Ollama extraction chains for book metadata and concepts.
"""

import json
import logging
import shutil
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bookeeper.config import Settings

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
    """Canonical domain idea, architectural pattern, or technical theme."""

    name: str = Field(
        description="Canonical concise 2-4 word concept name (e.g. 'Event Sourcing', 'Circuit Breaker', 'Two-Phase Commit')."
    )
    category: str = Field(
        description="Category classification (e.g. 'Architectural Pattern', 'Data Structure', 'System Design Principle', 'Tradeoff', 'Theme')."
    )
    summary: str = Field(
        description="Clear, 1-2 sentence explanation defining what this concept is and how it functions."
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="List of related concept names (2-4 words each) discussed in connection with this concept.",
    )


class SectionExtraction(BaseModel):
    """Extraction output containing all concepts discovered in a section."""

    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical concepts and architectural patterns discussed.",
    )


# ==============================================================================
# KnowledgeExtractor Chain
# ==============================================================================


class KnowledgeExtractor:
    """Extracts structured metadata and concept graphs using ChatOllama with structured output."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.temperature = temperature

        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature,
        )

        # Create typed structured chains using with_structured_output
        self.metadata_chain = self.llm.with_structured_output(BookMetadata)
        self.section_chain = self.llm.with_structured_output(SectionExtraction)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "KnowledgeExtractor":
        return cls(
            base_url=base_url or settings.ollama_base_url,
            model=model or settings.llm_model,
        )

    def clean_metadata(
        self,
        raw_title: str,
        raw_authors: List[str],
        raw_comments: Optional[str] = None,
        content_sample: Optional[str] = None,
        file_hint: Optional[str] = None,
    ) -> BookMetadata:
        """Clean and normalize title, authors, and summary blurb using content sampling."""
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
            "(e.g., mojibake/question marks like ' 2033'), missing data, generic placeholders ('Unknown'), or is "
            "just a filename ('zubkov'), identify the actual canonical title and author from the content excerpt, "
            "RTF/OPF headers, and file hints.\n"
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

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("human", user_content),
            ]
        )

        try:
            result = self.metadata_chain.invoke(prompt.format_messages())
            if isinstance(result, BookMetadata):
                return result
            return BookMetadata(**dict(result))
        except Exception as e:
            logger.warning(f"Metadata cleaning LLM call failed: {e}. Falling back to raw.")
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
    ) -> SectionExtraction:
        """Extract atomic concepts and their relationships from a section chunk."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical knowledge graph extractor. "
                    "Extract canonical, specific technical concepts, patterns, algorithms, and design tradeoffs. "
                    "Ensure concept names are concise (2-4 words, e.g. 'Read-Copy-Update', 'Consistent Hashing'). "
                    "Identify how these concepts relate to one another.",
                ),
                (
                    "human",
                    "Book: '{book_title}'\n"
                    "Section: '{section_title}'\n\n"
                    "Text excerpt:\n\"\"\"\n{text}\n\"\"\"\n",
                ),
            ]
        )

        formatted_messages = prompt.format_messages(
            book_title=book_title,
            section_title=section_title,
            text=text[:4000],  # Ensure token safety
        )

        try:
            result = self.section_chain.invoke(formatted_messages)
            if isinstance(result, SectionExtraction):
                return result
            return SectionExtraction(**dict(result))
        except Exception as e:
            logger.warning(f"Section extraction LLM call failed for '{section_title}': {e}")
            return SectionExtraction(concepts=[])

    def warmup_and_check_device(self, timeout: int = 30) -> Dict[str, Any]:
        """
        Trigger a lightweight warmup request to ensure the model is loaded in Ollama,
        and query /api/ps to verify whether Ollama is executing on GPU (VRAM) or CPU.
        Returns a dict containing execution device, VRAM metrics, and warning if CPU-bound.
        """
        def _fetch_json(endpoint: str, post_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            url = f"{self.base_url}{endpoint}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(post_body).encode("utf-8") if post_body else None,
                    headers={"Content-Type": "application/json"} if post_body else {},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                # Graceful fallback to curl (handles macOS non-interactive subshell socket sandboxing)
                if shutil.which("curl"):
                    cmd = ["curl", "-s", "--max-time", str(timeout)]
                    if post_body:
                        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(post_body)])
                    cmd.append(url)
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        return json.loads(res.stdout)
                raise

        # 1. Warmup ping to ensure the model is loaded into memory / VRAM
        try:
            _fetch_json(
                "/api/generate",
                {"model": self.model_name, "prompt": "warmup ping", "options": {"num_predict": 1}, "stream": False},
            )
        except Exception as e:
            logger.warning(f"Ollama warmup ping failed for {self.model_name}: {e}")

        # 2. Query /api/ps to inspect running model allocation
        try:
            ps_data = _fetch_json("/api/ps")
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

                if size_vram == 0:
                    return {
                        "status": "ok",
                        "is_gpu": False,
                        "device": "CPU",
                        "size": size,
                        "size_vram": 0,
                        "vram_pct": 0.0,
                        "runner": runner,
                        "model": model_active,
                        "warning": (
                            f"Ollama model '{model_active}' is running on CPU (0% VRAM offloaded). "
                            "Extraction speed may be significantly slower without GPU acceleration."
                        ),
                    }
                elif size_vram >= size:
                    return {
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
                        "status": "ok",
                        "is_gpu": True,
                        "device": f"Partial GPU ({pct}% VRAM offload)",
                        "size": size,
                        "size_vram": size_vram,
                        "vram_pct": pct,
                        "runner": runner,
                        "model": model_active,
                        "warning": f"Ollama model '{model_active}' is partially offloaded ({pct}% VRAM, remainder on CPU).",
                    }

            return {
                "status": "unknown",
                "is_gpu": False,
                "device": "Unknown",
                "size": 0,
                "size_vram": 0,
                "vram_pct": 0.0,
                "runner": "unknown",
                "model": self.model_name,
                "warning": None,
            }
        except Exception as e:
            return {
                "status": "error",
                "is_gpu": False,
                "device": "Unknown",
                "size": 0,
                "size_vram": 0,
                "vram_pct": 0.0,
                "runner": "unknown",
                "model": self.model_name,
                "warning": f"Unable to reach Ollama process metrics at {self.base_url}: {e}",
                "error": str(e),
            }
