"""
LangChain structured output schemas and Ollama extraction chains for book metadata and concepts.
"""

import json
import logging
import re
import shutil
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator

from bookeeper.calibre.parser import BookParser
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

    @model_validator(mode="before")
    @classmethod
    def sanitize_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for fld in ("title", "author", "summary"):
                if fld in data and isinstance(data[fld], str):
                    data[fld] = BookParser.repair_mojibake(data[fld])
        return data


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
    weight: int = Field(
        default=5,
        ge=0,
        le=10,
        description=(
            "Significance, depth, and specificity weight from 0 to 10. "
            "0-2: very basic/generic concepts (e.g. 'Identification', 'Storage', 'Process', weight=0). "
            "3-5: less generic, valuable foundational concepts (e.g. 'Risk Assessment', weight=3; 'Load Balancing', weight=4). "
            "6-8: specific, advanced architectural approaches or paradigms (e.g. 'RAG', weight=7; 'Event Sourcing', weight=7). "
            "9-10: highly specific, novel, or deep architectural/philosophical concepts (e.g. 'SmartRAG', weight=9; 'Deterministic Simulation Testing', weight=10)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def populate_descriptions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("name", "brief_description", "detailed_explanation", "category", "supporting_quote", "summary"):
                if k in data and isinstance(data[k], str):
                    data[k] = BookParser.repair_mojibake(data[k])
            if "related_concepts" in data and isinstance(data["related_concepts"], list):
                data["related_concepts"] = [
                    BookParser.repair_mojibake(c) if isinstance(c, str) else c
                    for c in data["related_concepts"]
                ]

            # If 'summary' was provided instead of brief/detailed descriptions
            if "summary" in data and not data.get("brief_description"):
                data["brief_description"] = data["summary"]
            if not data.get("detailed_explanation") and data.get("brief_description"):
                data["detailed_explanation"] = data["brief_description"]
            if not data.get("brief_description") and data.get("detailed_explanation"):
                data["brief_description"] = data["detailed_explanation"]
            if "weight" in data:
                try:
                    w = int(data["weight"])
                    data["weight"] = max(0, min(10, w))
                except (ValueError, TypeError):
                    data["weight"] = 5
        return data

    @property
    def summary(self) -> str:
        """Backward-compatible summary property."""
        return self.brief_description or self.detailed_explanation

    @summary.setter
    def summary(self, val: str) -> None:
        self.brief_description = val


class SectionExtraction(BaseModel):
    """Extraction output containing all concepts discovered in a section."""

    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical ideas, concepts, and architectural patterns discussed.",
    )


def validate_concept_chunk_grounding(
    concept_name: str,
    supporting_quote: Optional[str],
    chunk_text: str,
    brief_description: str = "",
) -> Tuple[bool, str]:
    """
    Validate whether an extracted concept has genuine textual grounding in chunk_text.
    Returns (is_grounded, validated_quote).

    - If supporting_quote is present and grounded in chunk_text -> (True, supporting_quote)
    - If supporting_quote is present but ungrounded in chunk_text:
        - If concept_name or brief_description has keyword overlap in chunk -> (True, "") [strip bogus quote]
        - If neither quote nor concept has any presence in chunk -> (False, "") [omit false link]
    - If supporting_quote is not provided:
        - If concept_name or brief_description has keywords in chunk -> (True, "")
        - Otherwise -> (False, "")
    """
    if not chunk_text:
        return False, ""

    t_clean = BookParser.repair_mojibake(chunk_text).lower()
    t_words = set(re.findall(r"\w{3,}", t_clean))

    raw_quote = (supporting_quote or "").strip()
    quote_grounded = False
    valid_quote = ""

    if raw_quote:
        clean_q = BookParser.repair_mojibake(raw_quote)
        q_lower = clean_q.lower()
        # 1. Exact or normalized whitespace substring check
        if " ".join(q_lower.split()) in " ".join(t_clean.split()):
            quote_grounded = True
            valid_quote = clean_q
        else:
            # 2. Significant word overlap check (ignoring tiny punctuation differences)
            q_words = set(re.findall(r"\w{4,}", q_lower))
            if not q_words:
                q_words = set(re.findall(r"\w{3,}", q_lower))
            if q_words and t_words:
                overlap = len(q_words & t_words) / len(q_words)
                if overlap >= 0.5:
                    quote_grounded = True
                    valid_quote = clean_q

    name_clean = BookParser.repair_mojibake(concept_name).lower()
    name_words = set(re.findall(r"\w{3,}", name_clean))
    name_overlap = bool(name_words & t_words)

    desc_words = set(re.findall(r"\w{4,}", (brief_description or "").lower()))
    desc_overlap = bool(desc_words and len(desc_words & t_words) >= 2)

    if quote_grounded:
        return True, valid_quote

    # Quote was provided but failed grounding
    if raw_quote:
        if name_overlap or desc_overlap:
            # Concept itself appears in chunk, but the quote was inaccurate -> keep concept without quote
            return True, ""
        # Both quote and concept name/description are completely absent
        return False, ""

    # No quote was provided
    if name_overlap or desc_overlap:
        return True, ""

    return False, ""


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
        request_timeout: int = 30,
        max_retries: int = 1,
        fallback_model: Optional[str] = None,
    ):
        self.model_name = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.cooldown_seconds = cooldown_seconds
        self.request_timeout = request_timeout
        self.max_retries = max_retries

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
        fallback_model: Optional[str] = None,
    ) -> "KnowledgeExtractor":
        if base_url:
            urls = [u.strip() for u in base_url.split(",") if u.strip()]
            pool = OllamaPool.from_urls(
                urls,
                cooldown_seconds=settings.failover_cooldown_seconds,
                max_tasks_per_server=1,  # Strictly sequential per Ollama server: no mutual execution
            )
        else:
            servers = getattr(settings, "resolved_llm_servers", None) or settings.resolved_ollama_servers
            pool = OllamaPool(
                servers=servers,
                cooldown_seconds=settings.failover_cooldown_seconds,
                max_tasks_per_server=1,  # Strictly sequential per Ollama server: no mutual execution
            )
        return cls(
            pool=pool,
            model=model or settings.llm_model,
            fallback_model=fallback_model or settings.llm_model_fallback,
            cooldown_seconds=settings.failover_cooldown_seconds,
            request_timeout=getattr(settings, "request_timeout", 30),
            max_retries=getattr(settings, "max_retries", 1),
        )

    def _execute_structured_invoke(
        self,
        schema_cls: Any,
        messages: Any,
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        model: Optional[str] = None,
        input_chars: int = 0,
        task_type: str = "llm",
    ) -> Any:
        """Execute structured output invocation with multi-server failover, timeout, and retries."""
        timeout_sec = str(self.request_timeout)
        timeout_float = float(self.request_timeout)
        effective_model = model or self.model_name

        if input_chars <= 0 and messages:
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    input_chars += len(content)

        def _invoke(url: str):
            # 1. Direct curl invocation with format="json" for maximum reliability with Ollama
            if shutil.which("curl"):
                try:
                    msgs_payload = []
                    for m in messages:
                        role = "user"
                        if hasattr(m, "type"):
                            role = "system" if m.type == "system" else "user"
                        content = getattr(m, "content", str(m))
                        msgs_payload.append({"role": role, "content": content})

                    body = {
                        "model": effective_model,
                        "messages": msgs_payload,
                        "format": "json",
                        "options": {
                            "temperature": self.temperature,
                            "top_p": 0.9,
                        },
                        "stream": False,
                    }
                    cmd = [
                        "curl",
                        "-s",
                        "--connect-timeout",
                        "3",
                        "--max-time",
                        timeout_sec,
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(body),
                        f"{url.rstrip('/')}/api/chat",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        data = json.loads(res.stdout)
                        content = data.get("message", {}).get("content", "")
                        if "<think>" in content and "</think>" in content:
                            content = content.split("</think>")[-1].strip()
                        if content and content.strip():
                            content = content.strip()
                            if content.startswith("```json"):
                                content = content[7:]
                            elif content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            content = content.strip()
                            if hasattr(schema_cls, "model_validate_json"):
                                return schema_cls.model_validate_json(content)
                            return json.loads(content)
                    elif res.returncode == 7:
                        raise ConnectionRefusedError(f"Ollama server {url} unreachable / connection refused (curl code 7)")
                    elif res.returncode == 28:
                        raise TimeoutError(f"Ollama request to {url}/api/chat timed out after {timeout_sec}s")
                    elif res.returncode != 0:
                        logger.debug(f"Direct curl format=json returned non-zero code {res.returncode}")
                except (TimeoutError, ConnectionError):
                    raise
                except Exception as curl_err:
                    logger.debug(f"Direct curl format=json invoke failed: {curl_err}")

            # 2. Fallback to ChatOllama structured output chain with strict timeout
            try:
                # Clean sampling parameters: pass only valid parameters (no mirostat, mirostat_eta, mirostat_tau, or tfs_z)
                llm = ChatOllama(
                    base_url=url,
                    model=effective_model,
                    temperature=self.temperature,
                    top_p=0.9,
                    sync_client_kwargs={"timeout": timeout_float},
                    client_kwargs={"timeout": timeout_float},
                )
                chain = llm.with_structured_output(schema_cls)
                return chain.invoke(messages)
            except Exception as exc:
                raise exc

        return self.pool.execute_with_failover(
            _invoke,
            retries=self.max_retries if retries is None else retries,
            max_task_duration=max(60.0, 2.5 * self.request_timeout),
            model_name=effective_model,
            quarantine_server=quarantine_server,
            exclude_urls=exclude_urls,
            capability="llm",
            input_chars=input_chars,
            task_type=task_type,
        )

    def clean_metadata(
        self,
        raw_title: str,
        raw_authors: List[str],
        raw_comments: Optional[str] = None,
        content_sample: Optional[str] = None,
        file_hint: Optional[str] = None,
        retries: Optional[int] = None,
        model: Optional[str] = None,
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

        meta_chars = len(raw_title) + sum(len(a) for a in raw_authors) + len(raw_comments or "") + len(content_sample or "")

        try:
            result = self._execute_structured_invoke(
                BookMetadata,
                messages,
                retries=retries,
                model=model,
                input_chars=meta_chars,
                task_type="llm",
            )
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
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        raise_on_error: bool = False,
        model: Optional[str] = None,
    ) -> SectionExtraction:
        """
        Extract canonical domain ideas, principles, patterns, and their relationships
        from a section or chunk with multi-server failover.
        Supports parent macro context for hierarchical RAG understanding.
        """
        system_text = (
            "You are an expert knowledge extractor, domain ontologist, and conceptual analyst.\n"
            "Your objective is to identify and extract 1 to 5 canonical ideas, concepts, themes, principles, "
            "strategies, lore elements, or domain mechanisms present in the provided text.\n\n"
            "EXTRACTION GUIDELINES:\n"
            "1. GROUNDING & EVIDENCE:\n"
            "   - All extracted concepts must be directly grounded in the Target Text.\n"
            "   - 'supporting_quote': MUST be an exact, verbatim sentence or phrase from the Target Text supporting the concept.\n"
            "   - Only return an empty list (\"concepts\": []) if the excerpt is purely administrative boilerplate (e.g. copyright notices, table of contents, ISBNs, page numbers).\n\n"
            "2. DOMAIN-APPROPRIATE CONCEPTS:\n"
            "   - For narrative literature & fiction: extract literary themes, character motifs, lore principles, tactical strategies, alliances, conflicts, cultural customs, or mythical concepts.\n"
            "   - For technical & engineering works: extract architectural patterns, algorithms, system principles, data structures, tradeoffs, or protocols.\n"
            "   - For general non-fiction: extract organizational principles, mental models, historical dynamics, or sociological concepts.\n"
            "   - Negative Rule: Do NOT artificially label narrative fiction events as technical engineering concepts (e.g. do not label character accidents as 'System Design Principle').\n\n"
            "3. SCHEMA REQUIREMENTS:\n"
            "   - 'name': Concise canonical title (2-4 words, capitalized noun phrase, e.g. 'Tactical Ambush', 'Divination Ritual', 'Consistent Hashing').\n"
            "   - 'brief_description': 1-2 sentence definition of the concept in context.\n"
            "   - 'detailed_explanation': Thorough explanation of how the concept functions, its mechanism, role, and nuances.\n"
            "   - 'category': Domain category (e.g. Tactical Strategy, Thematic Motif, Lore Concept, Architectural Pattern, Ethical Principle).\n"
            "   - 'supporting_quote': Exact verbatim quote from the text.\n"
            "   - 'related_concepts': List of related concept names.\n"
            "   - 'weight': Significance score from 1 to 10 (1-3: basic mention; 4-6: prominent recurring concept or motif; 7-10: major defining pillar or core paradigm).\n\n"
            "You MUST output valid JSON matching this schema:\n"
            "{\n"
            '  "concepts": [\n'
            "    {\n"
            '      "name": "...",\n'
            '      "brief_description": "...",\n'
            '      "detailed_explanation": "...",\n'
            '      "category": "...",\n'
            '      "supporting_quote": "...",\n'
            '      "related_concepts": ["..."],\n'
            '      "weight": 5\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        book_title = BookParser.repair_mojibake(book_title)
        section_title = BookParser.repair_mojibake(section_title)
        subtitle = BookParser.repair_mojibake(subtitle) if subtitle else ""
        text = BookParser.repair_mojibake(text)
        if parent_context:
            parent_context = BookParser.repair_mojibake(parent_context)

        loc = f"Book: '{book_title}'\nSection/Chapter: '{section_title}'"
        if subtitle and subtitle != section_title:
            loc += f"\nSubsection/Subtitle: '{subtitle}'"

        human_text = f"{loc}\n\n"
        human_text += (
            f"Target Text for Idea Extraction (MANDATORY: All concepts and verbatim quotes must come strictly from this text):\n"
            f"\"\"\"\n{text[:4000]}\n\"\"\"\n"
        )
        if parent_context and parent_context != text:
            human_text += (
                f"\nBackground Context (Surrounding passage provided ONLY for narrative orientation - DO NOT extract concepts or quotes from here):\n"
                f"\"\"\"\n{parent_context[:2000]}\n\"\"\"\n"
            )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]

        chunk_chars = len(text)

        try:
            result = self._execute_structured_invoke(
                SectionExtraction,
                messages,
                retries=retries,
                quarantine_server=quarantine_server,
                exclude_urls=exclude_urls,
                model=model,
                input_chars=chunk_chars,
                task_type="llm",
            )
            if isinstance(result, SectionExtraction):
                return result
            return SectionExtraction(**dict(result))
        except Exception as e:
            if raise_on_error:
                raise
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
