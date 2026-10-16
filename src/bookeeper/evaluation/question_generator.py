"""
Question Generation Subsystem for RAG and Knowledge Graph Evaluation.

Leverages the configured verification model (e.g., Llama 3.1 8B, Qwen 2.5 7B/14B)
to synthesize a discerning, multi-category question library from indexed books.
Stored in an editable JSON format allowing human revision before execution.
"""

import json
import logging
import re
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.extractor import OllamaPool

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
except ImportError:
    Console = None
    Panel = None
    Table = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TaskProgressColumn = None
    MofNCompleteColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None


class QuestionCategory(str, Enum):
    """Evaluation question categories for nuanced retrieval testing."""
    CONCEPT = "concept"          # In-depth definition, mechanics, or architectural pattern
    RELATIONAL = "relational"    # Interactions, trade-offs, or dependencies between concepts
    FACTUAL = "factual"          # Precise mechanisms, algorithmic steps, failure modes, metrics
    THEMATIC = "thematic"        # Broad thematic synthesis across chapters or sections


class ChecklistItem(BaseModel):
    """A verifiable invariant criterion evaluated as boolean (True/False)."""
    criterion: str
    expected: bool = True
    aspect_label: str = ""


class QuestionItem(BaseModel):
    """A single evaluation query with grounding context and binary verification checklist."""
    id: str
    book_id: Optional[Union[int, str]] = None
    book_title: str = ""
    category: str = "concept"
    query: str = ""
    context_hint: str = ""
    checklist: List[ChecklistItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("query") and data.get("question"):
                data["query"] = data["question"]
            elif not data.get("question") and data.get("query"):
                data["question"] = data["query"]
            if not data.get("checklist") and data.get("expected_aspects"):
                aspects = data.get("expected_aspects") or []
                data["checklist"] = [
                    {"criterion": f"Does the response affirm or explain '{asp}'?", "expected": True, "aspect_label": str(asp)}
                    for asp in aspects
                ]
        return data

    @property
    def question(self) -> str:
        return self.query

    @question.setter
    def question(self, val: str) -> None:
        self.query = val

    @property
    def expected_aspects(self) -> List[str]:
        return [c.aspect_label or c.criterion for c in self.checklist]


class QuestionLibrary(BaseModel):
    """Collection of evaluation questions for benchmarking Bookkeeper vs LightRAG."""
    created_at: float = Field(default_factory=time.time)
    model: str = "unknown"
    book_titles: List[str] = Field(default_factory=list)
    questions: List[QuestionItem] = Field(default_factory=list)

    def save(self, path: Union[Path, str]) -> Path:
        """Save question library to a formatted JSON file."""
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        return p

    @classmethod
    def load(cls, path: Union[Path, str]) -> "QuestionLibrary":
        """Load question library from a JSON file."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Question library file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


def _clean_json_str(text: str) -> str:
    """Clean model response removing thinking tags, markdown fences, and noise."""
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start_brace = text.find("{")
    start_bracket = text.find("[")
    if start_brace != -1 or start_bracket != -1:
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = text.rfind("}")
            if end_brace > start_brace:
                return text[start_brace : end_brace + 1]
        elif start_bracket != -1:
            end_bracket = text.rfind("]")
            if end_bracket > start_bracket:
                return text[start_bracket : end_bracket + 1]
    return text


class QuestionGenerator:
    """
    Generates high-quality evaluation questions using the dedicated verification model pool.
    """

    def __init__(
        self,
        settings: Settings,
        model_name: Optional[str] = None,
        pool: Optional[OllamaPool] = None,
        timeout: int = 120,
        console: Optional[Any] = None,
    ):
        self.settings = settings
        self.model_name = model_name or getattr(settings, "resolved_verifier_model", settings.llm_model)
        servers = getattr(settings, "resolved_verification_servers", None) or settings.resolved_ollama_servers
        cooldown = getattr(settings, "resolved_verification_cooldown_seconds", 60)
        self.pool = pool or OllamaPool(
            servers=servers,
            cooldown_seconds=cooldown,
            max_tasks_per_server=1,
        )
        self.timeout = timeout
        self.console = console or (Console() if Console else None)

    def display_setup_overview(self, books: List[Dict[str, Any]], questions_per_book: int) -> None:
        """Display selected book titles, pool nodes, and synthesis parameters."""
        if not self.console or not Table:
            return

        # 1. Selected Books Table
        books_table = Table(
            title=f"📚 Selected Book Titles for Question Library ({len(books)} books)",
            header_style="bold cyan",
            border_style="cyan",
            expand=True,
        )
        books_table.add_column("#", justify="center", style="dim", width=4)
        books_table.add_column("Book ID", justify="right", style="bold yellow", width=9)
        books_table.add_column("Book Title", style="bold white", ratio=3)
        books_table.add_column("Source / File Path", style="dim cyan", ratio=3)

        for i, b in enumerate(books, 1):
            bid = str(b.get("id", "-"))
            title = b.get("title", f"Book #{bid}")
            path = b.get("path") or "Exported in cache"
            p_str = Path(path).name if path and Path(path).is_file() else str(path)
            books_table.add_row(str(i), bid, title, p_str)

        self.console.print()
        self.console.print(books_table)
        self.console.print()

        # 2. Ollama Pool Nodes Table
        pool_table = Table(
            title=f"🤖 Ollama Verification Model Pool (Model: [bold white]{self.model_name}[/bold white])",
            header_style="bold magenta",
            border_style="magenta",
            expand=True,
        )
        pool_table.add_column("Prio", justify="center", style="bold", width=6)
        pool_table.add_column("Server Node", style="bold white", ratio=2)
        pool_table.add_column("Endpoint URL", style="cyan", ratio=3)
        pool_table.add_column("Capabilities", style="dim", ratio=2)
        pool_table.add_column("Node State", justify="center", width=18)

        status_list = self.pool.get_status()
        for node in status_list:
            eligible = node.get("status") == "active"
            status_text = "[bold green]● Active[/bold green]" if eligible else f"[yellow]● Cooldown ({node.get('cooldown_remaining_sec', 0):.0f}s)[/yellow]"
            caps = ", ".join(node.get("capabilities", [])) or "all"
            pool_table.add_row(
                str(node.get("priority", 1)),
                node.get("name", "Ollama Node"),
                node.get("url", ""),
                caps,
                status_text,
            )

        self.console.print(pool_table)
        self.console.print(
            f"[dim]Synthesis Target: [bold white]{len(books) * questions_per_book}[/bold white] total questions "
            f"([bold white]{questions_per_book}[/bold white] per book across concept, relational, factual, and thematic)[/dim]\n"
        )

    def _call_llm_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Call verification model with structured JSON formatting and curl fallback."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _invoke(node_or_url: Any) -> Dict[str, Any]:
            base_url = getattr(node_or_url, "target_url", getattr(node_or_url, "url", str(node_or_url)))
            url = f"{base_url.rstrip('/')}/api/chat"
            body = {
                "model": self.model_name,
                "messages": messages,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
                "stream": False,
            }

            raw_stdout = ""
            if shutil.which("curl"):
                cmd = [
                    "curl",
                    "-s",
                    "--connect-timeout", "4",
                    "--max-time", str(self.timeout),
                    "-X", "POST",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(body),
                    url,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0 and res.stdout.strip():
                    raw_stdout = res.stdout.strip()

            if not raw_stdout:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_bytes = resp.read()
                    raw_stdout = raw_bytes.decode("utf-8", errors="replace").strip()

            if raw_stdout:
                data = json.loads(raw_stdout)
                if "error" in data:
                    raise RuntimeError(f"Ollama server {url} error: {data['error']}")
                content = data.get("message", {}).get("content", "")
                cleaned = _clean_json_str(content)
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return {"questions": parsed}
                return parsed

            raise RuntimeError(f"Ollama server {url} returned empty response or timed out after {self.timeout}s")

        return self.pool.execute_with_failover(
            _invoke,
            model_name=self.model_name,
            input_chars=len(prompt),
            max_task_duration=float(self.timeout),
            task_type="verification",
        )

    def extract_book_sample_context(self, file_path: Union[Path, str], max_chars: int = 12000) -> Tuple[List[str], str]:
        """
        Extract outline (chapter/section titles) and sampled excerpt text from book.
        """
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            return [], ""

        ext = p.suffix.lower()
        if ext in (".txt", ".md"):
            raw = p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
            return ["Full Text"], raw

        sections = BookParser.parse(p)
        titles = [s.title for s in sections if s.title]

        sampled_parts = []
        chars_collected = 0
        # Sample across beginning, middle, and later chapters
        step = max(1, len(sections) // 6)
        for idx in range(0, len(sections), step):
            s = sections[idx]
            txt = s.text.strip() if s.text else ""
            if len(txt) > 80:
                header = f"\n[Chapter/Section: {s.title}]\n" if s.title else "\n"
                snippet = txt[:1800]
                sampled_parts.append(f"{header}{snippet}")
                chars_collected += len(snippet)
                if chars_collected >= max_chars:
                    break

        return titles[:20], "".join(sampled_parts)

    def generate_questions_for_book(
        self,
        book_id: Union[int, str],
        book_title: str,
        book_path: Optional[Union[Path, str]] = None,
        questions_count: int = 5,
        start_index: int = 1,
    ) -> List[QuestionItem]:
        """
        Generate questions_count discerning questions for a given book using the verification model.
        """
        outline: List[str] = []
        sample_text = ""
        if book_path:
            try:
                outline, sample_text = self.extract_book_sample_context(book_path)
            except Exception as e:
                logger.warning(f"Could not extract outline from {book_path}: {e}")

        system_prompt = (
            "You are a principal technical author and AI evaluation auditor designing benchmark queries "
            "for testing knowledge graphs and retrieval-augmented generation systems across large digital libraries.\n"
            "CRITICAL ARCHITECTURAL CONSTRAINTS:\n"
            "1. STRICT SUBJECT-MATTER QUERIES ONLY: Queries will be evaluated against a unified knowledge base across many books without book or author metadata. Context about author and book title is NOT available to the search engine. Therefore, questions MUST be framed purely around concrete subject matter, mechanisms, algorithms, models, and theories. NEVER use meta-references like 'in this book', 'the author', 'the current paper', 'according to the text', 'this chapter', etc.\n"
            "2. BINARY VERIFICATION CHECKLIST: For each query, formulate 3 to 4 objective, falsifiable checklist criteria that will be checked directly against the retrieved answer (acting as context). Each checklist item must be a direct Yes/No question testing whether the response affirms a required core mechanism (expected: true) or avoids an erroneous misconception/distractor (expected: false)."
        )

        user_prompt = f"""
Domain Context / Source Material:
Book Title: "{book_title}"
{f"Table of Contents / Key Chapters: {', '.join(outline)}" if outline else ""}

Sample Excerpts:
---
{sample_text[:8000] if sample_text else f"Technical domain material on {book_title}."}
---

Generate exactly {questions_count} diverse, challenging benchmark questions testing deep understanding of this subject matter.
Distribute questions across these 4 categories:
1. "concept": Definition, mechanics, and core logic of key ideas or patterns.
2. "relational": How two concepts interact, compare, or form trade-offs.
3. "factual": Specific algorithms, concrete failure modes, guarantees, or properties.
4. "thematic": Broad principles or synthesis across domain topics.

CRITICAL INSTRUCTIONS:
- Write the "query" purely in terms of the subject matter concepts (e.g. "What causes write amplification in Log-Structured Merge (LSM) trees?", "Why does spline-based difficulty modeling conclude that scaling human task execution time from 3 to 30 minutes represents a smaller jump in AI difficulty than scaling from 30 minutes to 5 hours?"). Do NOT mention "the book", "the author", "in chapter", etc.
- Provide a "checklist" with 3-4 boolean verification questions to grade the response.

Respond with ONLY a valid JSON object matching this schema:
{{
  "questions": [
    {{
      "category": "concept|relational|factual|thematic",
      "query": "Direct subject-matter query...",
      "context_hint": "Domain subtopic or mechanism name",
      "checklist": [
        {{
          "criterion": "Does the response explain that ...?",
          "expected": true,
          "aspect_label": "Key mechanism"
        }},
        {{
          "criterion": "Does the response state that ...?",
          "expected": false,
          "aspect_label": "Negative distractor check"
        }}
      ]
    }}
  ]
}}
"""
        try:
            res_json = self._call_llm_json(user_prompt, system_prompt=system_prompt)
            raw_qs = res_json.get("questions", [])
        except Exception as e:
            logger.error(f"LLM question generation failed for '{book_title}': {e}. Using deterministic fallback.")
            raw_qs = []

        # If LLM failed or returned empty list, provide high-quality fallback questions based on outline/title
        if not raw_qs:
            fallback_concepts = outline[:3] if outline else [book_title]
            raw_qs = [
                {
                    "category": "concept",
                    "query": f"What are the foundational principles, architecture, and mechanisms of {fallback_concepts[0]}?",
                    "context_hint": f"Core concepts of {fallback_concepts[0]}",
                    "checklist": [
                        {
                            "criterion": f"Does the response explain the core definitions and operational model of {fallback_concepts[0]}?",
                            "expected": True,
                            "aspect_label": "Core definitions",
                        },
                        {
                            "criterion": "Does the response identify practical trade-offs or constraints?",
                            "expected": True,
                            "aspect_label": "Trade-off analysis",
                        },
                    ],
                },
                {
                    "category": "thematic",
                    "query": f"How do system architecture trade-offs balance complexity, consistency, and performance in {fallback_concepts[-1]}?",
                    "context_hint": f"Thematic trade-offs in {fallback_concepts[-1]}",
                    "checklist": [
                        {
                            "criterion": "Does the response analyze consistency or reliability guarantees?",
                            "expected": True,
                            "aspect_label": "Guarantees",
                        },
                        {
                            "criterion": "Does the response detail performance or scalability implications?",
                            "expected": True,
                            "aspect_label": "Scalability",
                        },
                    ],
                },
            ]

        items: List[QuestionItem] = []
        for i, q in enumerate(raw_qs[:questions_count]):
            q_id = f"q_{start_index + i:03d}"
            cat = q.get("category", "concept").lower()
            if cat not in ("concept", "relational", "factual", "thematic"):
                cat = "concept"

            q_text = q.get("query") or q.get("question") or ""
            # Strip residual meta-references if any were hallucinated by the model
            q_text = re.sub(
                r"(?i)\b(in this book|according to the authors?|the authors? argue[sd]? that|the authors? state[sd]? that|in this paper|in chapter \d+)\b,?\s*",
                "",
                q_text,
            ).strip()
            if q_text:
                q_text = q_text[0].upper() + q_text[1:]

            checklist_raw = q.get("checklist", [])
            checklist_items = []
            for c in checklist_raw:
                if isinstance(c, dict) and c.get("criterion"):
                    checklist_items.append(
                        ChecklistItem(
                            criterion=c.get("criterion", "").strip(),
                            expected=bool(c.get("expected", True)),
                            aspect_label=c.get("aspect_label", "").strip(),
                        )
                    )
            # If no checklist items were returned, convert expected_aspects if available
            if not checklist_items:
                for asp in q.get("expected_aspects", []):
                    checklist_items.append(
                        ChecklistItem(
                            criterion=f"Does the response affirm or explain '{asp}'?",
                            expected=True,
                            aspect_label=str(asp),
                        )
                    )

            items.append(
                QuestionItem(
                    id=q_id,
                    book_id=book_id,
                    book_title=book_title,
                    category=cat,
                    query=q_text,
                    context_hint=q.get("context_hint", "").strip(),
                    checklist=checklist_items,
                )
            )

        return items

    def generate_library(
        self,
        books: List[Dict[str, Any]],
        questions_per_book: int = 5,
        output_file: Optional[Union[Path, str]] = None,
        show_progress: bool = True,
    ) -> QuestionLibrary:
        """
        Generate a comprehensive question library across N books with live progress feedback.
        """
        if show_progress and self.console:
            self.display_setup_overview(books, questions_per_book)

        all_questions: List[QuestionItem] = []
        titles = []
        q_idx = 1

        if show_progress and self.console and Progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=32),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                main_task = progress.add_task(
                    "[bold cyan]Synthesizing question library across books...[/bold cyan]",
                    total=len(books),
                )
                for i, b in enumerate(books, 1):
                    bid = b.get("id")
                    title = b.get("title", f"Book #{bid}")
                    path = b.get("path")
                    titles.append(title)

                    short_title = (title[:40] + "...") if len(title) > 43 else title
                    progress.update(
                        main_task,
                        description=f"[bold cyan][{i}/{len(books)}][/bold cyan] Analyzing [bold white]{short_title}[/bold white]",
                    )

                    t0 = time.perf_counter()
                    qs = self.generate_questions_for_book(
                        book_id=bid,
                        book_title=title,
                        book_path=path,
                        questions_count=questions_per_book,
                        start_index=q_idx,
                    )
                    elapsed = time.perf_counter() - t0
                    all_questions.extend(qs)
                    q_idx += len(qs)

                    last_server = self.pool.get_last_used_server() or "Ollama Node"
                    progress.advance(main_task)

                    self.console.print(
                        f"  [bold green]✓[/bold green] [dim][{i}/{len(books)}][/dim] "
                        f"Synthesized [bold white]{len(qs)}[/bold white] questions for "
                        f"'[bold white]{title}[/bold white]' "
                        f"[dim]({last_server} in {elapsed:.1f}s)[/dim]"
                    )
        else:
            for b in books:
                bid = b.get("id")
                title = b.get("title", f"Book #{bid}")
                path = b.get("path")
                titles.append(title)

                logger.info(f"Generating questions for '{title}' (model: {self.model_name})...")
                qs = self.generate_questions_for_book(
                    book_id=bid,
                    book_title=title,
                    book_path=path,
                    questions_count=questions_per_book,
                    start_index=q_idx,
                )
                all_questions.extend(qs)
                q_idx += len(qs)

        library = QuestionLibrary(
            created_at=time.time(),
            model=self.model_name,
            book_titles=titles,
            questions=all_questions,
        )

        if output_file:
            library.save(output_file)

        return library
