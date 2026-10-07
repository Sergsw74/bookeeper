"""
Production-quality Typer CLI interface for bookeeper with rich progress bars and configurable Ollama / Calibre paths.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings, get_settings
from bookeeper.graph.exporters import GraphMLExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import KnowledgeExtractor

app = typer.Typer(
    name="bookeeper",
    help="Connects Calibre libraries to local/remote Ollama LLMs and extracts Concept Knowledge Graphs.",
    add_completion=False,
)
console = Console()


def _get_effective_settings(
    config_path: Optional[str] = None,
    calibre_path: Optional[str] = None,
    ollama_url: Optional[str] = None,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Settings:
    """Retrieve settings and merge explicit CLI arguments with highest precedence."""
    cfg = get_settings(config_path)
    overrides = {}
    if calibre_path:
        overrides["calibre_library_path"] = calibre_path
    if ollama_url:
        overrides["ollama_base_url"] = ollama_url
    if model:
        overrides["llm_model"] = model
    if embedding_model:
        overrides["embedding_model"] = embedding_model

    if overrides:
        data = cfg.model_dump()
        data.update(overrides)
        return Settings(**data)
    return cfg


def _perform_ollama_warmup(extractor: KnowledgeExtractor, console: Console) -> Dict[str, Any]:
    """Execute warmup ping to load model and report CPU vs GPU acceleration status with warnings."""
    with console.status(f"[bold blue]Checking Ollama acceleration for '{extractor.model_name}'...[/bold blue]"):
        status = extractor.warmup_and_check_device()

    device = status.get("device", "Unknown")
    is_gpu = status.get("is_gpu", False)
    vram_mb = status.get("size_vram", 0) / (1024 * 1024)
    size_mb = status.get("size", 0) / (1024 * 1024)

    if not is_gpu:
        warning_msg = (
            f"[bold yellow]⚠️ Ollama is executing model '[cyan]{status.get('model')}[/cyan]' entirely on [bold red]CPU[/bold red][/bold yellow]\n\n"
            f"• [bold]VRAM Allocated:[/bold] 0 MB / {size_mb:.0f} MB (0% offloaded)\n"
            f"• [bold]Inference Runner:[/bold] {status.get('runner', 'llamacpp')}\n\n"
            f"[dim]Note: Extraction will be noticeably slower on CPU than with GPU acceleration (CUDA, ROCm, or Metal).\n"
            f"If your Ollama server has a dedicated GPU, verify NVIDIA drivers / Container Toolkit or Ollama GPU permissions.[/dim]"
        )
        console.print(
            Panel(
                warning_msg,
                title="[bold yellow]Hardware Acceleration Notice[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        vram_pct = status.get("vram_pct", 100.0)
        console.print(
            f"[bold green]✓ Ollama GPU acceleration active:[/bold green] [bold cyan]{device}[/bold cyan] "
            f"({vram_mb:.0f} MB / {size_mb:.0f} MB VRAM, {vram_pct}% offloaded)"
        )

    return status


@app.command()
def config(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to custom config.yaml file."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Override Calibre library path or SMB share mount."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Override Ollama base URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override Ollama LLM model name (e.g. llama3.1:8b)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Override Ollama embedding model (e.g. nomic-embed-text)."
    ),
):
    """Display the active bookeeper configuration settings."""
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model, embedding_model)
    console.print(
        Panel.fit(
            f"[bold green]Calibre Library / SMB Share:[/bold green] {cfg.calibre_library_path}\n"
            f"[bold green]Calibre Auth:[/bold green] user={cfg.calibre_user or '[dim]none[/dim]'}\n"
            f"[bold green]Ollama Endpoint:[/bold green] {cfg.ollama_base_url}\n"
            f"[bold green]LLM Model:[/bold green] {cfg.llm_model}\n"
            f"[bold green]Embedding Model:[/bold green] {cfg.embedding_model}\n"
            f"[bold green]Similarity Threshold:[/bold green] {cfg.similarity_threshold}\n"
            f"[bold green]Output Directory:[/bold green] {cfg.resolved_output_dir}",
            title="Active Configuration",
        )
    )


@app.command("list-books")
def list_books(
    limit: Optional[int] = typer.Option(
        25, "--limit", "-n", help="Maximum number of books to display (use 0 for all)."
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Filter books by title or author query."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    List and search books in the configured Calibre library or mounted SMB share.
    Useful for testing database connectivity and discovering book IDs.
    """
    cfg = _get_effective_settings(config_path, calibre_path=calibre_path)
    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] Cannot access Calibre library at '[bold cyan]{cfg.calibre_library_path}[/bold cyan]'.\n"
            f"If this is on a remote SMB share, ensure the share is mounted (e.g. /Volumes/...) or that calibredb is installed."
        )
        raise typer.Exit(1)

    with console.status(f"[bold blue]Connecting to Calibre library at {cfg.calibre_library_path}...[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "formats"])
        except Exception as e:
            console.print(f"[bold red]Failed to read Calibre library:[/bold red] {e}")
            raise typer.Exit(1)

    if search:
        q = search.lower()
        books = [
            b
            for b in books
            if q in b.get("title", "").lower()
            or any(q in a.lower() for a in b.get("authors", []))
        ]

    total_count = len(books)
    if limit and limit > 0:
        books = books[:limit]

    table = Table(
        title=f"Calibre Library: {cfg.calibre_library_path} (Showing {len(books)} of {total_count})",
        show_lines=False,
    )
    table.add_column("ID", justify="right", style="cyan", width=6)
    table.add_column("Title", style="bold white", min_width=30)
    table.add_column("Authors", style="green", width=24)
    table.add_column("Formats", style="magenta", width=16)

    for b in books:
        authors = b.get("authors", [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        formats = b.get("formats", [])
        formats_str = ", ".join(formats) if isinstance(formats, list) else str(formats)

        title = b.get("title", "Untitled")
        if len(title) > 50:
            title = title[:47] + "..."

        table.add_row(
            str(b.get("id", "")),
            title,
            authors_str[:24],
            formats_str or "[dim]None[/dim]",
        )

    console.print(table)
    if limit and total_count > limit:
        console.print(
            f"[dim]Showing first {limit} books. Use --limit 0 to view all {total_count} books, or --search <term>.[/dim]"
        )


@app.command("clean-metadata")
def clean_metadata(
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Specific Calibre book ID to clean."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Inspect and display cleaned metadata without writing back."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Ollama server URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Ollama LLM model name (e.g. llama3.1:8b, qwen2.5:7b-instruct)."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    skip_warmup: bool = typer.Option(
        False, "--skip-warmup", help="Skip Ollama warmup and GPU acceleration check."
    ),
):
    """
    Query Calibre (local, SMB share, or server), inspect titles/authors/summaries,
    prompt Ollama to normalize, and write clean metadata back.
    """
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model)
    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] Cannot access Calibre library at '{cfg.calibre_library_path}'.\n"
            f"Ensure the SMB share is mounted or calibredb is in PATH."
        )
        raise typer.Exit(1)

    extractor = KnowledgeExtractor(
        base_url=cfg.ollama_base_url,
        model=cfg.llm_model,
    )

    if not skip_warmup:
        _perform_ollama_warmup(extractor, console)

    with console.status("[bold blue]Querying Calibre library...[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "comments"])
        except Exception as e:
            console.print(f"[bold red]Failed to query Calibre library:[/bold red] {e}")
            raise typer.Exit(1)

    if book_id is not None:
        books = [b for b in books if b.get("id") == book_id]
        if not books:
            console.print(f"[bold yellow]Book ID {book_id} not found in Calibre library.[/bold yellow]")
            raise typer.Exit(1)

    console.print(f"[bold green]Found {len(books)} book(s) to process.[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Normalizing metadata with Ollama...", total=len(books))

        for b in books:
            bid = b["id"]
            raw_title = b.get("title", "Untitled")
            raw_authors = b.get("authors", [])
            raw_comments = b.get("comments", "")
            book_path_str = b.get("path")

            progress.update(task, description=f"Cleaning: [bold cyan]{raw_title[:30]}[/bold cyan]")

            # Sample book content and file hint from filesystem / SMB share
            content_sample = None
            file_hint = None
            if book_path_str:
                book_dir = Path(book_path_str)
                if book_dir.exists():
                    content_sample, file_hint = BookParser.sample_content(book_dir)

            # Run Ollama structured normalization with content sample
            cleaned = extractor.clean_metadata(
                raw_title=raw_title,
                raw_authors=raw_authors,
                raw_comments=raw_comments,
                content_sample=content_sample,
                file_hint=file_hint,
            )

            # Display diff panel
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Field", style="dim", width=12)
            table.add_column("Original Calibre Value")
            table.add_column("Cleaned LLM Value", style="bold green")

            table.add_row("Title", raw_title, cleaned.title)
            table.add_row("Authors", ", ".join(raw_authors), cleaned.author)
            table.add_row(
                "Summary",
                (raw_comments[:120] + "...") if raw_comments else "[dim]None[/dim]",
                cleaned.summary,
            )
            if file_hint:
                table.add_row("Source File", file_hint, "[dim green]Content sampled[/dim green]" if content_sample else "[dim]Inspected[/dim]")

            console.print(Panel(table, title=f"Book #{bid} Metadata Diff"))

            if not dry_run:
                try:
                    client.update_metadata(
                        book_id=bid,
                        title=cleaned.title,
                        authors=[cleaned.author],
                        comments=cleaned.summary,
                    )
                    console.print(f"[green]✓ Successfully updated book #{bid} in Calibre.[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Failed to update book #{bid}: {e}[/red]")
            else:
                console.print(f"[yellow]⚡ [Dry-Run] Skipped writing back to Calibre.[/yellow]")

            progress.advance(task)

    console.print("[bold green]Metadata cleaning process completed.[/bold green]")


@app.command("build-graph")
def build_graph(
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Process specific Calibre book ID."
    ),
    file_path: Optional[str] = typer.Option(
        None, "--file", "-f", help="Process a standalone EPUB or PDF file directly."
    ),
    all_books: bool = typer.Option(
        False, "--all", "-a", help="Process all books in the Calibre library."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Ollama server URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Ollama LLM model name (e.g. llama3.1:8b, qwen2.5:7b-instruct)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Ollama embedding model name (e.g. nomic-embed-text)."
    ),
    export_obsidian: Optional[str] = typer.Option(
        None, "--export-obsidian", help="Custom destination directory for Obsidian Markdown vault."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    skip_warmup: bool = typer.Option(
        False, "--skip-warmup", help="Skip Ollama warmup and GPU acceleration check."
    ),
):
    """
    Ingest sections, perform semantic chunking, extract concepts via Ollama,
    deduplicate entities, and export the Concept Knowledge Graph.
    """
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model, embedding_model)
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "knowledge_graph.json"

    store = ConceptGraphStore()
    if graph_file.is_file():
        try:
            store.load(graph_file)
            console.print(f"[dim]Loaded existing graph with {store.graph.number_of_nodes()} nodes.[/dim]")
        except Exception:
            pass

    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    console.print(
        f"[dim]Configured Ollama: [bold cyan]{cfg.ollama_base_url}[/bold cyan] | "
        f"LLM: [bold cyan]{cfg.llm_model}[/bold cyan] | "
        f"Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan][/dim]"
    )

    extractor = KnowledgeExtractor(
        base_url=cfg.ollama_base_url,
        model=cfg.llm_model,
    )

    if not skip_warmup:
        _perform_ollama_warmup(extractor, console)

    deduplicator = EntityDeduplicator.from_settings(
        cfg,
        base_url=cfg.ollama_base_url,
        embedding_model=cfg.embedding_model,
        similarity_threshold=cfg.similarity_threshold,
    )

    # Initialize HierarchicalChunker with Ollama embeddings if reachable
    try:
        embeddings = OllamaEmbeddings(
            base_url=cfg.ollama_base_url.rstrip("/"),
            model=cfg.embedding_model,
        )
        chunker = HierarchicalChunker(embeddings=embeddings)
    except Exception as e:
        console.print(f"[dim yellow]Warning: Remote Ollama embeddings init skipped ({e}); using paragraph chunker.[/dim yellow]")
        chunker = HierarchicalChunker()

    books_to_process = []

    # Case 1: Standalone file passed
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {p}")
            raise typer.Exit(1)
        books_to_process.append({"id": 1, "title": p.stem, "author": "Unknown", "path": p})

    # Case 2: From Calibre / SMB share
    elif client.is_available():
        with console.status("[bold blue]Querying Calibre library...[/bold blue]"):
            all_calibre_books = client.list_books(fields=["id", "title", "authors", "formats"])

        if book_id is not None:
            matches = [b for b in all_calibre_books if b["id"] == book_id]
            if not matches:
                console.print(f"[bold red]Book ID {book_id} not found in Calibre library.[/bold red]")
                raise typer.Exit(1)
            target_list = matches
        elif all_books:
            target_list = all_calibre_books
        else:
            console.print(
                "[bold yellow]Please specify --book-id <ID>, --file <path>, or --all to build graph.[/bold yellow]"
            )
            raise typer.Exit(1)

        for b in target_list:
            bid = b["id"]
            export_dir = output_dir / "calibre_ingest"
            epub_path = client.export_book(bid, target_dir=export_dir, fmt="EPUB")
            if epub_path and epub_path.is_file():
                books_to_process.append(
                    {
                        "id": bid,
                        "title": b.get("title", f"Book {bid}"),
                        "author": ", ".join(b.get("authors", [])) or "Unknown",
                        "path": epub_path,
                    }
                )
            else:
                console.print(f"[yellow]Skipping book #{bid}: No EPUB format available.[/yellow]")
    else:
        console.print(
            "[bold red]Calibre not available. Pass an explicit book file via --file <path.epub> "
            "or mount the SMB share and pass --calibre-path <path>.[/bold red]"
        )
        raise typer.Exit(1)

    if not books_to_process:
        console.print("[bold yellow]No books available to process.[/bold yellow]")
        raise typer.Exit(0)

    console.print(f"[bold green]Starting pipeline for {len(books_to_process)} book(s)...[/bold green]")

    for binfo in books_to_process:
        bid = binfo["id"]
        btitle = binfo["title"]
        bauthor = binfo["author"]
        bpath: Path = binfo["path"]

        console.print(f"\n[bold blue]► Ingesting Book #{bid}: {btitle}[/bold blue] ({bpath.name})")
        store.add_book(bid, title=btitle, author=bauthor)

        # Parse sections
        sections = BookParser.parse(bpath)
        console.print(f"  Extracted [green]{len(sections)} sections/chapters[/green].")

        # Chunk sections
        chunks = chunker.chunk_book(sections, book_id=bid, book_title=btitle)
        console.print(f"  Created [green]{len(chunks)} atomic thematic chunks[/green].")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Extracting concepts from '{btitle[:25]}'...", total=len(chunks))

            for chk in chunks:
                progress.update(
                    task,
                    description=f"Processing: [cyan]{chk.section_title[:20]} [p{chk.chunk_idx}][/cyan]",
                )

                # Ensure section exists in graph
                sec_node_id = store.add_section(
                    book_id=bid,
                    chapter_idx=chk.chapter_idx,
                    title=chk.section_title,
                    text=chk.text,
                )

                # Run Ollama Concept Extraction
                extraction = extractor.extract_section(
                    text=chk.text,
                    book_title=btitle,
                    section_title=chk.section_title,
                )

                # Deduplicate and register concepts
                for concept in extraction.concepts:
                    canonical_concept = deduplicator.resolve_concept(concept)
                    store.add_concept(canonical_concept)

                    # Link Section -> DISCUSSES -> Concept
                    store.add_section_concept_link(
                        section_node_id=sec_node_id,
                        concept_name=canonical_concept.name,
                        summary=canonical_concept.summary,
                    )

                    # Link related concepts
                    for rel_name in canonical_concept.related_concepts:
                        store.add_concept_relation(
                            src_concept_name=canonical_concept.name,
                            tgt_concept_name=rel_name,
                        )

                progress.advance(task)

    # Persist graph to JSON
    store.save(graph_file)
    console.print(f"\n[bold green]✓ Knowledge Graph persisted to:[/bold green] {graph_file}")

    # Export to Obsidian Vault
    vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
    obs = ObsidianExporter(vault_dest)
    obs.export(store)
    console.print(f"[bold green]✓ Obsidian Markdown Vault exported to:[/bold green] {vault_dest}")

    # Export GraphML
    graphml_dest = output_dir / "knowledge_graph.graphml"
    gml = GraphMLExporter(graphml_dest)
    gml.export(store)
    console.print(f"[bold green]✓ GraphML exported to:[/bold green] {graphml_dest}")

    # Display final graph statistics
    st = store.stats()
    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Build Complete",
        )
    )


@app.command()
def stats(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
):
    """Display metrics for the current Knowledge Graph."""
    cfg = get_settings(config_path)
    store = ConceptGraphStore()
    graph_file = cfg.resolved_output_dir / "knowledge_graph.json"
    if not graph_file.is_file():
        console.print(f"[yellow]No graph found at {graph_file}[/yellow]")
        return

    store.load(graph_file)
    st = store.stats()
    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Metrics",
        )
    )


if __name__ == "__main__":
    app()
