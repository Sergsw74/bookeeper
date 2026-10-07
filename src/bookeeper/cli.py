"""
Production-quality Typer CLI interface for bookeeper with rich progress bars and formatting.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser
from bookeeper.config import get_settings
from bookeeper.graph.exporters import GraphMLExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import KnowledgeExtractor

app = typer.Typer(
    name="bookeeper",
    help="Connects Calibre libraries to local Ollama LLMs and extracts Concept Knowledge Graphs.",
    add_completion=False,
)
console = Console()


@app.command()
def config(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to custom config.yaml file."
    )
):
    """Display the active bookeeper configuration settings."""
    cfg = get_settings(config_path)
    console.print(
        Panel.fit(
            f"[bold green]Calibre Library:[/bold green] {cfg.calibre_library_path}\n"
            f"[bold green]Calibre Auth:[/bold green] user={cfg.calibre_user or '[dim]none[/dim]'}\n"
            f"[bold green]Ollama Endpoint:[/bold green] {cfg.ollama_base_url}\n"
            f"[bold green]LLM Model:[/bold green] {cfg.llm_model}\n"
            f"[bold green]Embedding Model:[/bold green] {cfg.embedding_model}\n"
            f"[bold green]Similarity Threshold:[/bold green] {cfg.similarity_threshold}\n"
            f"[bold green]Output Directory:[/bold green] {cfg.resolved_output_dir}",
            title="Active Configuration",
        )
    )


@app.command("clean-metadata")
def clean_metadata(
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Specific Calibre book ID to clean."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Inspect and display cleaned metadata without writing back."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    Query Calibre, inspect dirty/missing titles, authors, and summaries,
    prompt Ollama to normalize, and write clean metadata back to Calibre.
    """
    cfg = get_settings(config_path)
    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] calibredb command not found or library '{cfg.calibre_library_path}' unavailable."
        )
        raise typer.Exit(1)

    extractor = KnowledgeExtractor.from_settings(cfg)

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

            progress.update(task, description=f"Cleaning: [bold cyan]{raw_title[:30]}[/bold cyan]")

            # Run Ollama structured normalization
            cleaned = extractor.clean_metadata(raw_title, raw_authors, raw_comments)

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
    export_obsidian: Optional[str] = typer.Option(
        None, "--export-obsidian", help="Custom destination directory for Obsidian Markdown vault."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    Ingest sections, perform semantic chunking, extract concepts via Ollama,
    deduplicate entities, and export the Concept Knowledge Graph.
    """
    cfg = get_settings(config_path)
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

    extractor = KnowledgeExtractor.from_settings(cfg)
    deduplicator = EntityDeduplicator.from_settings(cfg)
    chunker = HierarchicalChunker()

    books_to_process = []

    # Case 1: Standalone file passed
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {p}")
            raise typer.Exit(1)
        books_to_process.append({"id": 1, "title": p.stem, "author": "Unknown", "path": p})

    # Case 2: From Calibre
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
            "[bold red]Calibre not available. Pass an explicit book file via --file <path.epub>.[/bold red]"
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
