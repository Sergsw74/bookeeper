"""
Typer-based CLI interface for bookeeper.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser
from bookeeper.config import get_settings
from bookeeper.graph.exporters import CytoscapeExporter, GraphMLExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import SemanticChunker
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
        None, "--config", "-c", help="Path to config.yaml file."
    )
):
    """Display the active bookeeper configuration settings."""
    cfg = get_settings(config_path)
    console.print(
        Panel.fit(
            f"[bold green]Calibre Library:[/bold green] {cfg.calibre.library_path}\n"
            f"[bold green]Ollama Endpoint:[/bold green] {cfg.ollama.base_url} (model: {cfg.ollama.model})\n"
            f"[bold green]Embedding Model:[/bold green] {cfg.ollama.embedding_model}\n"
            f"[bold green]Chunk Size:[/bold green] {cfg.processing.chunk_size} chars (overlap: {cfg.processing.chunk_overlap})\n"
            f"[bold green]Graph Backend:[/bold green] {cfg.graph.backend} ({cfg.graph.storage_path})",
            title="Active Configuration",
        )
    )


@app.command()
def list_books(
    limit: Optional[int] = typer.Option(20, "--limit", "-n", help="Max books to show."),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter by query."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
):
    """List books available in the configured Calibre library."""
    cfg = get_settings(config_path)
    client = CalibreClient(cfg.calibre)

    if not client.is_available():
        console.print(f"[bold red]Error:[/bold red] Calibre library not found at {client.db_path}")
        raise typer.Exit(1)

    books = client.search_books(search) if search else client.list_books(limit=limit)

    table = Table(title=f"Calibre Books ({len(books)} found)")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Title", style="bold white")
    table.add_column("Authors", style="green")
    table.add_column("Formats", style="magenta")

    for b in books:
        table.add_row(
            str(b.id),
            b.title[:45] + ("..." if len(b.title) > 45 else ""),
            b.author_display[:30],
            ", ".join(b.formats.keys()) or "[dim]None[/dim]",
        )

    console.print(table)


@app.command()
def chunk(
    file_path: str = typer.Argument(..., help="Path to EPUB or PDF file."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
):
    """Parse an ebook and display extracted TOC sections and chunk statistics."""
    cfg = get_settings(config_path)
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        console.print(f"[bold red]File not found:[/bold red] {path}")
        raise typer.Exit(1)

    console.print(f"[bold blue]Parsing ebook:[/bold blue] {path.name}")
    sections = BookParser.parse(path)
    console.print(f"[green]Extracted {len(sections)} sections/chapters.[/green]")

    chunker = SemanticChunker(cfg.processing)
    chunks = chunker.chunk_sections(sections, book_id=1, book_title=path.stem)
    console.print(f"[green]Generated {len(chunks)} text chunks.[/green]")

    table = Table(title="Sample Chunks (First 5)")
    table.add_column("Chunk ID", style="cyan")
    table.add_column("Chapter", style="yellow")
    table.add_column("Length (chars)", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Excerpt", style="dim")

    for c in chunks[:5]:
        table.add_row(
            c.chunk_id,
            c.chapter_title[:25],
            str(c.character_count),
            str(c.word_count),
            c.text[:60].replace("\n", " ") + "...",
        )

    console.print(table)


@app.command()
def stats(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
):
    """Display statistics for the persisted Concept Knowledge Graph."""
    cfg = get_settings(config_path)
    store = ConceptGraphStore(cfg.graph)
    if not Path(cfg.graph.storage_path).is_file():
        console.print(f"[yellow]No existing graph found at {cfg.graph.storage_path}[/yellow]")
        return

    store.load()
    s = store.stats()

    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {s['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {s['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in s["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in s["edge_types"].items()),
            title="Knowledge Graph Statistics",
        )
    )


@app.command()
def export(
    output_dir: Optional[str] = typer.Option(
        None, "--out", "-o", help="Custom output directory for Obsidian vault."
    ),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
):
    """Export the existing knowledge graph into Obsidian vault, GraphML, and Cytoscape."""
    cfg = get_settings(config_path)
    store = ConceptGraphStore(cfg.graph)
    if not Path(cfg.graph.storage_path).is_file():
        console.print(f"[bold red]Graph file not found at {cfg.graph.storage_path}[/bold red]")
        raise typer.Exit(1)

    store.load()

    # Obsidian Export
    if cfg.graph.exports.obsidian.enabled:
        vault_dest = Path(output_dir) if output_dir else cfg.graph.exports.obsidian.vault_path
        obs = ObsidianExporter(vault_dest, create_moc=cfg.graph.exports.obsidian.create_moc)
        out = obs.export(store)
        console.print(f"[bold green]✓[/bold green] Obsidian Vault exported to: {out}")

    # GraphML Export
    if cfg.graph.exports.graphml.enabled:
        gml = GraphMLExporter(cfg.graph.exports.graphml.output_path)
        out = gml.export(store)
        console.print(f"[bold green]✓[/bold green] GraphML exported to: {out}")

    # Cytoscape Export
    if cfg.graph.exports.cytoscape.enabled:
        cyto = CytoscapeExporter(cfg.graph.exports.cytoscape.output_path)
        out = cyto.export(store)
        console.print(f"[bold green]✓[/bold green] Cytoscape JSON exported to: {out}")


if __name__ == "__main__":
    app()
