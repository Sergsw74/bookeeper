"""
Typer CLI interface for Archive Downloader.
"""

from pathlib import Path
from typing import Optional

import typer
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
)
from rich.table import Table

from archive_downloader.downloader import DEFAULT_URL, ArxivDownloader
from archive_downloader.indexer import DownloadIndex
from archive_downloader.models import ArticleMetadata
from archive_downloader.parser import normalize_arxiv_url, parse_arxiv_list_page

app = typer.Typer(
    name="archive-downloader",
    help="Automated arXiv article downloader with pagination, deduplication, and Calibre metadata.",
    add_completion=False,
)
console = Console()


@app.command("download")
def download_command(
    url: str = typer.Option(
        DEFAULT_URL,
        "--url",
        "-u",
        help="Source URL to scrape and download articles from.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-n",
        "--count",
        help="Target number of articles to download / process.",
    ),
    output_dir: Path = typer.Option(
        Path("./downloads"),
        "--output-dir",
        "-o",
        help="Subdirectory to store downloaded books and articles.",
    ),
    count_existing: bool = typer.Option(
        True,
        "--count-existing/--new-only",
        help="Count already downloaded articles towards the limit target (default: True). Use --new-only to download N new articles.",
    ),
    page_size: int = typer.Option(
        500,
        "--page-size",
        "--show",
        help="Number of entries to fetch per page on arXiv (e.g. 50, 100, 250, 500, 1000, 2000).",
    ),
    delay: float = typer.Option(
        0.5,
        "--delay",
        help="Polite delay between requests in seconds to respect arXiv rate limits.",
    ),
    no_opf: bool = typer.Option(
        False,
        "--no-opf",
        help="Skip generating Calibre .opf metadata XML files alongside PDFs.",
    ),
):
    """
    Download articles from arXiv up to the specified limit with automatic pagination and deduplication.
    """
    dest_dir = output_dir.expanduser().resolve()
    effective_url = normalize_arxiv_url(url, default_show=page_size)
    console.print(
        Panel.fit(
            f"[bold]Source URL:[/bold] [cyan]{effective_url}[/cyan]\n"
            f"[bold]Destination:[/bold] [green]{dest_dir}[/green]\n"
            f"[bold]Target Limit:[/bold] [yellow]{limit}[/yellow] articles\n"
            f"[bold]Page Size (show):[/bold] [cyan]{page_size}[/cyan] entries/page\n"
            f"[bold]Accounting Mode:[/bold] [dim]{'Count existing & new' if count_existing else 'New downloads only'}[/dim]\n"
            f"[bold]Calibre Metadata (.opf):[/bold] [dim]{'Disabled' if no_opf else 'Enabled'}[/dim]",
            title="Archive Downloader Starting",
            border_style="cyan",
        )
    )

    downloader = ArxivDownloader(
        base_url=effective_url,
        output_dir=dest_dir,
        limit=limit,
        page_size=page_size,
        count_existing=count_existing,
        request_delay=delay,
        write_opf=not no_opf,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles...", total=limit)

        def _on_progress(
            action: str,
            current_satisfied: int,
            target_limit: int,
            article: ArticleMetadata,
            is_new: bool,
        ):
            short_title = (article.title[:30] + "...") if len(article.title) > 33 else article.title
            if is_new:
                desc = f"[bold green]✓ Downloaded:[/bold green] [cyan]{article.arxiv_id}[/cyan] ({short_title})"
            else:
                desc = f"[dim yellow]↺ Skipped (existing):[/dim yellow] [cyan]{article.arxiv_id}[/cyan]"

            progress.update(task, completed=current_satisfied, total=target_limit, description=desc)

        stats = downloader.run(progress_callback=_on_progress)

    # Summary table
    table = Table(title="Download Run Summary", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Target Limit", str(stats.target_limit))
    table.add_row("Total Articles Inspected", str(stats.total_inspected))
    table.add_row("New Articles Downloaded", f"[bold green]{stats.new_downloaded}[/bold green]")
    table.add_row("Existing Articles Skipped", f"[yellow]{stats.already_existing}[/yellow]")
    table.add_row("Failed Downloads", f"[red]{stats.failed}[/red]" if stats.failed > 0 else "0")
    table.add_row("Pages Crawled", str(stats.pages_visited))
    table.add_row("Total Data Downloaded", f"{stats.total_bytes / (1024 * 1024):.2f} MB")
    table.add_row("Duration", f"{stats.duration_seconds}s")
    table.add_row("Storage Directory", str(dest_dir))
    table.add_row("Total Indexed in Storage", str(downloader.indexer.count()))

    console.print("\n", table)
    console.print(f"[bold green]✓ Download session completed successfully![/bold green]\n")


@app.command("status")
def status_command(
    output_dir: Path = typer.Option(
        Path("./downloads"),
        "--output-dir",
        "-o",
        help="Subdirectory where downloads and index are stored.",
    ),
    limit: int = typer.Option(
        15,
        "--limit",
        "-n",
        help="Number of recent downloads to display in table.",
    ),
):
    """
    Display statistics, disk usage, and recent items from the download index.
    """
    dest_dir = output_dir.expanduser().resolve()
    index = DownloadIndex(dest_dir)
    entries = index.all_entries()

    if not entries:
        console.print(f"[yellow]No indexed downloads found in {dest_dir}.[/yellow]")
        return

    total_bytes = sum(e.file_size for e in entries)
    categories = {}
    for e in entries:
        cat = e.primary_category or "Uncategorized"
        categories[cat] = categories.get(cat, 0) + 1

    console.print(
        Panel.fit(
            f"[bold]Storage Directory:[/bold] [green]{dest_dir}[/green]\n"
            f"[bold]Total Articles Indexed:[/bold] [cyan]{len(entries)}[/cyan]\n"
            f"[bold]Total Disk Usage:[/bold] [yellow]{total_bytes / (1024 * 1024):.2f} MB[/yellow]\n"
            f"[bold]Categories:[/bold] {', '.join(f'{k} ({v})' for k, v in sorted(categories.items()))}",
            title="Download Index Status",
            border_style="cyan",
        )
    )

    table = Table(title=f"Recent Downloads (Top {min(limit, len(entries))})", border_style="blue")
    table.add_column("arXiv ID", style="bold cyan")
    table.add_column("Title", style="white")
    table.add_column("Authors", style="dim")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Downloaded At", style="dim")

    for e in entries[:limit]:
        authors_str = ", ".join(e.authors[:2]) + (" et al." if len(e.authors) > 2 else "")
        short_title = (e.title[:45] + "...") if len(e.title) > 48 else e.title
        table.add_row(
            e.arxiv_id,
            short_title,
            authors_str,
            f"{e.file_size / 1024:.1f} KB",
            e.downloaded_at[:19].replace("T", " "),
        )

    console.print(table)


@app.command("inspect")
def inspect_command(
    url: str = typer.Option(
        DEFAULT_URL,
        "--url",
        "-u",
        help="Source URL to inspect.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of articles to preview.",
    ),
    page_size: int = typer.Option(
        500,
        "--page-size",
        "--show",
        help="Number of entries to fetch per page on arXiv (e.g. 50, 100, 250, 500).",
    ),
):
    """
    Inspect an arXiv list page and preview article metadata without downloading.
    """
    import httpx

    effective_url = normalize_arxiv_url(url, default_show=page_size)
    console.print(f"[bold cyan]Fetching page for inspection:[/bold cyan] {effective_url}")
    try:
        resp = httpx.get(
            effective_url,
            headers={"User-Agent": "ArchiveDownloader/1.0"},
            follow_redirects=True,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        console.print(f"[bold red]Failed to fetch URL:[/bold red] {e}")
        raise typer.Exit(1)

    articles, next_url = parse_arxiv_list_page(resp.text, current_url=url)
    table = Table(title=f"Articles Found ({len(articles)} on page)", border_style="cyan")
    table.add_column("arXiv ID", style="bold cyan")
    table.add_column("Title", style="white")
    table.add_column("Authors", style="dim")
    table.add_column("Category", style="yellow")
    table.add_column("Formats", style="green")

    for a in articles[:limit]:
        authors_str = ", ".join(a.authors[:2]) + (" et al." if len(a.authors) > 2 else "")
        short_title = (a.title[:45] + "...") if len(a.title) > 48 else a.title
        table.add_row(
            a.arxiv_id,
            short_title,
            authors_str,
            a.primary_category,
            ", ".join(a.available_formats),
        )

    console.print(table)
    if next_url:
        console.print(f"[dim]Next pagination page:[/dim] [cyan]{next_url}[/cyan]")


if __name__ == "__main__":
    app()
