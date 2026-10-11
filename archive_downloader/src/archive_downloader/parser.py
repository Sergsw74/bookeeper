"""
HTML parser for arXiv recent / list pages and pagination extraction.
"""

import re
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from archive_downloader.models import ArticleMetadata


def normalize_arxiv_url(url: str, default_show: int = 500, default_skip: int = 0) -> str:
    """
    Ensure arXiv list URL has appropriate skip and show parameters.
    If already present in URL, preserves user-specified values.
    Otherwise sets skip=default_skip and show=default_show.
    """
    parsed = urlparse(url)
    if "/list/" not in parsed.path:
        return url

    qs = parse_qs(parsed.query)
    if "skip" not in qs:
        qs["skip"] = [str(default_skip)]
    if "show" not in qs:
        valid_sizes = [25, 50, 100, 250, 500, 1000, 2000]
        chosen_show = default_show if default_show in valid_sizes else 500
        qs["show"] = [str(chosen_show)]

    new_query = urlencode([(k, v[0]) for k, v in qs.items()])
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def sanitize_filename(text: str, max_length: int = 120) -> str:
    """
    Sanitize text to create a safe, clean filesystem filename.
    """
    # Remove control characters and newlines
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    # Remove illegal filename characters: / \ : * ? " < > |
    cleaned = re.sub(r'[\\/*?:"<>|]+', "_", cleaned)
    # Collapse consecutive underscores or spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    # Truncate to max length without breaking words abruptly
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" ._")
    return cleaned or "untitled"


def parse_arxiv_list_page(
    html: str,
    current_url: str = "https://arxiv.org/list/cs.AI/recent",
) -> Tuple[List[ArticleMetadata], Optional[str]]:
    """
    Parse an arXiv list page (such as /list/cs.AI/recent).
    Extracts article metadata and the link to the next pagination page.
    """
    soup = BeautifulSoup(html, "html.parser")
    articles: List[ArticleMetadata] = []

    dt_list = soup.find_all("dt")
    dd_list = soup.find_all("dd")

    for dt, dd in zip(dt_list, dd_list):
        # 1. Extract arXiv ID
        arxiv_id = ""
        abs_link = dt.find("a", title="Abstract") or dt.find("a", href=lambda h: h and "/abs/" in h)
        if abs_link:
            match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", abs_link.text)
            if match:
                arxiv_id = match.group(1)
            else:
                arxiv_id = abs_link.text.strip().replace("arXiv:", "")

        if not arxiv_id:
            # Fallback regex across dt text
            match = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", dt.get_text())
            if match:
                arxiv_id = match.group(1)

        if not arxiv_id:
            continue

        abstract_url = urljoin(current_url, f"/abs/{arxiv_id}")
        pdf_url = urljoin(current_url, f"/pdf/{arxiv_id}")

        # 2. Extract available format links
        formats = []
        for a in dt.find_all("a"):
            txt = a.text.strip().lower()
            if txt in {"pdf", "html", "other"}:
                formats.append(txt)
            elif "/pdf/" in a.get("href", ""):
                if "pdf" not in formats:
                    formats.append("pdf")
                pdf_url = urljoin(current_url, a.get("href"))

        if not formats and pdf_url:
            formats.append("pdf")

        # 3. Extract title from dd
        title = ""
        title_div = dd.find("div", class_="list-title")
        if title_div:
            # Clone and remove descriptor span if present
            desc_span = title_div.find("span", class_="descriptor")
            if desc_span:
                desc_span.extract()
            title = title_div.get_text(separator=" ", strip=True)
            if title.lower().startswith("title:"):
                title = title[6:].strip()

        if not title:
            title = f"arXiv Paper {arxiv_id}"

        # 4. Extract authors
        authors = []
        authors_div = dd.find("div", class_="list-authors")
        if authors_div:
            for a in authors_div.find_all("a"):
                author_name = a.get_text(strip=True)
                if author_name and author_name not in authors:
                    authors.append(author_name)

        # 5. Extract subjects / categories
        categories = []
        primary_cat = ""
        subj_div = dd.find("div", class_="list-subjects")
        if subj_div:
            prim_span = subj_div.find("span", class_="primary-subject")
            if prim_span:
                primary_cat = prim_span.get_text(strip=True)
            # Find all category codes e.g. (cs.AI), (cs.LG)
            cat_matches = re.findall(r"\(([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)\)", subj_div.get_text())
            for cm in cat_matches:
                if cm not in categories:
                    categories.append(cm)

        # Fallback primary category from URL or categories
        if not primary_cat:
            if categories:
                primary_cat = categories[0]
            else:
                url_match = re.search(r"/list/([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)/", current_url)
                primary_cat = url_match.group(1) if url_match else "General"

        # Determine download URL: prefer PDF
        preferred_format = "pdf"
        download_url = pdf_url

        articles.append(
            ArticleMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                primary_category=primary_cat,
                categories=categories,
                abstract_url=abstract_url,
                pdf_url=pdf_url,
                available_formats=formats,
                preferred_format=preferred_format,
                download_url=download_url,
            )
        )

    # 6. Extract Next Page URL
    next_page_url = _find_next_page(soup, current_url, items_on_page=len(articles))

    return articles, next_page_url


def _find_next_page(
    soup: BeautifulSoup,
    current_url: str,
    items_on_page: int,
) -> Optional[str]:
    """
    Find the next pagination link on the page.
    """
    current_skip = 0
    parsed = urlparse(current_url)
    qs = parse_qs(parsed.query)
    if "skip" in qs:
        try:
            current_skip = int(qs["skip"][0])
        except Exception:
            pass

    # Look for paging containers: <div class="paging"> or <div class="pagination">
    paging_containers = soup.find_all(
        lambda t: t.name in {"div", "nav"}
        and any(c in t.get("class", []) for c in ["paging", "pagination"])
    )

    candidate_links: List[Tuple[int, str]] = []

    for container in paging_containers:
        for a in container.find_all("a"):
            href = a.get("href", "")
            if "skip=" in href:
                a_qs = parse_qs(urlparse(href).query)
                if "skip" in a_qs:
                    try:
                        s_val = int(a_qs["skip"][0])
                        candidate_links.append((s_val, urljoin(current_url, href)))
                    except Exception:
                        pass

    # Fallback to any links with skip= on the page
    if not candidate_links:
        for a in soup.find_all("a", href=lambda h: h and "skip=" in h):
            href = a.get("href", "")
            a_qs = parse_qs(urlparse(href).query)
            if "skip" in a_qs:
                try:
                    s_val = int(a_qs["skip"][0])
                    candidate_links.append((s_val, urljoin(current_url, href)))
                except Exception:
                    pass

    # Filter links that advance skip forward
    next_candidates = [
        link for s_val, link in candidate_links if s_val > current_skip
    ]

    if next_candidates:
        # Sort by skip ascending to get immediate next page
        next_candidates.sort(
            key=lambda l: int(parse_qs(urlparse(l).query).get("skip", [0])[0])
        )
        return next_candidates[0]

    # Algorithmic fallback if there are more items indicated
    total_entries_match = None
    paging_text = ""
    for container in paging_containers:
        paging_text += container.get_text()
    match = re.search(r"Total of\s+(\d+)\s+entries", paging_text, re.IGNORECASE)
    if match:
        total_entries = int(match.group(1))
        next_skip = current_skip + items_on_page
        if next_skip < total_entries:
            show_val = qs.get("show", ["50"])[0]
            # Construct next page url
            base_path = urljoin(current_url, parsed.path)
            return f"{base_path}?skip={next_skip}&show={show_val}"

    return None
