"""
HTML parser for arXiv recent / list pages and search result pages, including pagination discovery.
"""

import re
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from archive_downloader.models import ArticleMetadata


def normalize_arxiv_url(url: str, default_show: int = 500, default_skip: int = 0) -> str:
    """
    Ensure arXiv list or search URL has appropriate pagination parameters.
    For /list/ URLs: ensures skip and show parameters.
    For /search/ URLs: ensures start, size, searchtype, order, and abstracts parameters.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "/search" in parsed.path:
        if "start" not in qs:
            qs["start"] = [str(default_skip)]
        if "size" not in qs:
            valid_sizes = [25, 50, 100, 200]
            chosen_size = default_show if default_show in valid_sizes else 100
            qs["size"] = [str(chosen_size)]
        if "abstracts" not in qs:
            qs["abstracts"] = ["show"]
        if "order" not in qs:
            qs["order"] = ["-announced_date_first"]
        if "searchtype" not in qs:
            qs["searchtype"] = ["all"]
        new_query = urlencode([(k, v[0]) for k, v in qs.items()])
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

    if "/list/" in parsed.path:
        if "skip" not in qs:
            qs["skip"] = [str(default_skip)]
        if "show" not in qs:
            valid_sizes = [25, 50, 100, 250, 500, 1000, 2000]
            chosen_show = default_show if default_show in valid_sizes else 500
            qs["show"] = [str(chosen_show)]
        new_query = urlencode([(k, v[0]) for k, v in qs.items()])
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

    return url


def build_arxiv_search_url(
    query: str,
    exact_match: bool = False,
    size: int = 100,
    order: str = "-announced_date_first",
    searchtype: str = "all",
) -> str:
    """
    Construct a canonical arXiv search query URL.
    If exact_match is True, wraps query in quotes for exact phrase search.
    """
    cleaned = query.strip()
    if exact_match and not (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = f'"{cleaned}"'

    valid_sizes = [25, 50, 100, 200]
    chosen_size = size if size in valid_sizes else 100

    params = [
        ("searchtype", searchtype),
        ("query", cleaned),
        ("abstracts", "show"),
        ("size", str(chosen_size)),
        ("order", order),
        ("start", "0"),
    ]
    return f"https://arxiv.org/search/?{urlencode(params)}"


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

        # Strip trailing version for canonical identity
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

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


def parse_arxiv_search_page(
    html: str,
    current_url: str = "https://arxiv.org/search/",
) -> Tuple[List[ArticleMetadata], Optional[str]]:
    """
    Parse an arXiv search results page (such as /search/?query=...).
    Extracts article metadata and the link to the next pagination page.
    """
    soup = BeautifulSoup(html, "html.parser")
    articles: List[ArticleMetadata] = []

    results = soup.find_all("li", class_="arxiv-result")
    for item in results:
        # 1. Extract arXiv ID and available formats
        title_p = item.find("p", class_="list-title")
        arxiv_id = ""
        formats = []
        pdf_url = ""
        abs_url = ""

        if title_p:
            abs_link = title_p.find("a", href=lambda h: h and "/abs/" in h)
            if abs_link:
                abs_url = urljoin(current_url, abs_link.get("href"))
                m = re.search(r"/abs/([a-zA-Z0-9.\-_/]+)", abs_link.get("href"))
                if m:
                    arxiv_id = m.group(1)
                else:
                    m2 = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", abs_link.get_text())
                    if m2:
                        arxiv_id = m2.group(1)

            for a in title_p.find_all("a"):
                txt = a.get_text(strip=True).lower()
                href = a.get("href", "")
                if "/pdf/" in href or txt == "pdf":
                    if "pdf" not in formats:
                        formats.append("pdf")
                    pdf_url = urljoin(current_url, href)
                elif "/format/" in href or txt == "other":
                    if "other" not in formats:
                        formats.append("other")
                elif "/ps/" in href or txt == "ps":
                    if "ps" not in formats:
                        formats.append("ps")
                elif "/html/" in href or txt == "html":
                    if "html" not in formats:
                        formats.append("html")

        if not arxiv_id:
            # Fallback regex across item text
            match = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", item.get_text())
            if match:
                arxiv_id = match.group(1)

        if not arxiv_id:
            continue

        # Strip trailing version for canonical identity
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        if not abs_url:
            abs_url = urljoin(current_url, f"/abs/{arxiv_id}")
        if not pdf_url:
            pdf_url = urljoin(current_url, f"/pdf/{arxiv_id}")
        if "pdf" not in formats and pdf_url:
            formats.append("pdf")

        # 2. Extract Title
        title = ""
        title_el = item.find("p", class_="title")
        if title_el:
            desc_span = title_el.find("span", class_="descriptor")
            if desc_span:
                desc_span.extract()
            raw_title = title_el.get_text()
            title = re.sub(r"\s+", " ", raw_title).strip()
            if title.lower().startswith("title:"):
                title = title[6:].strip()

        if not title:
            title = f"arXiv Paper {arxiv_id}"

        # 3. Extract Authors
        authors = []
        authors_el = item.find("p", class_="authors")
        if authors_el:
            for a in authors_el.find_all("a"):
                author_name = a.get_text(strip=True)
                if author_name and author_name not in authors:
                    authors.append(author_name)

        # 4. Extract Tags / Categories
        categories = []
        primary_cat = ""
        tags_div = item.find("div", class_="tags")
        if tags_div:
            for span in tags_div.find_all("span", class_="tag"):
                c = span.get_text(strip=True)
                if c and c not in categories:
                    categories.append(c)
                    if not primary_cat:
                        primary_cat = c

        if not primary_cat:
            primary_cat = categories[0] if categories else "General"

        # 5. Extract Abstract
        abstract = ""
        abs_p = item.find("p", class_="abstract")
        if abs_p:
            full_span = abs_p.find("span", class_=lambda c: c and "abstract-full" in c)
            short_span = abs_p.find("span", class_=lambda c: c and "abstract-short" in c)
            target = full_span or short_span or abs_p
            raw_abs = target.get_text()
            cleaned_abs = re.sub(r"\s+", " ", raw_abs).strip()
            cleaned_abs = re.sub(r"^[Aa]bstract:\s*", "", cleaned_abs)
            abstract = re.sub(r"(?:▽|△)\s*(?:More|Less)", "", cleaned_abs).strip()

        preferred_format = "pdf"
        download_url = pdf_url

        articles.append(
            ArticleMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                primary_category=primary_cat,
                categories=categories,
                abstract=abstract,
                abstract_url=abs_url,
                pdf_url=pdf_url,
                available_formats=formats,
                preferred_format=preferred_format,
                download_url=download_url,
            )
        )

    # 6. Extract Next Page URL
    next_page_url = _find_search_next_page(soup, current_url, items_on_page=len(articles))

    return articles, next_page_url


def parse_arxiv_page(
    html: str,
    current_url: str = "https://arxiv.org/",
) -> Tuple[List[ArticleMetadata], Optional[str]]:
    """
    Parse an arXiv page (auto-detecting list view or search results view).
    """
    soup = BeautifulSoup(html, "html.parser")
    if "/search" in current_url or soup.find("li", class_="arxiv-result") is not None:
        return parse_arxiv_search_page(html, current_url=current_url)
    return parse_arxiv_list_page(html, current_url=current_url)


def _find_search_next_page(
    soup: BeautifulSoup,
    current_url: str,
    items_on_page: int,
) -> Optional[str]:
    """
    Find the next pagination link on an arXiv search page.
    """
    next_a = soup.find("a", class_="pagination-next")
    if next_a:
        classes = next_a.get("class", [])
        href = next_a.get("href", "").strip()
        if href and "is-invisible" not in classes and "disabled" not in classes:
            return urljoin(current_url, href)

    # Fallback using query parameters start and size
    parsed = urlparse(current_url)
    qs = parse_qs(parsed.query)
    current_start = 0
    if "start" in qs:
        try:
            current_start = int(qs["start"][0])
        except Exception:
            pass
    page_size = 100
    if "size" in qs:
        try:
            page_size = int(qs["size"][0])
        except Exception:
            pass
    elif items_on_page > 0:
        page_size = items_on_page

    for heading in soup.find_all(["h1", "h2", "h3", "div", "p"]):
        m = re.search(r"Showing\s+\d+[–\-]\d+\s+of\s+([\d,]+)\s+results", heading.get_text(), re.IGNORECASE)
        if m:
            total_str = m.group(1).replace(",", "")
            try:
                total_entries = int(total_str)
                next_start = current_start + page_size
                if next_start < total_entries:
                    new_qs = dict(qs)
                    new_qs["start"] = [str(next_start)]
                    new_query = urlencode([(k, v[0]) for k, v in new_qs.items()])
                    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            except Exception:
                pass
            break

    return None


def _find_next_page(
    soup: BeautifulSoup,
    current_url: str,
    items_on_page: int,
) -> Optional[str]:
    """
    Find the next pagination link on a list page.
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
    paging_text = ""
    for container in paging_containers:
        paging_text += container.get_text()
    match = re.search(r"Total of\s+(\d+)\s+entries", paging_text, re.IGNORECASE)
    if match:
        total_entries = int(match.group(1))
        next_skip = current_skip + items_on_page
        if next_skip < total_entries:
            show_val = qs.get("show", ["50"])[0]
            base_path = urljoin(current_url, parsed.path)
            return f"{base_path}?skip={next_skip}&show={show_val}"

    return None
