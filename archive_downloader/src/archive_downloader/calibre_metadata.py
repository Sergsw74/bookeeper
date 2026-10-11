"""
Calibre OPF metadata generator for downloaded articles and papers.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from archive_downloader.models import ArticleMetadata


def generate_calibre_opf(article: ArticleMetadata, output_path: Path) -> Path:
    """
    Generate a Calibre-compatible Open Packaging Format (.opf) XML file
    containing metadata (title, authors, subjects, identifier) for the article.
    """
    package = ET.Element(
        "package",
        {
            "xmlns": "http://www.idpf.org/2007/opf",
            "unique-identifier": "arxiv_id",
            "version": "2.0",
        },
    )

    metadata = ET.SubElement(
        package,
        "metadata",
        {
            "xmlns:dc": "http://purl.org/dc/elements/1.1/",
            "xmlns:opf": "http://www.idpf.org/2007/opf",
        },
    )

    # Title
    dc_title = ET.SubElement(metadata, "dc:title")
    dc_title.text = article.title

    # Authors
    if article.authors:
        for idx, author in enumerate(article.authors):
            dc_creator = ET.SubElement(
                metadata,
                "dc:creator",
                {"opf:role": "aut", "opf:file-as": author},
            )
            dc_creator.text = author
    else:
        dc_creator = ET.SubElement(metadata, "dc:creator", {"opf:role": "aut"})
        dc_creator.text = "Unknown Author"

    # Identifier
    dc_id = ET.SubElement(
        metadata,
        "dc:identifier",
        {"id": "arxiv_id", "opf:scheme": "arxiv"},
    )
    dc_id.text = article.arxiv_id

    # Subjects / Tags
    all_subjects = []
    if article.primary_category:
        all_subjects.append(article.primary_category)
    for c in article.categories:
        if c not in all_subjects:
            all_subjects.append(c)

    for subj in all_subjects:
        dc_subj = ET.SubElement(metadata, "dc:subject")
        dc_subj.text = subj

    # Description
    dc_desc = ET.SubElement(metadata, "dc:description")
    dc_desc.text = f"arXiv article {article.arxiv_id}. Abstract: {article.abstract_url}"

    # Pretty print XML
    raw_xml = ET.tostring(package, encoding="utf-8")
    parsed = minidom.parseString(raw_xml)
    pretty_xml = parsed.toprettyxml(indent="  ", encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pretty_xml)
    return output_path
