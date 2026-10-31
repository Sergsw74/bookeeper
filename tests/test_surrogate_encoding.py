import pytest
from pathlib import Path
from bookeeper.calibre.parser import BookParser
from bookeeper.processing.chunker import HierarchicalChunker


def test_sanitize_surrogates_recombines_valid_pairs():
    # U+1D400 (Mathematical Bold Capital A) encoded as surrogate pair U+D835 U+DC00
    surrogate_pair = "\ud835\udc00"
    sanitized = BookParser.sanitize_surrogates(surrogate_pair)
    assert sanitized == "𝐀"
    assert ord(sanitized) == 0x1D400
    # Must encode cleanly to UTF-8
    utf8_bytes = sanitized.encode("utf-8")
    assert len(utf8_bytes) == 4


def test_sanitize_surrogates_replaces_isolated_surrogates():
    # Isolated high surrogate without matching low surrogate
    isolated = "formula: \ud835\ud835 = x"
    sanitized = BookParser.sanitize_surrogates(isolated)
    assert "\ud835" not in sanitized
    # Must encode cleanly to UTF-8 without raising UnicodeEncodeError
    encoded = sanitized.encode("utf-8")
    assert b"formula:" in encoded


def test_repair_mojibake_integrates_surrogate_sanitization():
    raw_text = "BM25 topic score (\ud835\ud835 = \ud835(\ud835))"
    repaired = BookParser.repair_mojibake(raw_text)
    assert "\ud835" not in repaired
    # Verify sha256 hashing and json dumping
    import hashlib
    import json
    h = hashlib.sha256(repaired.encode("utf-8")).hexdigest()
    assert len(h) == 64
    dumped = json.dumps({"text": repaired})
    assert isinstance(dumped, str)


def test_precomem_pdf_chunking_with_surrogates():
    pdf_path = Path("archive_downloader/downloads/2610.29467 - PreCoMem_ Predictive Cognitive Memory for Self-Evolving Long-Term Dialogue Agents.pdf")
    if not pdf_path.is_file():
        pytest.skip("PreCoMem PDF not present in environment")

    sections = BookParser.parse(pdf_path)
    assert len(sections) > 0

    chunker = HierarchicalChunker()
    chunks = chunker.chunk_book(sections, book_id=217, book_title="PreCoMem")
    assert len(chunks) > 0

    # Ensure all chunks can be encoded to UTF-8 without surrogate error
    for c in chunks:
        c.text.encode("utf-8")
        assert "\ud835" not in c.text
