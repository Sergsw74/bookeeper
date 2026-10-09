"""
Unit tests for RTF parsing and archive handling (ZIP, TAR, etc.).
"""

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser, Section


def test_rtf_extraction_english(tmp_path):
    """Verify parsing of standard English RTF file into Sections."""
    rtf_file = tmp_path / "sample.rtf"
    content = rb"{\rtf1\ansi\deff0 {\fonttbl{\f0\fnil Arial;}}\viewkind4\uc1\pard\lang1033\f0\fs20 Chapter 1\par\par This is the first chapter of the book.\par\par Chapter 2\par\par This is the second chapter with interesting concepts.\par}"
    rtf_file.write_bytes(content)

    sections = BookParser.parse(rtf_file)
    assert len(sections) >= 2
    assert "Chapter 1" in sections[0].title or "Chapter 1" in sections[0].text
    assert "Chapter 2" in sections[1].title or "Chapter 2" in sections[1].text


def test_rtf_extraction_cyrillic_cp1251(tmp_path):
    """Verify parsing of Russian/Cyrillic CP1251 RTF file (e.g. Метро 2033)."""
    rtf_file = tmp_path / "metro2033.rtf"
    # RTF declaring \ansicpg1251 with hex-escaped Cyrillic: Метро 2033. Глава 1.
    content = rb"{\rtf1\ansi\ansicpg1251\deff0 {\fonttbl{\f0\fmodern Arial;}}\viewkind4\uc1\pard\f0\fs20 \'cc\'e5\'f2\'f0\'ee 2033\par\par \'c3\'eb\'e0\'e2\'e0 1\par\par \'d2\'e5\'ea\'f1\'f2 \'ef\'e5\'f0\'e2\'ee\'e9 \'e3\'eb\'e0\'e2\'fb \'ef\'f0\'ee \'f1\'f2\'e0\'ed\'f6\'e8\'fe \'c2\'c4\'cd\'d5.\par}"
    rtf_file.write_bytes(content)

    sections = BookParser.parse(rtf_file)
    assert len(sections) >= 1
    combined_text = " ".join(s.text for s in sections)
    assert "Метро 2033" in combined_text
    assert "Глава 1" in combined_text or any("Глава 1" in s.title for s in sections)
    assert "ВДНХ" in combined_text


def test_rtf_fallback_stripper():
    """Verify fallback regex RTF stripper when striprtf is unavailable."""
    raw_rtf = rb"{\rtf1\ansi\ansicpg1251\deff0 {\fonttbl{\f0\fmodern Arial;}}\viewkind4\uc1\pard\f0\fs20 \'cc\'e5\'f2\'f0\'ee 2033\par \'c3\'eb\'e0\'e2\'e0 1\par}"
    text = BookParser._fallback_rtf_to_text(raw_rtf, encoding="cp1251")
    assert "Метро 2033" in text
    assert "Глава 1" in text


def test_zip_archive_containing_fb2(tmp_path):
    """Verify extraction and parsing of an FB2 file stored inside a ZIP archive."""
    fb2_xml = """<?xml version="1.0" encoding="utf-8"?>
    <FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
      <body>
        <section>
          <title><p>Глава 1: Введение в Ассемблер</p></title>
          <p>Архитектура x86 и базовые регистры процессора EAX, EBX, ECX, EDX.</p>
        </section>
        <section>
          <title><p>Глава 2: Прерывания DOS и BIOS</p></title>
          <p>Работа с прерыванием int 21h в реальном режиме DOS.</p>
        </section>
      </body>
    </FictionBook>
    """
    zip_path = tmp_path / "assembler.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("book.fb2", fb2_xml.encode("utf-8"))
        zf.writestr("__MACOSX/._book.fb2", b"junk")

    sections = BookParser.parse(zip_path)
    assert len(sections) == 2
    assert "Глава 1" in sections[0].title
    assert "Ассемблер" in sections[0].title or "Ассемблер" in sections[0].text
    assert "EAX" in sections[0].text
    assert "Глава 2" in sections[1].title
    assert "int 21h" in sections[1].text


def test_zip_archive_containing_txt(tmp_path):
    """Verify extraction and parsing of a TXT file stored inside a ZIP archive."""
    txt_content = "Chapter 1: Assembly Language\nBasic CPU registers and memory addressing.\n\nChapter 2: System Calls\nKernel operations and context switches."
    zip_path = tmp_path / "assembly.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Assembly.txt", txt_content.encode("utf-8"))

    sections = BookParser.parse(zip_path)
    assert len(sections) == 2
    assert "Chapter 1" in sections[0].title or "Chapter 1" in sections[0].text
    assert "Chapter 2" in sections[1].title or "Chapter 2" in sections[1].text


def test_zip_archive_multiple_txt_parts(tmp_path):
    """Verify merging of multiple chapter text files inside an archive."""
    zip_path = tmp_path / "multi_part.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("part01.txt", "Part 1: The Beginning\nInitial setup and background details.")
        zf.writestr("part02.txt", "Part 2: The Journey\nTraveling through the wilderness.")

    sections = BookParser.parse(zip_path)
    assert len(sections) == 2
    assert sections[0].chapter_idx == 1
    assert "Part 1" in sections[0].text or "Part 1" in sections[0].title
    assert sections[1].chapter_idx == 2
    assert "Part 2" in sections[1].text or "Part 2" in sections[1].title


def test_tar_archive_containing_txt(tmp_path):
    """Verify extraction and parsing of a TXT file inside a TAR archive."""
    tar_path = tmp_path / "archive.tar"
    txt_bytes = b"Chapter 1\nContent inside tar archive."
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="book.txt")
        info.size = len(txt_bytes)
        tf.addfile(info, io.BytesIO(txt_bytes))

    sections = BookParser.parse(tar_path)
    assert len(sections) >= 1
    assert "tar archive" in sections[0].text


def test_archive_rejection_for_non_ebook_files(tmp_path):
    """Verify that an archive with only unsupported files raises a clear error."""
    zip_path = tmp_path / "images_only.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("cover.jpg", b"\xff\xd8\xff")
        zf.writestr("page1.png", b"\x89PNG")

    with pytest.raises(ValueError, match="only contains images"):
        BookParser.parse(zip_path)


def test_calibre_client_export_priority_includes_zip_and_rtf(tmp_path):
    """Verify CalibreClient export_book prioritizes and exports ZIP and RTF files."""
    import sqlite3

    db_file = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
    """)
    conn.execute("INSERT INTO books (id, title, path) VALUES (2, 'Assembly Book', 'Author/Assembly')")
    conn.execute("INSERT INTO data (id, book, format, name) VALUES (1, 2, 'ZIP', 'Assembly Book')")
    conn.execute("INSERT INTO books (id, title, path) VALUES (3, 'Metro 2033', 'Author/Metro')")
    conn.execute("INSERT INTO data (id, book, format, name) VALUES (2, 3, 'RTF', 'Metro 2033')")
    conn.commit()
    conn.close()

    (tmp_path / "Author" / "Assembly").mkdir(parents=True)
    zip_file = tmp_path / "Author" / "Assembly" / "Assembly Book.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("book.txt", "Chapter 1\nAssembly content.")

    (tmp_path / "Author" / "Metro").mkdir(parents=True)
    rtf_file = tmp_path / "Author" / "Metro" / "Metro 2033.rtf"
    rtf_file.write_bytes(rb"{\rtf1\ansi Metro 2033\par Chapter 1\par Text}")

    client = CalibreClient(library_path=tmp_path)
    out_dir = tmp_path / "exported"

    # Export book #2 (ZIP)
    exported_zip = client.export_book(2, target_dir=out_dir)
    assert exported_zip is not None
    assert exported_zip.suffix.lower() == ".zip"
    sections_zip = BookParser.parse(exported_zip)
    assert len(sections_zip) >= 1

    # Export book #3 (RTF)
    exported_rtf = client.export_book(3, target_dir=out_dir)
    assert exported_rtf is not None
    assert exported_rtf.suffix.lower() == ".rtf"
    sections_rtf = BookParser.parse(exported_rtf)
    assert len(sections_rtf) >= 1
