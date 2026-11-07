import errno
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from bookeeper.cli import app, is_disk_full_error


def test_is_disk_full_error():
    """Verify is_disk_full_error correctly detects ENOSPC and Errno 28 in all forms."""
    # 1. Native OSError with errno.ENOSPC
    err_native = OSError(errno.ENOSPC, "No space left on device")
    assert is_disk_full_error(err_native) is True

    # 2. RuntimeError with Errno 28 in message
    err_msg = RuntimeError("[Errno 28] No space left on device")
    assert is_disk_full_error(err_msg) is True

    # 3. Exact error from user prompt
    err_user = Exception("Failed to build graph for book #5857 ('Cross-Domain Joint-Embedding Pretraining For Heterogeneous Industrial Time Series'): [Errno 28] No space left on device")
    assert is_disk_full_error(err_user) is True

    # 4. Chained exception context
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise RuntimeError("Extraction failed") from inner
    except RuntimeError as outer:
        assert is_disk_full_error(outer) is True

    # 5. Unrelated errors must return False
    assert is_disk_full_error(ValueError("Cannot parse json")) is False
    assert is_disk_full_error(OSError(errno.ENOENT, "No such file or directory")) is False
    assert is_disk_full_error(None) is False


def test_build_graph_hard_stop_on_enospc(tmp_path):
    """
    Verify that when build-graph encounters an ENOSPC ([Errno 28] No space left on device) error:
    1. It logs the hard stop error.
    2. It immediately halts without processing any remaining books.
    3. Exits with exit code 28.
    """
    runner = CliRunner(env={"COLUMNS": "200"})

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    books_dir = tmp_path / "books"
    books_dir.mkdir()

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "output_dir": str(out_dir),
        "books_dir": str(books_dir),
        "neo4j": {"enabled": False},
    }))

    # Create 3 book files
    b1 = books_dir / "5857_Cross_Domain.pdf"
    b1.write_bytes(b"%PDF-sample-1")
    b2 = books_dir / "5858_Next_Book.pdf"
    b2.write_bytes(b"%PDF-sample-2")
    b3 = books_dir / "5859_Third_Book.pdf"
    b3.write_bytes(b"%PDF-sample-3")

    mock_client = MagicMock()
    mock_client.list_books.return_value = [
        {"id": 5857, "title": "Cross-Domain Joint-Embedding Pretraining", "path": str(b1)},
        {"id": 5858, "title": "Next Book In Queue", "path": str(b2)},
        {"id": 5859, "title": "Third Book In Queue", "path": str(b3)},
    ]

    processed_book_paths = []

    def mock_parse(bpath):
        processed_book_paths.append(bpath)
        if "5857" in str(bpath):
            # Simulate disk full error on book #5857
            raise OSError(errno.ENOSPC, "No space left on device")
        return [MagicMock(title="Ch1", text="Sample text")]

    with patch("bookeeper.calibre.get_book_client", return_value=mock_client), \
         patch("bookeeper.calibre.parser.BookParser.parse", side_effect=mock_parse):

        res = runner.invoke(app, [
            "build-graph",
            "--config", str(cfg_file),
            "--no-dual-pass",
            "--no-export-neo4j",
        ])

        # Must exit with code 28 (ENOSPC)
        assert res.exit_code == 28

        # Hard stop alert must be printed
        assert "HARD STOP" in res.stdout
        assert "No space left on device" in res.stdout
        assert "5857" in res.stdout

        # Book #5858 and #5859 must NEVER have been processed!
        assert len(processed_book_paths) == 1
        assert "5857" in str(processed_book_paths[0])
        assert not any("5858" in str(p) for p in processed_book_paths)
        assert not any("5859" in str(p) for p in processed_book_paths)
