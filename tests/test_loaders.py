"""Tests for document loaders: TextLoader, MarkdownLoader, PDFLoader."""

from pathlib import Path

import pytest

from knowledge_agent.loaders.base import Document
from knowledge_agent.loaders.text_loader import TextLoader
from knowledge_agent.loaders.markdown_loader import MarkdownLoader
from knowledge_agent.loaders.pdf_loader import PDFLoader


# ---------------------------------------------------------------------------
# Helper: create a minimal valid PDF with text on each page
# ---------------------------------------------------------------------------

def _create_test_pdf(path: Path, texts: list[str]) -> None:
    """Create a minimal valid PDF with one page per *texts* entry.

    Builds a PDF-1.4 document manually without any third-party PDF
    generation library, so the test is self-contained.
    """
    import io

    buf = io.BytesIO()
    obj_offsets: dict[int, int] = {}
    obj_counter = 0

    def _next_obj(data: bytes) -> int:
        nonlocal obj_counter
        obj_counter += 1
        obj_offsets[obj_counter] = buf.tell()
        buf.write(data)
        return obj_counter

    # Header
    buf.write(b"%PDF-1.4\n")

    n = len(texts)

    # Obj 1: Catalog
    _next_obj(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Obj 2: Pages tree
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    _next_obj(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n} >>\nendobj\n".encode()
    )

    font_num = 3 + 2 * n

    for i, text in enumerate(texts):
        pnum = 3 + 2 * i
        cnum = pnum + 1

        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_data = (
            f"BT /F1 12 Tf 100 700 Td ({safe}) Tj ET\n".encode("latin-1")
        )

        _next_obj(
            f"{pnum} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Contents {cnum} 0 R"
            f" /Resources << /Font << /F1 {font_num} 0 R >> >> >>\n"
            f"endobj\n".encode()
        )
        _next_obj(
            f"{cnum} 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode()
            + stream_data
            + b"\nendstream\nendobj\n"
        )

    # Font resource
    font_obj = (
        f"{font_num} 0 obj\n"
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        "endobj\n"
    )
    _next_obj(font_obj.encode())

    # xref table
    xref_offset = buf.tell()
    xref = b"xref\n"
    xref += f"0 {obj_counter + 1}\n".encode()
    xref += b"0000000000 65535 f \n"
    for i in range(1, obj_counter + 1):
        xref += f"{obj_offsets[i]:010d} 00000 n \n".encode()
    buf.write(xref)

    # Trailer
    trailer = (
        f"trailer\n"
        f"<< /Size {obj_counter + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n"
        "%%EOF\n"
    )
    buf.write(trailer.encode())

    path.write_bytes(buf.getvalue())


# ===================================================================
# TextLoader
# ===================================================================

class TestTextLoader:
    """Tests for TextLoader."""

    def test_can_handle_supported_extensions(self):
        loader = TextLoader()
        for ext in (".txt", ".log", ".csv", ".json"):
            assert loader.can_handle(Path(f"test{ext}")), f"missed {ext}"
        assert not loader.can_handle(Path("test.md"))
        assert not loader.can_handle(Path("test.pdf"))
        assert not loader.can_handle(Path("test.html"))

    def test_can_handle_case_insensitive(self):
        loader = TextLoader()
        assert loader.can_handle(Path("test.TXT"))
        assert loader.can_handle(Path("test.LOG"))

    def test_load_text_file(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("Hello, world!", encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["type"] == ".txt"
        assert docs[0].metadata["size"] > 0
        assert docs[0].source == str(file.resolve())

    def test_load_log_file(self, tmp_path):
        file = tmp_path / "app.log"
        file.write_text("INFO: test log entry\nWARN: something", encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert "INFO: test log entry" in docs[0].content

    def test_load_csv_file(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("a,b,c\n1,2,3\n4,5,6", encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert "a,b,c" in docs[0].content

    def test_load_json_file(self, tmp_path):
        file = tmp_path / "data.json"
        file.write_text('{"name": "test", "value": 42}', encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert docs[0].metadata["type"] == ".json"

    def test_load_with_latin1_fallback(self, tmp_path):
        file = tmp_path / "latin.txt"
        file.write_bytes(b"Latin-1 text: \xe9\xe0\xf9")  # é à ù
        loader = TextLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert docs[0].content == "Latin-1 text: \u00e9\u00e0\u00f9"

    def test_file_not_found(self):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.txt"))


# ===================================================================
# MarkdownLoader
# ===================================================================

class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_can_handle_md(self):
        loader = MarkdownLoader()
        assert loader.can_handle(Path("test.md"))
        assert not loader.can_handle(Path("test.txt"))
        assert not loader.can_handle(Path("test.html"))

    def test_file_not_found(self):
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/test.md"))

    def test_split_by_headings(self, tmp_path):
        content = """Introduction paragraph.

## Section One
Content of section one.

## Section Two
Content of section two.
"""
        file = tmp_path / "test.md"
        file.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(file)
        assert len(docs) == 3  # preamble + 2 sections
        assert docs[1].metadata["heading"] == "Section One"
        assert docs[2].metadata["heading"] == "Section Two"
        assert "Content of section one" in docs[1].content
        assert "Content of section two" in docs[2].content

    def test_no_headings_returns_single_doc(self, tmp_path):
        content = "Just a plain paragraph.\n\nNo headings here."
        file = tmp_path / "test.md"
        file.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert "heading" not in docs[0].metadata
        assert docs[0].content == content.strip()

    def test_single_heading(self, tmp_path):
        content = "## Only Section\nSome content."
        file = tmp_path / "test.md"
        file.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert docs[0].metadata["heading"] == "Only Section"
        assert "Some content." in docs[0].content

    def test_empty_file(self, tmp_path):
        file = tmp_path / "empty.md"
        file.write_text("", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(file)
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_metadata_preserved(self, tmp_path):
        file = tmp_path / "test.md"
        file.write_text("## A\nbody", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(file)
        assert docs[0].metadata["filename"] == "test.md"
        assert docs[0].metadata["type"] == ".md"
        assert docs[0].metadata["size"] > 0
        assert docs[0].source == str(file.resolve())


# ===================================================================
# PDFLoader
# ===================================================================

class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_can_handle_pdf(self):
        loader = PDFLoader()
        assert loader.can_handle(Path("test.pdf"))
        assert not loader.can_handle(Path("test.txt"))

    def test_file_not_found(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/test.pdf"))

    def test_each_page_is_a_document(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        _create_test_pdf(pdf_path, ["Page One Text", "Page Two Text", "Page Three Text"])

        loader = PDFLoader()
        docs = loader.load(pdf_path)

        assert len(docs) == 3
        assert "Page One Text" in docs[0].content
        assert "Page Two Text" in docs[1].content
        assert "Page Three Text" in docs[2].content

    def test_page_numbers_in_metadata(self, tmp_path):
        pdf_path = tmp_path / "pages.pdf"
        _create_test_pdf(pdf_path, ["First", "Second", "Third"])

        loader = PDFLoader()
        docs = loader.load(pdf_path)

        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2
        assert docs[2].metadata["page"] == 3

    def test_base_metadata(self, tmp_path):
        pdf_path = tmp_path / "meta.pdf"
        _create_test_pdf(pdf_path, ["Hello"])

        loader = PDFLoader()
        docs = loader.load(pdf_path)

        assert docs[0].metadata["filename"] == "meta.pdf"
        assert docs[0].metadata["type"] == ".pdf"
        assert docs[0].metadata["total_pages"] == 1
        assert docs[0].metadata["size"] > 0
        assert docs[0].source == str(pdf_path.resolve())

    def test_skip_empty_pages(self, tmp_path):
        pdf_path = tmp_path / "mixed.pdf"
        # Third text is empty string -> blank page -> should be skipped
        _create_test_pdf(pdf_path, ["Page 1", "Page 2", ""])

        loader = PDFLoader()
        docs = loader.load(pdf_path)

        assert len(docs) == 2
        assert "Page 1" in docs[0].content
        assert "Page 2" in docs[1].content
