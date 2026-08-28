from io import BytesIO

from pypdf import PdfWriter

from app.ingest.extract import ExtractError, extract_text, is_supported


def test_markdown_is_supported():
    assert is_supported("notes.md", "text/markdown")
    assert extract_text("notes.md", b"# Hello\n\nWorld", "text/markdown") == "# Hello\n\nWorld"


def test_pdf_extension_is_supported():
    assert is_supported("brief.pdf", "application/pdf")


def test_blank_pdf_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    raw = BytesIO()
    writer.write(raw)
    try:
        extract_text("blank.pdf", raw.getvalue(), "application/pdf")
    except ExtractError as exc:
        assert "no extractable text" in str(exc)
    else:
        raise AssertionError("blank PDF should be rejected")
