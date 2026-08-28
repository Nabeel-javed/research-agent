"""Extract indexable text from supported source files.

Text and Markdown inputs must contain valid UTF-8. PDF extraction supports
digital documents with a text layer; scanned documents require a separate OCR
stage and are rejected explicitly.
"""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractError(ValueError):
    pass


def extract_text(filename: str, raw: bytes, content_type: str | None) -> str:
    name = (filename or "upload").lower()
    if name.endswith(".pdf") or (content_type or "") == "application/pdf":
        return _from_pdf(filename, raw)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractError(f"{filename} is not valid UTF-8 text.") from exc


def is_supported(filename: str, content_type: str | None) -> bool:
    name = (filename or "").lower()
    if name.endswith((".txt", ".md", ".markdown", ".csv", ".json", ".log", ".rst", ".pdf")):
        return True
    kind = content_type or ""
    return kind.startswith("text/") or kind == "application/pdf"


def _from_pdf(filename: str, raw: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(raw))
    except PdfReadError as exc:
        raise ExtractError(f"{filename} is not a readable PDF.") from exc

    if getattr(reader, "is_encrypted", False):
        raise ExtractError(f"{filename} is an encrypted PDF.")

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        piece = page.extract_text() or ""
        piece = piece.strip()
        if piece:
            pages.append(f"[PDF page {index}]\n{piece}")

    text = "\n\n".join(pages).strip()
    if not text:
        raise ExtractError(
            f"{filename} has no extractable text. It is probably a scan; "
            "Only digital PDFs with a text layer are supported."
        )
    return text
