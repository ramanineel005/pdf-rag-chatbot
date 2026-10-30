"""PDF validation, text extraction with page metadata, cleaning, optional OCR."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import unicodedata
from typing import Callable

import pymupdf

from .models import ExtractedDocument, PageText, PDFError

MIN_TOTAL_CHARS = 50


def compute_doc_id(data: bytes) -> str:
    """Content hash: identical files get identical IDs (duplicate detection)."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # soft-wrap -> space, keep paragraphs
    return text.strip()


def ocr_available(tesseract_cmd: str = "") -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return bool(tesseract_cmd and os.path.exists(tesseract_cmd)) or shutil.which("tesseract") is not None


def _ocr_page(page: pymupdf.Page, tesseract_cmd: str, lang: str) -> str:
    import pytesseract
    from PIL import Image

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=lang)


def extract_pdf(
    data: bytes,
    filename: str,
    *,
    ocr: bool = False,
    tesseract_cmd: str = "",
    ocr_language: str = "eng",
    progress: Callable[[str], None] | None = None,
) -> ExtractedDocument:
    filename = os.path.basename(filename) or "document.pdf"
    if not data:
        raise PDFError(f"“{filename}” is empty (0 bytes).")
    if b"%PDF-" not in data[:1024]:
        raise PDFError(f"“{filename}” does not look like a PDF file.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFError(f"“{filename}” could not be opened; the file may be corrupt.") from exc

    pages: list[PageText] = []
    ocr_used = False
    with doc:
        if doc.needs_pass:
            raise PDFError(
                f"“{filename}” is password-protected. Remove the password "
                "(open it in a PDF viewer and save an unprotected copy), then upload it again."
            )
        if doc.page_count == 0:
            raise PDFError(f"“{filename}” has no pages.")
        if ocr and not ocr_available(tesseract_cmd):
            raise PDFError("OCR was requested but Tesseract is not installed (see README, “Optional OCR”).")
        try:
            for i in range(doc.page_count):
                page = doc[i]
                text = clean_text(page.get_text("text"))
                used_ocr = False
                if not text and ocr:
                    if progress:
                        progress(f"Running OCR on page {i + 1}/{doc.page_count}…")
                    text = clean_text(_ocr_page(page, tesseract_cmd, ocr_language))
                    used_ocr = bool(text)
                    ocr_used = ocr_used or used_ocr
                if text:
                    pages.append(PageText(page_number=i + 1, text=text, ocr=used_ocr))
        except PDFError:
            raise
        except Exception as exc:
            raise PDFError(f"“{filename}” could not be read; the file may be damaged.") from exc
        page_count = doc.page_count

    if sum(len(p.text) for p in pages) < MIN_TOTAL_CHARS:
        if ocr:
            raise PDFError(f"No readable text was found in “{filename}”, even with OCR.")
        raise PDFError(
            f"“{filename}” contains little or no extractable text — it is probably a scanned document. "
            "Enable OCR in the sidebar (requires Tesseract, see README) and upload it again."
        )
    return ExtractedDocument(doc_id=compute_doc_id(data), filename=filename, page_count=page_count, pages=pages)
