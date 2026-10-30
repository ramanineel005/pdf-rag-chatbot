import pytest

from ragchat.chunking import chunk_document, split_text
from ragchat.models import PDFError
from ragchat.pdf_processing import clean_text, compute_doc_id, extract_pdf
from conftest import make_image_only_pdf, make_pdf


def test_extraction_keeps_one_based_page_numbers():
    data = make_pdf(["First page about apples and oranges in the orchard.", "", "Third page about warranty terms and conditions."])
    doc = extract_pdf(data, "a.pdf")
    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 3]  # blank page 2 skipped, numbering preserved
    assert "warranty" in doc.pages[1].text


def test_empty_file_rejected():
    with pytest.raises(PDFError, match="empty"):
        extract_pdf(b"", "e.pdf")


def test_non_pdf_rejected():
    with pytest.raises(PDFError, match="does not look like a PDF"):
        extract_pdf(b"hello world, definitely not a pdf", "x.pdf")


def test_corrupt_pdf_rejected():
    data = make_pdf(["some text that is long enough to matter here"])
    with pytest.raises(PDFError):
        extract_pdf(data[: len(data) // 3], "broken.pdf")


def test_password_protected_pdf_explained():
    with pytest.raises(PDFError, match="password-protected"):
        extract_pdf(make_pdf(["secret text secret text secret text secret text"], encrypt_pw="pw"), "locked.pdf")


def test_scanned_pdf_explains_ocr_needed():
    with pytest.raises(PDFError, match="OCR"):
        extract_pdf(make_image_only_pdf(), "scan.pdf")


def test_doc_id_is_content_based():
    assert compute_doc_id(b"abc") == compute_doc_id(b"abc") != compute_doc_id(b"abd")


def test_clean_text_joins_hyphenation_and_soft_wraps():
    assert clean_text("inter-\nnational trade\nagreement\n\n\n\nNext") == "international trade agreement\n\nNext"


def test_chunks_overlap_and_respect_size():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 3 and all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        tail_words = set(a.split()[-6:])
        assert tail_words & set(b.split()[:8]), "consecutive chunks should share overlapping text"
    assert " ".join(chunks).count("word299") >= 1


def test_chunks_never_cross_pages_and_keep_metadata():
    data = make_pdf(["Alpha " * 120, "Bravo " * 120])
    doc = extract_pdf(data, "m.pdf")
    chunks = chunk_document(doc, 300, 50)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.filename == "m.pdf" and c.doc_id == doc.doc_id and c.chunk_id for c in chunks)
    assert all(("Alpha" in c.text) == (c.page == 1) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("abc " * 100, 100, 100)
