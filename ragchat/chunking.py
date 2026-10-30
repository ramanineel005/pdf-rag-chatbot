"""Page-aware chunking. Chunks never cross page boundaries, so page citations stay exact."""
from __future__ import annotations

from .models import Chunk, ExtractedDocument

_SEPARATORS = ("\n\n", ". ", "? ", "! ", "; ", ", ", " ")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk size must be > 0 and 0 <= overlap < size")
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_start = start + int(size * 0.6)
            for sep in _SEPARATORS:
                pos = text.rfind(sep, window_start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        if nxt > 0 and not text[nxt - 1].isspace():  # don't start mid-word
            space = text.find(" ", nxt, nxt + 40)
            if space != -1:
                nxt = space + 1
        start = min(nxt, end)
    return out


def chunk_document(doc: ExtractedDocument, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for page in doc.pages:
        for piece in split_text(page.text, size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-p{page.page_number}-c{n}",
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page=page.page_number,
                    index=n,
                    text=piece,
                )
            )
            n += 1
    return chunks
