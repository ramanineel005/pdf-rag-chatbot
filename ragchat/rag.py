"""Pipeline orchestration: ingest, retrieve, answer, delete."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .chunking import chunk_document
from .citations import NO_INFO_TEXT, build_messages, finalize_answer
from .config import Settings
from .embeddings import Embedder
from .llm import LLM
from .models import Answer, PDFError, RetrievedChunk
from .pdf_processing import compute_doc_id, extract_pdf
from .registry import DocRecord, DocumentRegistry
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    status: str  # "indexed" | "duplicate"
    record: DocRecord


@dataclass
class Prepared:
    chunks: list[RetrievedChunk]
    messages: list[dict] | None  # None => nothing relevant, do not call the LLM


class RAGEngine:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, llm: LLM, registry: DocumentRegistry):
        self.s, self.embedder, self.store, self.llm, self.registry = settings, embedder, store, llm, registry
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # ---------- documents ----------
    def _pdf_path(self, doc_id: str):
        return self.s.uploads_dir / f"{doc_id}.pdf"

    def list_documents(self) -> list[DocRecord]:
        return self.registry.all()

    def stale_documents(self) -> list[DocRecord]:
        return [r for r in self.registry.all() if r.embedding_model != self.embedder.name]

    def ingest(self, filename: str, data: bytes, *, ocr: bool = False, progress: Callable[[str], None] | None = None) -> IngestResult:
        say = progress or (lambda _m: None)
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise PDFError(f"“{filename}” is larger than the {self.s.max_upload_mb} MB limit.")
        existing = self.registry.get(compute_doc_id(data))
        if existing and existing.embedding_model == self.embedder.name:
            return IngestResult("duplicate", existing)
        say("Extracting text…")
        doc = extract_pdf(data, filename, ocr=ocr, tesseract_cmd=self.s.tesseract_cmd, ocr_language=self.s.ocr_language, progress=say)
        chunks = chunk_document(doc, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            raise PDFError(f"No indexable text found in “{filename}”.")
        say(f"Embedding {len(chunks)} passages…")
        try:
            self.store.add_chunks(chunks)
        except Exception:
            self.store.delete_document(doc.doc_id)  # no half-indexed documents
            raise
        if existing:  # old-model vectors are removed only after the new index succeeded
            self.store.delete_document(doc.doc_id, existing.embedding_model)
        self._pdf_path(doc.doc_id).write_bytes(data)
        rec = DocumentRegistry.new_record(
            doc_id=doc.doc_id, filename=doc.filename, pages=doc.page_count, chunks=len(chunks),
            embedding_model=self.embedder.name, ocr=any(p.ocr for p in doc.pages),
        )
        self.registry.add(rec)
        return IngestResult("indexed", rec)

    def delete_document(self, doc_id: str) -> None:
        rec = self.registry.get(doc_id)
        self.store.delete_document(doc_id, rec.embedding_model if rec else None)
        self.registry.remove(doc_id)
        self._pdf_path(doc_id).unlink(missing_ok=True)

    def reindex_stale(self, progress: Callable[[str], None] | None = None) -> list[str]:
        """Re-embed documents indexed under a different embedding model. Returns failed filenames."""
        failed = []
        for rec in self.stale_documents():
            path = self._pdf_path(rec.doc_id)
            try:
                data = path.read_bytes()
                self.ingest(rec.filename, data, ocr=rec.ocr, progress=progress)
            except Exception:
                log.exception("re-index failed for %s", rec.filename)
                failed.append(rec.filename)
        return failed

    def reset_all(self) -> None:
        for rec in self.registry.all():
            self.delete_document(rec.doc_id)
        self.store.drop_all()

    # ---------- question answering ----------
    def _search(self, query: str, scope: Sequence[str], active: set[str]) -> list[RetrievedChunk]:
        hits = self.store.query(query, self.s.top_k, scope)
        return [h for h in hits if h.similarity >= self.s.min_similarity and h.doc_id in active]

    def prepare(self, question: str, history: Sequence[dict] = (), doc_ids: Sequence[str] | None = None) -> Prepared:
        active = {r.doc_id for r in self.registry.all() if r.embedding_model == self.embedder.name}
        scope = [d for d in doc_ids if d in active] if doc_ids else sorted(active)
        if not scope:
            return Prepared([], None)
        chunks = self._search(question, scope, active)
        prev = [m["content"] for m in history if m["role"] == "user"]
        if not chunks and prev and len(question.split()) <= 6:  # short follow-up ("and refunds?"): add prior question
            chunks = self._search(f"{prev[-1]} {question}", scope, active)
        if not chunks:
            return Prepared([], None)
        recent = list(history)[-2 * self.s.max_history_turns :]
        return Prepared(chunks, build_messages(question, chunks, recent))

    def no_info_answer(self) -> Answer:
        return Answer(text=NO_INFO_TEXT, sources=[], grounded=False)

    def finalize(self, raw: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        names = Counter(r.filename for r in self.registry.all())
        label = {c.doc_id: (f"{c.filename} [{c.doc_id[:6]}]" if names[c.filename] > 1 else c.filename) for c in chunks}
        return finalize_answer(raw, chunks, label)

    def ask(self, question: str, history: Sequence[dict] = (), doc_ids: Sequence[str] | None = None) -> Answer:
        prep = self.prepare(question, history, doc_ids)
        if prep.messages is None:
            return self.no_info_answer()
        return self.finalize("".join(self.llm.stream(prep.messages)), prep.chunks)
