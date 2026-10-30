"""ChromaDB wrapper. One collection per embedding model, so vectors from different models never mix."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import Embedder
from .models import Chunk, RetrievedChunk

BATCH = 64


def collection_name_for(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")[:40]
    return f"chunks_{slug}_{hashlib.sha1(model.encode()).hexdigest()[:8]}"


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._embedder = embedder
        self._col = self._get_collection(embedder.name)

    def _get_collection(self, model: str):
        return self._client.get_or_create_collection(
            collection_name_for(model), metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            self._col.upsert(  # upsert + deterministic IDs => no duplicate chunks
                ids=[c.chunk_id for c in batch],
                embeddings=self._embedder.embed_documents([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[
                    {"doc_id": c.doc_id, "filename": c.filename, "page": c.page, "chunk_index": c.index}
                    for c in batch
                ],
            )

    def query(self, text: str, k: int, doc_ids: Sequence[str] | None = None) -> list[RetrievedChunk]:
        if doc_ids is not None and not doc_ids:
            return []
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[self._embedder.embed_query(text)],
            n_results=min(k, total),
            where={"doc_id": {"$in": list(doc_ids)}} if doc_ids else None,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page=int(meta["page"]),
                    text=doc,
                    similarity=1.0 - float(dist),
                )
            )
        return out

    def count_for(self, doc_id: str) -> int:
        return len(self._col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def delete_document(self, doc_id: str, model: str | None = None) -> None:
        col = self._col if model in (None, self._embedder.name) else self._get_collection(model)
        col.delete(where={"doc_id": doc_id})

    def drop_all(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._get_collection(self._embedder.name)
