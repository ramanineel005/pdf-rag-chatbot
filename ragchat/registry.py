"""Small JSON registry of indexed documents (atomic writes)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DocRecord:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    embedding_model: str
    added_at: float
    ocr: bool = False


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> list[DocRecord]:
        return sorted((DocRecord(**v) for v in self._read().values()), key=lambda r: r.added_at)

    def get(self, doc_id: str) -> DocRecord | None:
        v = self._read().get(doc_id)
        return DocRecord(**v) if v else None

    def add(self, rec: DocRecord) -> None:
        data = self._read()
        data[rec.doc_id] = asdict(rec)
        self._write(data)

    def remove(self, doc_id: str) -> None:
        data = self._read()
        if data.pop(doc_id, None) is not None:
            self._write(data)

    @staticmethod
    def new_record(**kw) -> DocRecord:
        return DocRecord(added_at=time.time(), **kw)
