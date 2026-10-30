"""Local embeddings via Sentence Transformers. Never downloads without an explicit call to download()."""
from __future__ import annotations

from typing import Protocol, Sequence

from .models import EmbeddingModelUnavailable


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, download_hint: str = ""):
        self.name = model_name
        self.download_hint = download_hint
        self._model = None

    def _load(self, local_only: bool):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.name, device="cpu", local_files_only=local_only)

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._model = self._load(local_only=True)
            return True
        except Exception:
            return False

    def download(self) -> None:
        """Explicit, user-approved download (first-time setup only)."""
        try:
            self._model = self._load(local_only=False)
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                f"Could not download the embedding model “{self.name}”. Check your internet connection and try again."
            ) from exc

    def _ensure(self):
        if self._model is None and not self.is_available():
            raise EmbeddingModelUnavailable(
                f"The embedding model “{self.name}” is not downloaded yet ({self.download_hint}). "
                "Use the download button in the sidebar."
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        return model.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
