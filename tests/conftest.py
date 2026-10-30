import hashlib
import re
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragchat.config import load_settings
from ragchat.rag import RAGEngine
from ragchat.registry import DocumentRegistry
from ragchat.vector_store import VectorStore

STOP = {"the", "a", "an", "of", "and", "to", "is", "in", "on", "for", "what", "how", "does", "do", "are", "it", "this", "that"}


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download needed)."""

    def __init__(self, name="fake-hash-embedder"):
        self.name = name

    def _vec(self, text, dim=512):
        v = [0.0] * dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w not in STOP:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply="The warranty lasts 24 months [S1]."):
        self.reply, self.calls = reply, []

    def stream(self, messages):
        self.calls.append(messages)
        yield from (self.reply[i : i + 5] for i in range(0, len(self.reply), 5))


def make_pdf(pages, encrypt_pw=None):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    if encrypt_pw:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=encrypt_pw, owner_pw=encrypt_pw)
    return doc.tobytes()


def make_image_only_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    return doc.tobytes()


@pytest.fixture
def settings(tmp_path):
    return load_settings(data_dir=tmp_path, min_similarity=0.2, chunk_size=300, chunk_overlap=50)


@pytest.fixture
def make_engine(settings):
    def _make(embedder=None, llm=None, s=None):
        s = s or settings
        embedder = embedder or FakeEmbedder()
        store = VectorStore(s.chroma_dir, embedder)
        return RAGEngine(s, embedder, store, llm or FakeLLM(), DocumentRegistry(s.registry_path))

    return _make


@pytest.fixture
def engine(make_engine):
    return make_engine()
