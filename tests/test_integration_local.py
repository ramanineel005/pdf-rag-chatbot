"""Run only when real local models exist. Skipped otherwise."""
import pytest

from ragchat.config import load_settings
from ragchat.embeddings import SentenceTransformerEmbedder
from ragchat.llm import OllamaLLM
from ragchat.rag import RAGEngine
from ragchat.registry import DocumentRegistry
from ragchat.vector_store import VectorStore
from conftest import make_pdf

S = load_settings()
try:
    EMB = SentenceTransformerEmbedder(S.embedding_model)
    EMB_OK = EMB.is_available()
except Exception:
    EMB_OK = False
LLM = OllamaLLM(S.ollama_host, S.llm_model, S.temperature, S.num_ctx)
LLM_OK = LLM.status()[0] == "ok"


@pytest.mark.skipif(not EMB_OK, reason="embedding model not downloaded locally")
def test_real_embedding_retrieval(tmp_path):
    s = load_settings(data_dir=tmp_path)
    eng = RAGEngine(s, EMB, VectorStore(s.chroma_dir, EMB), LLM, DocumentRegistry(s.registry_path))
    eng.ingest("t.pdf", make_pdf(["The warranty lasts 24 months from the purchase date.", "Our pancake recipe uses flour, eggs and milk."]))
    top = eng.store.query("How long is the guarantee period?", 1)[0]
    assert top.page == 1


@pytest.mark.skipif(not (EMB_OK and LLM_OK), reason="needs embedding model and Ollama model")
def test_real_end_to_end(tmp_path):
    s = load_settings(data_dir=tmp_path)
    eng = RAGEngine(s, EMB, VectorStore(s.chroma_dir, EMB), LLM, DocumentRegistry(s.registry_path))
    eng.ingest("t.pdf", make_pdf(["The warranty lasts 24 months from the purchase date."]))
    ans = eng.ask("How long does the warranty last?")
    assert ans.grounded and "24" in ans.text and ans.sources[0].filename == "t.pdf"
