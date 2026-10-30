# Local PDF Chat (free, private RAG)

Upload PDFs, ask questions, get answers with **filename + page citations**. Everything runs on your computer:
no OpenAI/paid API, no API keys, no cloud AI, no telemetry. Target machine: Windows, 16 GB RAM, CPU only.

## Stack
Streamlit · PyMuPDF · Sentence Transformers (`all-MiniLM-L6-v2`) · ChromaDB (persistent, local) · Ollama (`llama3.2:3b`, ~2 GB)

## Pipeline
PDF upload → validation → per-page text extraction (+ optional Tesseract OCR) → cleaning → page-aware chunking
(default 1000 chars, 150 overlap, chunks never cross pages) → local embeddings → ChromaDB (cosine) → top-k retrieval
with similarity threshold → grounded prompt → Ollama → citation check → answer + sources.

Every chunk stores filename, 1-based PDF page, chunk id and document id (a SHA-256 content hash). No reranker is
used: it would add another model on a CPU-only machine and was not measured to help.

## Setup (Windows PowerShell)
```powershell
# 1. Python 3.12: https://www.python.org/downloads/  (check "Add to PATH")
cd pdf-rag-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt       # large: pulls PyTorch (internet needed once)

# 2. Ollama: https://ollama.com/download  then, in a new terminal:
ollama pull llama3.2:3b               # ~2 GB, one time

# 3. Run
streamlit run app.py
```
On first start the sidebar asks before downloading the embedding model (about 90 MB). After the models are on disk,
indexing and answering work offline. Alternatives for `LLM_MODEL` in `.env`: `qwen2.5:3b` (~2 GB), `gemma3:4b` (~3.3 GB),
`llama3.2:1b` (~1.3 GB, weaker). Model sizes come from the Ollama library listings; check ollama.com/library for current ones.

### Optional OCR (scanned PDFs)
Install Tesseract for Windows (https://github.com/UB-Mannheim/tesseract/wiki), put it on PATH or set `TESSERACT_CMD` in `.env`,
then tick “Use OCR for scanned pages”. Without it, scanned PDFs are rejected with an explanation.

## Tests
```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```
Unit tests use a fake embedder and a mocked LLM. `tests/test_integration_local.py` runs only if the real embedding
model and Ollama model are installed; otherwise it is skipped.

## Configuration
All settings live in `ragchat/config.py`; override via `.env` (see `.env.example`, no secrets needed).
Key ones: `LLM_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SIMILARITY` (default 0.25, an untuned
starting value: raise it if answers cite irrelevant passages, lower it if relevant ones are dropped).
Changing `EMBEDDING_MODEL` makes old documents unsearchable until you click “Re-index them” (original PDFs are kept for that).

## Privacy
Data lives in `./data/` (`uploads/` PDF copies, `chroma/` vector index, `registry.json`, plain files on disk, unencrypted).
Removing a document deletes its chunks, registry entry and stored PDF copy. “Reset everything” wipes all of it.
Nothing is sent to external services during use; Streamlit and Chroma telemetry are disabled. `data/` is git-ignored.

## Limitations
- Small local models can misread passages or skip citations; if none are marked, the UI says so and lists retrieved passages instead.
- Citations are validated against retrieved passages, but whether a cited passage truly supports a sentence is not verified.
- Follow-up questions of 6 words or fewer that match nothing on their own are retried with the previous question; an unrelated
  short question can therefore reach the model, which is instructed to answer “not enough information”.
- Chunks do not span pages, so facts split across a page break may be retrieved separately. Tables and multi-column layouts extract imperfectly.
- Two different files with the same name are told apart in citations by a short id.
- CPU generation of a 3B model is slow (several seconds to a minute per answer).

## Troubleshooting
- *“Ollama is not running”*: start the Ollama app, then reload. `ollama list` should show your model.
- *Model not installed*: `ollama pull llama3.2:3b`.
- *Embedding download fails*: check internet, click the sidebar button again.
- *PowerShell won't activate venv*: `Set-ExecutionPolicy -Scope Process Bypass`.
- *Slow answers / out of memory*: set `LLM_MODEL=llama3.2:1b` or lower `TOP_K`.
