"""Streamlit UI for the local PDF chatbot."""
from __future__ import annotations

import logging

import streamlit as st

from ragchat.config import load_settings
from ragchat.embeddings import SentenceTransformerEmbedder
from ragchat.llm import OllamaLLM
from ragchat.models import EmbeddingModelUnavailable, LLMUnavailable, UserFacingError
from ragchat.pdf_processing import ocr_available
from ragchat.rag import RAGEngine
from ragchat.registry import DocumentRegistry
from ragchat.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

st.set_page_config(page_title="Local PDF Chat", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_engine() -> tuple[RAGEngine, SentenceTransformerEmbedder, OllamaLLM]:
    s = load_settings()
    emb = SentenceTransformerEmbedder(s.embedding_model, s.embedding_download_hint)
    llm = OllamaLLM(s.ollama_host, s.llm_model, s.temperature, s.num_ctx)
    return RAGEngine(s, emb, VectorStore(s.chroma_dir, emb), llm, DocumentRegistry(s.registry_path)), emb, llm


engine, embedder, llm = get_engine()
cfg = engine.s
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("flash", [])

emb_ok = embedder.is_available()
llm_state, llm_msg = llm.status()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📚 Documents")

    if not emb_ok:
        st.warning(
            f"The embedding model **{cfg.embedding_model}** is not on this computer yet "
            f"(download: {cfg.embedding_download_hint}, one time)."
        )
        if st.button("Download embedding model", type="primary"):
            with st.spinner("Downloading…"):
                try:
                    embedder.download()
                    st.rerun()
                except UserFacingError as e:
                    st.error(e.user_message)
    if llm_state != "ok":
        st.warning(llm_msg)
        if llm_state == "no_model":
            st.caption("Models of ~2 GB download once. Alternatives: `qwen2.5:3b` (~2 GB), `gemma3:4b` (~3.3 GB).")

    stale = engine.stale_documents() if emb_ok else []
    if stale:
        st.warning(f"{len(stale)} document(s) were indexed with a different embedding model and are not searchable.")
        if st.button("Re-index them"):
            with st.spinner("Re-indexing…"):
                failed = engine.reindex_stale()
            if failed:
                st.error("Could not re-index: " + ", ".join(failed))
            st.rerun()

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key=f"up{ss.uploader_key}")
    use_ocr = st.checkbox(
        "Use OCR for scanned pages", value=False, disabled=not ocr_available(cfg.tesseract_cmd),
        help="Requires Tesseract. See README → Optional OCR.",
    )
    if st.button("Process uploaded PDFs", type="primary", disabled=not files or not emb_ok):
        ss.flash = []
        for f in files:
            with st.status(f"{f.name}", expanded=True) as status:
                try:
                    res = engine.ingest(f.name, f.getvalue(), ocr=use_ocr, progress=lambda m: st.write(m))
                    if res.status == "duplicate":
                        ss.flash.append(("info", f"“{f.name}” is already indexed."))
                        status.update(label=f"{f.name} — already indexed", state="complete")
                    else:
                        ss.flash.append(("success", f"Indexed “{f.name}”: {res.record.pages} pages, {res.record.chunks} passages."))
                        status.update(label=f"{f.name} — done", state="complete")
                except UserFacingError as e:
                    ss.flash.append(("error", e.user_message))
                    status.update(label=f"{f.name} — failed", state="error")
                except Exception:
                    log.exception("ingest failed")
                    ss.flash.append(("error", f"Something went wrong while processing “{f.name}”. See the terminal log."))
                    status.update(label=f"{f.name} — failed", state="error")
        ss.uploader_key += 1
        st.rerun()
    for kind, msg in ss.flash:
        getattr(st, kind)(msg)

    docs = engine.list_documents()
    st.subheader(f"Indexed ({len(docs)})")
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d.filename}**  \n{d.pages} pages · {d.chunks} passages" + (" · OCR" if d.ocr else ""))
        if c2.button("🗑", key=f"del_{d.doc_id}", help="Remove this document and its index data"):
            engine.delete_document(d.doc_id)
            st.rerun()

    st.divider()
    if st.button("Clear chat"):
        ss.messages = []
        st.rerun()
    with st.expander("Reset everything"):
        st.caption("Deletes all indexed documents, stored PDF copies and chat history.")
        if st.checkbox("I understand") and st.button("Reset session"):
            engine.reset_all()
            ss.messages = []
            st.rerun()

# ---------------- main ----------------
st.title("📄 Local PDF Chat")
st.caption("Ask questions about your PDFs. Everything runs on this computer — no cloud AI, no API keys.")


def show_sources(m: dict) -> None:
    if not m.get("sources"):
        return
    title = "Sources" if m.get("verified", True) else "Retrieved passages (the model did not mark which it used)"
    with st.expander(f"{title} ({len(m['sources'])})"):
        for s in m["sources"]:
            st.markdown(f"**{s['filename']} — page {s['page']}**")
            st.caption(f"“{s['excerpt']}”")


if not docs:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process uploaded PDFs** to get started.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m)

can_chat = bool(docs) and emb_ok and llm_state == "ok"
if prompt := st.chat_input("Ask about your documents…", disabled=not can_chat):
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            prep = engine.prepare(prompt, history)
            if prep.messages is None:
                answer = engine.no_info_answer()
            else:
                box, raw = st.empty(), ""
                for tok in llm.stream(prep.messages):
                    raw += tok
                    box.markdown(raw + "▌")
                answer = engine.finalize(raw, prep.chunks)
            msg = {
                "role": "assistant", "content": answer.text, "verified": answer.citations_verified,
                "sources": [s.__dict__ for s in answer.sources],
            }
            st.markdown(msg["content"]) if prep.messages is None else box.markdown(msg["content"])
            show_sources(msg)
            ss.messages.append(msg)
        except (LLMUnavailable, EmbeddingModelUnavailable) as e:
            st.error(e.user_message)
            ss.messages.pop()
        except Exception:
            log.exception("answering failed")
            st.error("Something went wrong while answering. Please try again; details are in the terminal log.")
            ss.messages.pop()
