"""
rag.py
------
Core Retrieval-Augmented Generation (RAG) engine for the PDF Q&A Chatbot.

Pipeline:
    PDF -> text extraction (per page) -> chunking -> embeddings
    -> vector store (Chroma) -> similarity search -> LLM answer w/ citations

Embeddings run locally (sentence-transformers) so no API key is required
just to index documents. Only the final answer-generation step needs an
LLM API key (OpenAI or Groq), and the app works in "extractive" mode
(no key needed) if you don't have one.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str          # original filename
    page: int             # 1-indexed page number
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


# --------------------------------------------------------------------------
# PDF loading
# --------------------------------------------------------------------------

def load_pdf(file_path: str) -> List[dict]:
    """Extract text from a PDF, page by page.

    Returns a list of {"page": int, "text": str} dicts (1-indexed pages).
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(pages: List[dict], filename: str, chunk_size: int = 1000,
               chunk_overlap: int = 200) -> List[Chunk]:
    """Split page text into overlapping chunks, preserving page numbers.

    A simple sliding-window character splitter that tries to break on
    sentence/paragraph boundaries when possible (mirrors what
    RecursiveCharacterTextSplitter does, without the extra dependency).
    """
    chunks: List[Chunk] = []
    separators = ["\n\n", "\n", ". ", " "]

    for page in pages:
        text = page["text"]
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)

            # try to break on a natural boundary near `end`
            if end < n:
                best_break = -1
                for sep in separators:
                    idx = text.rfind(sep, start, end)
                    if idx != -1 and idx > best_break:
                        best_break = idx + len(sep)
                if best_break != -1 and best_break > start:
                    end = best_break

            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=filename, page=page["page"]))

            if end <= start:
                break
            start = max(end - chunk_overlap, end) if chunk_overlap >= (end - start) else end - chunk_overlap
            if start <= 0:
                start = end

    return chunks


# --------------------------------------------------------------------------
# Vector store wrapper (ChromaDB, persistent, local embeddings)
# --------------------------------------------------------------------------

class VectorStore:
    def __init__(self, persist_dir: str = "vectordb", collection_name: str = "pdf_qa",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def similarity_search(self, query: str, k: int = 3,
                           sources: Optional[List[str]] = None) -> List[RetrievedChunk]:
        where = {"source": {"$in": sources}} if sources else None
        results = self.collection.query(query_texts=[query], n_results=k, where=where)

        out: List[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(
                text=doc, source=meta["source"], page=meta["page"],
                score=1 - dist  # convert distance -> similarity-ish score
            ))
        return out

    def list_sources(self) -> List[str]:
        data = self.collection.get()
        sources = {m["source"] for m in data.get("metadatas", [])}
        return sorted(sources)

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, embedding_function=self.embed_fn
        )

    def count(self) -> int:
        return self.collection.count()


# --------------------------------------------------------------------------
# Answer generation
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context extracted from the user's PDF documents. "
    "Every fact you state must come from the context. "
    "If the answer isn't in the context, say you couldn't find it in the "
    "uploaded documents — do not make anything up. "
    "Cite the source file and page number for each claim, e.g. (source.pdf, p.8)."
)


def build_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[{r.source}, page {r.page}]\n{r.text}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context from uploaded documents:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite (filename, page) for each claim."
    )


def generate_answer(question: str, retrieved: List[RetrievedChunk],
                     provider: str = "none", api_key: str = "",
                     model: str = "") -> str:
    """Generate a final answer. Falls back to extractive mode (no LLM key)."""

    if not retrieved:
        return "I couldn't find anything relevant to that question in the uploaded documents."

    if provider == "none" or not api_key:
        # Extractive fallback: no LLM key configured, just surface the best chunks.
        lines = ["*(No LLM API key set — showing the most relevant excerpts instead of a generated answer.)*", ""]
        for r in retrieved:
            snippet = r.text[:400] + ("..." if len(r.text) > 400 else "")
            lines.append(f"**{r.source}, page {r.page}:** {snippet}")
        return "\n\n".join(lines)

    prompt = build_prompt(question, retrieved)

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    elif provider == "groq":
        from groq import Groq, NotFoundError
        client = Groq(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=model or "openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except NotFoundError:
            return (
                f"⚠️ The model `{model}` isn't available on Groq right now "
                "(providers regularly retire/rename models). Open the sidebar "
                "and try `openai/gpt-oss-20b` or `llama-3.3-70b-versatile`, or "
                "check the current list at console.groq.com/docs/models."
            )

    raise ValueError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------
# High-level pipeline class used by the Streamlit app
# --------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, persist_dir: str = "vectordb"):
        self.store = VectorStore(persist_dir=persist_dir)

    def ingest_pdf(self, file_path: str, filename: str,
                    chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        pages = load_pdf(file_path)
        chunks = chunk_text(pages, filename, chunk_size, chunk_overlap)
        self.store.add_chunks(chunks)
        return len(chunks)

    def ask(self, question: str, k: int = 3, sources: Optional[List[str]] = None,
             provider: str = "none", api_key: str = "", model: str = "") -> dict:
        retrieved = self.store.similarity_search(question, k=k, sources=sources)
        answer = generate_answer(question, retrieved, provider=provider,
                                  api_key=api_key, model=model)
        return {
            "answer": answer,
            "sources": [
                {"source": r.source, "page": r.page, "score": round(r.score, 3), "text": r.text}
                for r in retrieved
            ],
        }
