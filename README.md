# 📚 PDF Q&A Chatbot — RAG System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from
your own PDFs, with page-level citations, multi-PDF support, and chat
history — built with Streamlit, ChromaDB, and Sentence-Transformers.

Instead of answering from general knowledge, it answers **only from the
documents you upload**, and tells you exactly which file and page each
answer came from.

## How it works (architecture)

```
PDF Upload → Text Extraction (pypdf) → Chunking → Embeddings
(sentence-transformers) → Vector DB (ChromaDB) → User Question
→ Similarity Search → Relevant Chunks → LLM → Final Answer + Citations
```

- **Embeddings run locally** (`all-MiniLM-L6-v2` via `sentence-transformers`),
  so indexing your PDFs never needs an API key or costs anything.
- **Answer generation** is optional: without an API key the app still works
  in *extractive mode*, showing you the best-matching excerpts directly.
  Add an OpenAI or Groq key in the sidebar to get natural-language answers
  instead.

## Project structure

```
pdf-qa-chatbot/
├── documents/        # (optional) drop PDFs here if you prefer file-based ingestion
├── vectordb/          # ChromaDB's persistent storage (auto-created)
├── app.py             # Streamlit UI
├── rag.py             # Core RAG engine (loading, chunking, embeddings, retrieval, LLM)
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > First run will download the `all-MiniLM-L6-v2` embedding model
   > (~80MB) from Hugging Face — this needs an internet connection once,
   > then it's cached locally.

3. **(Optional) Get an LLM API key** if you want generated answers instead
   of raw excerpts:
   - OpenAI: https://platform.openai.com/api-keys
   - Groq (free tier, very fast): https://console.groq.com/keys

## Run the app

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501` in your browser.

## Using it

1. **Upload PDFs** in the sidebar (one or more) and click **Process & Index
   PDFs**. You'll see how many chunks were created per file.
2. **(Optional)** Limit search to specific documents using the multiselect.
3. **(Optional)** Pick a provider (OpenAI/Groq) and paste your API key to
   get generated answers with citations, instead of raw excerpts. The key
   is only kept in the browser session, never written to disk.
4. **Ask questions** in the chat box. Each answer shows source chips
   (`filename · page`) and an expandable panel with the exact retrieved
   text.
5. Use **🗑️ Clear indexed documents** in the sidebar to wipe the vector
   store and start fresh.

## Key design choices / how it maps to the RAG pipeline

| Step | Implementation |
|---|---|
| Text extraction | `pypdf.PdfReader`, page-by-page so citations can include page numbers |
| Chunking | Custom sliding-window splitter (~1000 chars, 200 overlap), breaking on paragraph/sentence boundaries where possible |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, runs locally, no API key |
| Vector DB | `chromadb.PersistentClient`, data persisted to `vectordb/` |
| Retrieval | Cosine similarity search, top-k configurable in the sidebar |
| Generation | OpenAI or Groq chat completion, grounded strictly in retrieved chunks; extractive fallback when no key is set |
| UI | Streamlit chat interface with source citation chips and expandable excerpts |

## Extending it (mini-challenge ideas)

- **OCR for scanned PDFs**: swap `pypdf` text extraction for
  `pytesseract` + `pdf2image` when a page returns no extractable text.
- **Hybrid search**: combine ChromaDB similarity search with a keyword
  index (e.g. `rank_bm25`) and merge/re-rank results.
- **Conversation memory**: pass the last N chat turns into the prompt in
  `rag.py::build_prompt` so follow-up questions can refer to earlier ones.
- **Multi-agent RAG**: add a query-rewriting step before retrieval, or a
  verifier step that checks the answer against the retrieved chunks.

## Troubleshooting

- **"No module named 'sentence_transformers'"** → re-run
  `pip install -r requirements.txt` inside your active virtual environment.
- **Slow first run** → the embedding model downloads once and is cached in
  `~/.cache/huggingface`; subsequent runs are fast.
- **`ModuleNotFoundError: chromadb`** on Windows → make sure you're on
  Python 3.9–3.12; also try `pip install --upgrade pip` first.
- **Empty answers** → check the sidebar shows a non-zero chunk count after
  indexing; some scanned/image-only PDFs extract no text (see OCR note
  above).

## Resume-ready description

> **PDF Q&A Chatbot (RAG System)** — Built a Retrieval-Augmented Generation
> application in Python using ChromaDB, Sentence-Transformers, and
> Streamlit. Implemented PDF ingestion, page-aware chunking, local vector
> embeddings, similarity search, and LLM-grounded answer generation with
> source citations across multiple documents.
