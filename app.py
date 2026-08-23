"""
app.py
------
Streamlit UI for the PDF Q&A Chatbot (RAG System).

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from rag import RAGPipeline

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------------
# Theme (light / dark) — toggle stored in session_state, CSS rebuilt each run
# --------------------------------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

LIGHT = {
    "bg": "#FFFFFF",
    "bg_secondary": "#F7F7FA",
    "sidebar_bg": "#F7F7FA",
    "text": "#1A1A1E",
    "text_muted": "#6B6B76",
    "border": "#E4E4EA",
    "accent": "#6C5CE7",
    "accent_text": "#FFFFFF",
    "input_bg": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "user_bubble": "#E8F0FF",
    "user_bubble_text": "#0F2A5C",
    "bot_bubble": "#F1F1F5",
    "bot_bubble_text": "#1A1A1E",
    "chip_bg": "#EDEBFF",
    "chip_text": "#4B3F92",
    "shadow": "0 1px 3px rgba(0,0,0,0.06)",
}

DARK = {
    "bg": "#0E0F13",
    "bg_secondary": "#15171D",
    "sidebar_bg": "#15171D",
    "text": "#EDEDF0",
    "text_muted": "#9A9AA5",
    "border": "#2A2C35",
    "accent": "#8B7FFF",
    "accent_text": "#0E0F13",
    "input_bg": "#1C1E26",
    "card_bg": "#1A1C24",
    "user_bubble": "#1E3358",
    "user_bubble_text": "#DCE9FF",
    "bot_bubble": "#1F212A",
    "bot_bubble_text": "#EDEDF0",
    "chip_bg": "#2B2650",
    "chip_text": "#CFC6FF",
    "shadow": "0 1px 3px rgba(0,0,0,0.4)",
}

T = DARK if st.session_state.dark_mode else LIGHT

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {T['bg']};
        color: {T['text']};
    }}
    .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp .stMarkdown, .stApp .stCaption {{
        color: {T['text']};
    }}
    [data-testid="stCaptionContainer"], .stCaption, small {{
        color: {T['text_muted']} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {T['text']};
    }}

    /* Header */
    header[data-testid="stHeader"] {{
        background-color: {T['bg']};
    }}

    /* Inputs, selects, textareas, file uploader */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput input,
    .stTextArea textarea,
    div[data-testid="stFileUploaderDropzone"] {{
        background-color: {T['input_bg']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {T['text']} !important;
    }}

    /* Buttons */
    button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background-color: {T['accent']} !important;
        color: {T['accent_text']} !important;
        border: none !important;
    }}
    button[data-testid="baseButton-secondary"] {{
        background-color: {T['card_bg']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
    }}

    /* Slider */
    div[data-testid="stSlider"] [role="slider"] {{
        background-color: {T['accent']} !important;
    }}

    /* Chat input */
    div[data-testid="stChatInput"] {{
        background-color: {T['input_bg']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 14px !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {T['text']} !important;
        background-color: transparent !important;
    }}

    /* Expander */
    details {{
        background-color: {T['card_bg']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
    }}

    /* Divider */
    hr {{
        border-color: {T['border']} !important;
    }}

    /* App title */
    .app-title {{
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }}

    /* Chat bubbles */
    .chat-row {{
        display: flex;
        margin-bottom: 14px;
    }}
    .chat-row.user {{ justify-content: flex-end; }}
    .chat-row.bot {{ justify-content: flex-start; }}
    .chat-bubble-user {{
        background-color: {T['user_bubble']};
        color: {T['user_bubble_text']};
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        max-width: 75%;
        box-shadow: {T['shadow']};
        line-height: 1.5;
    }}
    .chat-bubble-bot {{
        background-color: {T['bot_bubble']};
        color: {T['bot_bubble_text']};
        padding: 12px 16px;
        border-radius: 16px 16px 16px 4px;
        max-width: 80%;
        box-shadow: {T['shadow']};
        line-height: 1.5;
    }}
    .chat-avatar {{
        font-size: 1.3rem;
        margin: 0 8px;
    }}

    /* Source chips */
    .source-chip {{
        display: inline-block;
        background-color: {T['chip_bg']};
        color: {T['chip_text']};
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.76rem;
        font-weight: 500;
        margin: 3px 5px 3px 0;
    }}

    /* Empty-state card */
    .empty-state {{
        background-color: {T['card_bg']};
        border: 1px dashed {T['border']};
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        color: {T['text_muted']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(persist_dir="vectordb")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set(st.session_state.pipeline.store.list_sources())

pipeline = st.session_state.pipeline

# --------------------------------------------------------------------------
# Sidebar: upload + settings
# --------------------------------------------------------------------------

with st.sidebar:
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown('<div class="app-title">📚 PDF Q&A Chatbot</div>', unsafe_allow_html=True)
        st.caption("Retrieval-Augmented Generation over your own documents")
    with top_right:
        st.toggle("🌙", value=st.session_state.dark_mode, key="dark_mode_toggle",
                  help="Toggle dark / light mode")
        st.session_state.dark_mode = st.session_state.dark_mode_toggle

    st.subheader("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process & Index PDFs", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting...")
            total = len(uploaded_files)
            for i, uf in enumerate(uploaded_files, start=1):
                progress.progress((i - 1) / total, text=f"Indexing {uf.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name
                try:
                    n_chunks = pipeline.ingest_pdf(tmp_path, uf.name)
                    st.session_state.indexed_files.add(uf.name)
                    st.toast(f"{uf.name}: {n_chunks} chunks indexed", icon="✅")
                finally:
                    os.remove(tmp_path)
                progress.progress(i / total, text=f"Indexed {uf.name}")
            progress.empty()
            st.success(f"Indexed {total} file(s). {pipeline.store.count()} chunks total.")

    st.subheader("2. Choose documents to search")
    all_sources = sorted(st.session_state.indexed_files)
    if all_sources:
        selected_sources = st.multiselect(
            "Limit search to (leave empty = all)", all_sources, default=[]
        )
    else:
        selected_sources = []
        st.info("Upload and index a PDF to get started.")

    st.subheader("3. LLM settings")
    provider = st.selectbox("Answer generation", ["none (extractive only)", "openai", "groq"])
    provider_key = "none" if provider.startswith("none") else provider
    api_key = ""
    model_name = ""
    if provider_key != "none":
        api_key = st.text_input(f"{provider_key.upper()} API key", type="password")
        default_model = "gpt-4o-mini" if provider_key == "openai" else "openai/gpt-oss-20b"
        model_name = st.text_input("Model", value=default_model)
        if provider_key == "groq":
            st.caption(
                "Groq rotates/deprecates model names over time. If you get a "
                "'model not found' error, check the current list at "
                "console.groq.com/docs/models and paste a valid model id above."
            )
        st.caption("Key is used only for this session and never saved to disk.")

    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=3)

    st.divider()
    if st.button("🗑️ Clear indexed documents", use_container_width=True):
        pipeline.store.clear()
        st.session_state.indexed_files = set()
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------

st.header("Chat with your PDFs")

if not st.session_state.indexed_files:
    st.markdown(
        '<div class="empty-state">👈 Upload and index at least one PDF from the '
        'sidebar to start asking questions.</div>',
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-row user"><div class="chat-bubble-user">{msg["content"]}</div>'
            f'<span class="chat-avatar">🧑</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-row bot"><span class="chat-avatar">🤖</span>'
            f'<div class="chat-bubble-bot">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s["source"]} · p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)
            with st.expander("View retrieved excerpts"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}, page {s['page']}** (score {s['score']})")
                    st.write(s["text"])
                    st.divider()

# Chat input
question = st.chat_input("Ask a question about your uploaded PDFs...")

if question:
    if not st.session_state.indexed_files:
        st.error("Please upload and index a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Searching documents and generating answer..."):
            result = pipeline.ask(
                question,
                k=top_k,
                sources=selected_sources or None,
                provider=provider_key,
                api_key=api_key,
                model=model_name,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()