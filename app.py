import os, json, uuid, hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import streamlit as st
from src.data_ingestion import ingest_documents, EmbeddingsManager, VectorStore
from src.rag import RAGRetriever, rag, llm

DATA_DIR = "./data/uploads"
HISTORY_FILE = "./data/chat_history.json"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

st.set_page_config(page_title="DocuMind AI", page_icon="✦", layout="wide")

# ============================================================
# Dark, glass-panel theme
# ============================================================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
:root {
  --bg:#0B0E14; --panel:#131722; --panel2:#171C29; --border:#232A3B;
  --accent:#39E5B0; --accent2:#5B8CFF; --text:#E7EAF2; --muted:#7C879C;
}
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #14243A 0%, var(--bg) 55%); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
h1,h2,h3,h4,p,span,label,div { color: var(--text); }
.stCaption, .st-emotion-cache-1629p8f, small { color: var(--muted) !important; }

/* Buttons */
.stButton>button {
  background: var(--panel2); color: var(--text); border: 1px solid var(--border);
  border-radius: 10px; font-weight: 500;
}
.stButton>button:hover { border-color: var(--accent); color: var(--accent); }
.stButton>button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #06110D; border: none; font-weight: 700;
}

/* Logo / header */
.dm-logo { display:flex; align-items:center; gap:10px; }
.dm-logo-mark { font-size:1.6rem; background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip:text; color:transparent; }
.dm-logo-text { font-size:1.25rem; font-weight:800; letter-spacing:-.02em; }
.dm-logo-sub { font-size:.72rem; color: var(--muted); margin-top:-4px; }
.dm-pill { display:inline-block; font-size:.68rem; font-weight:700; padding:3px 10px;
  border-radius:999px; background: rgba(57,229,176,.12); color: var(--accent); border:1px solid rgba(57,229,176,.3); }

/* Cards */
.dm-glass { background: var(--panel2); border:1px solid var(--border); border-radius:16px; padding:16px 18px; }
.dm-metric { background: var(--panel2); border:1px solid var(--border); border-radius:16px; padding:18px; text-align:center; }
.dm-metric .num { font-size:1.7rem; font-weight:800; background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.dm-metric .lbl { font-size:.72rem; color:var(--muted); font-weight:700; letter-spacing:.06em; text-transform:uppercase; }

.dm-chip { display:inline-block; background: rgba(91,140,255,.12); color:#9DB6FF; font-size:.76rem;
  font-weight:600; padding:4px 10px; border-radius:999px; margin:2px 4px 2px 0; border:1px solid rgba(91,140,255,.25); }

.dm-hero { text-align:center; padding: 34px 10px 6px; }
.dm-hero .mark { font-size:2.6rem; }
.dm-hero p { color: var(--muted); }

hr.dm-hr { border:none; border-top:1px solid var(--border); margin:12px 0; }

/* chat bubbles */
[data-testid="stChatMessage"] { background: var(--panel2); border:1px solid var(--border); border-radius:14px; }

/* inputs */
.stTextInput input, .stChatInput textarea, textarea { background: var(--panel2) !important;
  color: var(--text) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data helpers
# ============================================================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        data = json.load(open(HISTORY_FILE, encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history():
    json.dump(st.session_state.chats, open(HISTORY_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def create_chat():
    chat = {"id": str(uuid.uuid4()), "title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}
    st.session_state.chats.insert(0, chat)
    st.session_state.active_chat = chat["id"]
    save_history()


def get_active_chat():
    return next((c for c in st.session_state.chats if c.get("id") == st.session_state.active_chat), None)


def delete_chat(chat_id):
    st.session_state.chats = [c for c in st.session_state.chats if c.get("id") != chat_id]
    st.session_state.active_chat = st.session_state.chats[0]["id"] if st.session_state.chats else None
    if not st.session_state.chats:
        create_chat()
    save_history()


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def get_documents():
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]


def duplicate_file(data):
    h = file_hash(data)
    for p in get_documents():
        try:
            if hashlib.sha256(open(p, "rb").read()).hexdigest() == h:
                return os.path.basename(p)
        except Exception:
            pass
    return None


def chips_html(sources):
    return "".join(f'<span class="dm-chip">📄 {s["source"]} · p.{s["page"]}</span>' for s in sources)


@st.cache_resource
def get_backend():
    return RAGRetriever(VectorStore(), EmbeddingsManager())


def ask_question(query, top_k, score_threshold):
    """Run RAG using settings captured on the main Streamlit thread."""
    retriever = get_backend()
    result = rag(
        query, retriever, llm,
        top_k=top_k,
        min_score=score_threshold,
        return_context=True,
    )

    # Retry without threshold filtering if the first retrieval found no sources.
    if not result.get("sources"):
        relaxed = rag(
            query, retriever, llm,
            top_k=top_k,
            min_score=0.0,
            return_context=True,
        )
        if relaxed.get("sources"):
            return relaxed
    return result


@st.cache_resource
def get_executor():
    # Executor jobs continue even when Streamlit reruns after navigation.
    return ThreadPoolExecutor(max_workers=2)


def submit_question(user_input, chat):
    """Save the question and start RAG in a background worker."""
    chat_id = chat["id"]

    # Do not start retrieval when there are no PDFs.
    # Give the user a normal assistant response instead.
    if not get_documents():
        chat["messages"].append({
            "role": "user",
            "content": user_input
        })
        chat["messages"].append({
            "role": "assistant",
            "content": "Please upload a document first. I need at least one PDF to answer questions about your documents.",
            "sources": []
        })
        if chat["title"] == "New Chat":
            chat["title"] = user_input[:30]
        save_history()
        return

    chat["messages"].append({"role": "user", "content": user_input})
    if chat["title"] == "New Chat":
        chat["title"] = user_input[:30]
    save_history()

    # Capture Streamlit settings before entering the worker thread.
    top_k = int(st.session_state.top_k)
    score_threshold = float(st.session_state.score_threshold)

    future = get_executor().submit(
        ask_question, user_input, top_k, score_threshold
    )
    st.session_state.pending_jobs[chat_id] = future


def harvest_completed_jobs():
    """Persist results from finished jobs when the app next reruns."""
    for chat_id, future in list(st.session_state.pending_jobs.items()):
        if not future.done():
            continue

        chat = next(
            (c for c in st.session_state.chats if c.get("id") == chat_id),
            None,
        )

        try:
            result = future.result()
            answer = result.get("answer", "No relevant information found.")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"Generation failed: {e}"
            sources = []

        if chat is not None:
            chat["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            save_history()

        del st.session_state.pending_jobs[chat_id]


def is_chat_pending(chat_id):
    future = st.session_state.pending_jobs.get(chat_id)
    return future is not None and not future.done()


def confirm_delete(flag_key, label, on_confirm):
    st.warning(f"Delete **{label}**?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, delete", key=f"yes_{flag_key}", use_container_width=True, type="primary"):
            on_confirm()
            st.session_state[flag_key] = None
            st.rerun()
    with c2:
        if st.button("Cancel", key=f"no_{flag_key}", use_container_width=True):
            st.session_state[flag_key] = None
            st.rerun()


# ============================================================
# Session state
# ============================================================
defaults = {
    "chats": None, "active_chat": None, "page": "Chat", "top_k": 3, "score_threshold": 0.2,
    "confirm_delete_chat": None, "confirm_delete_doc": None, "pending_question": None,
    "confirm_clear_history": False, "pending_jobs": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if not isinstance(st.session_state.chats, list):
    st.session_state.chats = load_history()
if not st.session_state.chats:
    create_chat()
if not st.session_state.active_chat:
    st.session_state.active_chat = st.session_state.chats[0]["id"]

if not isinstance(st.session_state.pending_jobs, dict):
    st.session_state.pending_jobs = {}
harvest_completed_jobs()


# ============================================================
# Sidebar — chat history only
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="dm-logo"><span class="dm-logo-mark">✦</span>'
        '<div><div class="dm-logo-text">DocuMind</div>'
        '<div class="dm-logo-sub">AI Document Q&amp;A</div></div></div>'
        '<div style="margin-top:8px;"><span class="dm-pill">🔒 Offline &amp; Private</span></div>'
        '<hr class="dm-hr">',
        unsafe_allow_html=True,
    )

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        create_chat()
        st.session_state.page = "Chat"
        st.rerun()

    st.caption("RECENT CHATS")
    for chat in st.session_state.chats:
        title = chat.get("title", "New Chat")
        title = title[:24] + "…" if len(title) > 24 else title
        active = chat["id"] == st.session_state.active_chat and st.session_state.page == "Chat"

        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(("● " if active else "○ ") + title, key=f"open_{chat['id']}",
                         use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.active_chat = chat["id"]
                st.session_state.page = "Chat"
                st.session_state.confirm_delete_chat = None
                st.rerun()
        with c2:
            if st.button("✕", key=f"del_{chat['id']}"):
                st.session_state.confirm_delete_chat = chat["id"]
                st.rerun()

        if st.session_state.confirm_delete_chat == chat["id"]:
            confirm_delete("confirm_delete_chat", title, lambda cid=chat["id"]: delete_chat(cid))

    st.markdown('<hr class="dm-hr">', unsafe_allow_html=True)
    st.caption(f"{len(get_documents())} document(s) · {len(st.session_state.chats)} chat(s)")


# ============================================================
# Top nav (tabs replace the old sidebar page-switcher)
# ============================================================
top_l, top_r = st.columns([3, 2])
with top_l:
    st.markdown(
        '<div class="dm-logo"><span class="dm-logo-mark" style="font-size:2rem;">✦</span>'
        '<div><div class="dm-logo-text" style="font-size:1.5rem;">DocuMind AI</div>'
        '<div class="dm-logo-sub">Local RAG assistant, zero cloud calls</div></div></div>',
        unsafe_allow_html=True,
    )
with top_r:
    n1, n2, n3 = st.columns(3)
    for col, key, label in zip((n1, n2, n3),
                                ("Chat", "Documents", "Settings"),
                                ("💬 Chat", "📄 Documents", "⚙️ Settings")):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == key else "secondary", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

st.markdown('<hr class="dm-hr">', unsafe_allow_html=True)


# ============================================================
# CHAT
# ============================================================
if st.session_state.page == "Chat":
    chat = get_active_chat() or (create_chat() or get_active_chat())

    if not chat["messages"]:
        docs = get_documents()
        st.markdown(
            '<div class="dm-hero"><div class="mark">✦</div><h1>Ask anything about your files</h1>'
            '<p>Everything runs locally — your documents never leave this machine.</p></div>',
            unsafe_allow_html=True,
        )
        if not docs:
            st.info("No documents indexed yet. Switch to **📄 Documents** to upload your first PDF.")
        else:
            suggestions = [
                "Summarize this document in a few bullet points",
                "What are the key takeaways?",
                "List any important dates or numbers mentioned",
            ]
            for col, s in zip(st.columns(len(suggestions)), suggestions):
                with col:
                    if st.button(s, use_container_width=True, key=f"sugg_{s}"):
                        st.session_state.pending_question = s
    # Render all saved messages, including answers completed after navigation.
    for m in chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                with st.expander(f"📚 {len(m['sources'])} source(s) used"):
                    st.markdown(chips_html(m["sources"]), unsafe_allow_html=True)

    # Keep a simple loading message visible while the background RAG task runs.
    # Unlike the old warning, this does not show navigation instructions or a
    # "Check response" button.
    if is_chat_pending(chat["id"]):
        st.info("⏳ Reading your documents and generating your answer...")

    user_input = st.chat_input(
        "Ask something about your documents...",
        disabled=is_chat_pending(chat["id"]),
    )

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        submit_question(q, chat)
        st.rerun()

    if user_input:
        submit_question(user_input, chat)
        st.rerun()


# ============================================================
# DOCUMENTS
# ============================================================
elif st.session_state.page == "Documents":
    documents = sorted(get_documents())
    total_kb = sum(os.path.getsize(p) for p in documents) / 1024

    for col, num, lbl in zip(
        st.columns(3),
        [len(documents), f"{total_kb:.0f} KB", len(st.session_state.chats)],
        ["DOCUMENTS", "TOTAL SIZE", "CHATS"],
    ):
        col.markdown(f'<div class="dm-metric"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="dm-glass">', unsafe_allow_html=True)
    st.markdown("**Upload PDFs**")
    uploaded_files = st.file_uploader("Drag & drop or browse", type=["pdf"],
                                       accept_multiple_files=True, label_visibility="collapsed")
    new_files = []
    if uploaded_files:
        for f in uploaded_files:
            data = f.getvalue()
            dup = duplicate_file(data)
            if dup:
                st.warning(f"⚠️ **{f.name}** already exists as `{dup}` — skipped.")
            else:
                new_files.append(f)
                st.success(f"✅ **{f.name}** ready ({len(data)/1024:.1f} KB)")

        if new_files and st.button("⚙️ Process Documents", use_container_width=True, type="primary"):
            for f in new_files:
                open(os.path.join(DATA_DIR, f.name), "wb").write(f.getbuffer())
            with st.spinner("Extracting, chunking and creating embeddings..."):
                try:
                    ingest_documents(DATA_DIR)
                    get_backend.clear()
                    st.success("✅ Documents processed successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("**Your library**")

    if not documents:
        st.info("No documents uploaded yet.")
    for path in documents:
        filename = os.path.basename(path)
        size = os.path.getsize(path) / 1024
        key = hashlib.md5(filename.encode()).hexdigest()

        st.markdown('<div class="dm-glass" style="margin-bottom:10px;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([6, 2, 1])
        c1.markdown(f"📄 **{filename}**")
        c2.caption(f"{size:.1f} KB")
        with c3:
            if st.button("✕", key=f"doc_del_{key}"):
                st.session_state.confirm_delete_doc = path
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.confirm_delete_doc == path:
            def do_delete(p=path):
                os.remove(p)
                with st.spinner("Updating knowledge base..."):
                    ingest_documents(DATA_DIR)
                get_backend.clear()
            confirm_delete("confirm_delete_doc", filename, do_delete)


# ============================================================
# SETTINGS
# ============================================================
elif st.session_state.page == "Settings":
    st.markdown('<div class="dm-glass">', unsafe_allow_html=True)
    st.markdown("**🔍 Retrieval**")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.top_k = st.slider("Chunks to retrieve", 1, 10, st.session_state.top_k,
                                            help="Number of relevant chunks sent to the LLM for each answer.")
    with c2:
        st.session_state.score_threshold = st.slider("Similarity threshold", 0.0, 1.0,
                                                       st.session_state.score_threshold, 0.05,
                                                       help="Lower values allow less similar (looser) matches through.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    s1, s2 = st.columns(2)
    s1.markdown(
        '<div class="dm-glass"><b>🧠 LLM</b><br>Mistral via Ollama<br><br>'
        '<b>🧬 Embedding model</b><br>all-MiniLM-L6-v2<br><br>'
        '<b>🗂️ Vector database</b><br>ChromaDB</div>', unsafe_allow_html=True)
    s2.markdown(
        '<div class="dm-glass"><b>🖥️ Frontend</b><br>Streamlit<br><br>'
        '<b>🌐 Operation mode</b><br>Online / Local &amp; Offline<br><br>'
        '<b>🔒 Privacy</b><br>No data leaves your machine</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="dm-glass">', unsafe_allow_html=True)
    st.markdown("**⚠️ Danger zone**")
    if not st.session_state.confirm_clear_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.confirm_clear_history = True
            st.rerun()
    else:
        def clear_all():
            st.session_state.chats = []
            create_chat()
        confirm_delete("confirm_clear_history", "all chat history", clear_all)
    st.markdown("</div>", unsafe_allow_html=True)