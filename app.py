"""
Research Paper Agent — Streamlit Cloud Edition
All logic is inline (no separate FastAPI backend required).
Set ANTHROPIC_API_KEY in Streamlit Secrets.
"""

import io
import os
import uuid
import json
import textwrap

import arxiv
import pypdf
import anthropic
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background: #0d0f14; color: #e8eaf0; }
  section[data-testid="stSidebar"] { background: #13161f !important; border-right: 1px solid #1e2433; }
  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    font-weight: 600; letter-spacing: 0.5px;
    padding: 0.5rem 1.2rem; transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stTextInput input, .stTextArea textarea {
    background: #1a1d28 !important; border: 1px solid #2a2f42 !important;
    border-radius: 6px !important; color: #e8eaf0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }
  .stSelectbox > div > div { background: #1a1d28 !important; border: 1px solid #2a2f42 !important; color: #e8eaf0 !important; }
  .user-msg {
    background: #1a2540; border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 14px;
  }
  .assistant-msg {
    background: #161c2c; border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 14px;
  }
  .tag {
    display: inline-block; background: #1e2433; border: 1px solid #2a3452;
    border-radius: 4px; padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8892b0; margin: 2px;
  }
  .hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b2c 50%, #0d0f14 100%);
    border: 1px solid #1e2433; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem;
  }
  .hero h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
  }
  .hero p { color: #8892b0; margin: 0; font-size: 14px; }
  .paper-card {
    background: #13161f; border: 1px solid #1e2433; border-radius: 10px;
    padding: 1rem 1.2rem; margin: 0.6rem 0; transition: border-color 0.2s;
  }
  .paper-card:hover { border-color: #2563eb; }
  .paper-title { font-weight: 600; color: #60a5fa; font-size: 15px; margin-bottom: 4px; }
  .paper-meta { color: #5c6680; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
  .paper-summary { color: #9aa3b8; font-size: 13px; margin-top: 8px; line-height: 1.5; }
  .doc-badge {
    background: #162032; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 6px 12px; margin: 4px 0; font-size: 13px; color: #60a5fa;
  }
  .stat-box { background: #13161f; border: 1px solid #1e2433; border-radius: 8px; padding: 12px; text-align: center; }
  .stat-num { font-size: 1.8rem; font-weight: 700; color: #60a5fa; font-family: 'IBM Plex Mono', monospace; }
  .stat-label { font-size: 11px; color: #5c6680; margin-top: 2px; }
  .mode-pill {
    display: inline-block; background: linear-gradient(135deg, #1e3a5f, #2d1b69);
    border-radius: 20px; padding: 3px 12px; font-size: 11px;
    font-family: 'IBM Plex Mono', monospace; color: #a78bfa; margin-left: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Anthropic client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    # Streamlit Cloud: set via App Settings → Secrets
    # Local: set ANTHROPIC_API_KEY env var
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

MODEL = "claude-opus-4-5"


# ── Core logic (was backend/main.py) ──────────────────────────────────────────
def extract_pdf_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    arxiv_client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in arxiv_client.results(search):
        results.append({
            "title": r.title,
            "authors": [a.name for a in r.authors[:4]],
            "summary": r.summary[:600],
            "published": r.published.strftime("%Y-%m-%d"),
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "categories": r.categories,
        })
    return results


def build_doc_context(docs: dict, max_chars: int = 60_000) -> str:
    if not docs:
        return ""
    parts = []
    for name, info in docs.items():
        snippet = info["text"][: max_chars // max(len(docs), 1)]
        parts.append(f"--- Document: {name} ---\n{snippet}\n")
    return "\n".join(parts)


def run_agent(mode: str, query: str, docs: dict) -> str:
    client = get_client()
    if client is None:
        return "❌ **ANTHROPIC_API_KEY not set.** Add it in Streamlit Secrets (Settings → Secrets) as:\n```\nANTHROPIC_API_KEY = 'sk-ant-...'\n```"

    doc_context = build_doc_context(docs)

    if mode == "summarize":
        if not doc_context:
            return "No documents uploaded yet."
        system = textwrap.dedent("""
            You are an expert academic research assistant.
            Produce a structured summary for each paper:
            - **Title & Authors**
            - **Problem Statement**
            - **Key Contributions**
            - **Methodology**
            - **Results & Findings**
            - **Limitations**
            - **Future Work**
            Use Markdown.
        """).strip()
        user = f"Summarize all of the following papers:\n\n{doc_context}"

    elif mode == "compare":
        if not doc_context:
            return "No documents uploaded yet."
        system = textwrap.dedent("""
            You are an expert at comparative analysis of research papers.
            Compare and contrast across: research goals, methodologies,
            results, strengths/weaknesses, and overall recommendation.
            Use Markdown tables where helpful.
        """).strip()
        user = f"Compare these papers:\n\n{doc_context}"

    elif mode == "critique":
        if not doc_context:
            return "No documents uploaded yet."
        system = textwrap.dedent("""
            You are a rigorous academic peer reviewer. For each paper write a
            critique covering: clarity, methodology, statistical validity,
            novelty, writing quality, and an overall recommendation
            (Accept / Major Revision / Minor Revision / Reject). Use Markdown.
        """).strip()
        user = f"Peer-review critique of these papers:\n\n{doc_context}"

    elif mode == "claims":
        if not doc_context:
            return "No documents uploaded yet."
        system = textwrap.dedent("""
            You are a research analyst. Extract the top factual claims,
            findings, and cited evidence as a numbered Markdown list with:
            - The claim
            - Supporting evidence / data
            - Which paper it's from
        """).strip()
        user = f"Extract key claims from:\n\n{doc_context}"

    elif mode == "arxiv":
        try:
            papers = search_arxiv(query, max_results=5)
        except Exception as e:
            return f"❌ arXiv search failed: {e}"
        system = textwrap.dedent("""
            You are an expert research assistant. Synthesize arXiv search results
            to answer the user's question. Mention specific papers by title,
            highlight key findings, note consensus or disagreement, and
            suggest which papers are most worth reading. Use Markdown.
        """).strip()
        user = f"User question: {query}\n\narXiv results:\n{json.dumps(papers, indent=2)}"

    else:  # chat
        system = textwrap.dedent("""
            You are an expert research assistant. Help users understand and
            analyze academic papers. Answer clearly and precisely, citing
            specific sections when relevant. Use Markdown.
        """).strip()
        user = (f"Documents:\n{doc_context}\n\nQuestion: {query}") if doc_context else query

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = {}   # {filename: {text, size}}
if "arxiv_results" not in st.session_state:
    st.session_state.arxiv_results = []

SID = st.session_state.session_id
DOCS = st.session_state.documents


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
      <div style='font-size:2.5rem'>🔬</div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:16px;font-weight:600;color:#60a5fa'>Research Agent</div>
      <div style='font-size:11px;color:#5c6680;margin-top:4px'>Powered by Claude</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#5c6680'>
      SESSION: <span style='color:#7c3aed'>{SID}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Upload
    st.markdown("#### 📄 Upload Papers")
    uploaded_files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in DOCS:
                with st.spinner(f"Parsing {f.name}…"):
                    try:
                        text = extract_pdf_text(f.read())
                        DOCS[f.name] = {"text": text, "size": f.size}
                        st.success(f"✓ {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # Doc list
    if DOCS:
        st.markdown(f"**{len(DOCS)} document(s) loaded:**")
        for name in list(DOCS.keys()):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class='doc-badge'>
                  📋 {name[:26]}{'…' if len(name)>26 else ''}
                  <span style='color:#5c6680;font-size:11px'>{len(DOCS[name]['text'])//1000}k</span>
                </div>""", unsafe_allow_html=True)
            with col2:
                if st.button("🗑", key=f"del_{name}"):
                    del DOCS[name]
                    st.rerun()
    else:
        st.caption("No documents loaded yet.")

    st.markdown("---")

    # Mode
    st.markdown("#### ⚙️ Agent Mode")
    mode = st.selectbox(
        "Mode",
        ["chat", "summarize", "compare", "critique", "arxiv"],
        format_func=lambda m: {
            "chat": "💬 Chat / Q&A",
            "summarize": "📝 Summarize Papers",
            "compare": "⚖️ Compare Papers",
            "critique": "🎓 Peer Review",
            "arxiv": "🌐 arXiv Search",
        }[m],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>Research Paper Agent 🔬</h1>
  <p>Upload papers · Ask questions · Summarize · Compare · Critique · Search arXiv live</p>
</div>
""", unsafe_allow_html=True)

# Stats
c1, c2, c3, c4 = st.columns(4)
total_chars = sum(len(v["text"]) for v in DOCS.values())
for col, num, label in [
    (c1, len(DOCS), "PAPERS LOADED"),
    (c2, f"{total_chars//1000}k", "CHARS INDEXED"),
    (c3, len(st.session_state.messages)//2, "EXCHANGES"),
    (c4, mode.upper(), "ACTIVE MODE"),
]:
    col.markdown(f"""<div class='stat-box'>
    <div class='stat-num'>{num}</div>
    <div class='stat-label'>{label}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_chat, tab_arxiv = st.tabs(["💬 Research Chat", "🌐 arXiv Explorer"])

# ── Chat tab ───────────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='user-msg'>
              <span style='font-size:11px;font-family:IBM Plex Mono,monospace;color:#5c6680'>YOU</span><br>
              {msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='assistant-msg'>
              <span style='font-size:11px;font-family:IBM Plex Mono,monospace;color:#5c6680'>
                AGENT <span class='mode-pill'>{msg.get('mode','chat')}</span>
              </span>
            </div>""", unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Quick actions
    if DOCS:
        st.markdown("<br>", unsafe_allow_html=True)
        qa, qb, qc, qd = st.columns(4)
        quick_actions = [
            (qa, "📝 Summarize All", "summarize", "Summarize all papers"),
            (qb, "⚖️ Compare Papers", "compare", "Compare all papers"),
            (qc, "🎓 Peer Review", "critique", "Peer-review critique"),
            (qd, "🔑 Extract Claims", "claims", "Extract key claims"),
        ]
        for col, label, act_mode, act_query in quick_actions:
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": label})
                    with st.spinner(f"{label}…"):
                        answer = run_agent(act_mode, act_query, DOCS)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "mode": act_mode})
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    placeholder_map = {
        "chat": "Ask anything about your papers…",
        "summarize": "Ask for a specific kind of summary…",
        "compare": "What aspect should I compare?",
        "critique": "Which part should I critique?",
        "arxiv": "Search arXiv, e.g. 'attention mechanisms in transformers'",
    }
    user_input = st.chat_input(placeholder_map.get(mode, "Ask a question…"))
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Agent thinking…"):
            answer = run_agent(mode, user_input, DOCS)
        st.session_state.messages.append({"role": "assistant", "content": answer, "mode": mode})
        st.rerun()

# ── arXiv tab ──────────────────────────────────────────────────────────────────
with tab_arxiv:
    st.markdown("### 🌐 Live arXiv Search")
    st.caption("Search the arXiv preprint server in real time")

    s1, s2, s3 = st.columns([5, 1, 1])
    with s1:
        arxiv_query = st.text_input("Query", placeholder="e.g. 'RLHF language models'", label_visibility="collapsed")
    with s2:
        n_results = st.selectbox("N", [3, 5, 8, 10], index=1, label_visibility="collapsed")
    with s3:
        do_search = st.button("🔍 Search", use_container_width=True)

    if do_search and arxiv_query:
        with st.spinner("Searching arXiv…"):
            try:
                papers = search_arxiv(arxiv_query, n_results)
                st.session_state.arxiv_results = papers
            except Exception as e:
                st.error(f"arXiv search failed: {e}")

    if st.session_state.arxiv_results:
        papers = st.session_state.arxiv_results
        st.markdown(f"**{len(papers)} papers found**")
        for p in papers:
            cats = " · ".join(p.get("categories", [])[:3])
            authors = ", ".join(p.get("authors", []))
            st.markdown(f"""
            <div class='paper-card'>
              <div class='paper-title'>
                <a href="{p['url']}" target="_blank" style="color:#60a5fa;text-decoration:none">{p['title']}</a>
              </div>
              <div class='paper-meta'>👤 {authors} &nbsp;|&nbsp; 📅 {p['published']} &nbsp;|&nbsp; 🏷 {cats}</div>
              <div class='paper-summary'>{p['summary'][:500]}…</div>
              <div style='margin-top:8px'>
                <a href="{p['pdf_url']}" target="_blank" style='font-size:12px;color:#7c3aed'>📥 PDF</a>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 Ask Claude to synthesize these results", use_container_width=True):
            with st.spinner("Synthesizing…"):
                answer = run_agent("arxiv", arxiv_query, DOCS)
            st.session_state.messages.append({"role": "user", "content": f"Synthesize arXiv: '{arxiv_query}'"})
            st.session_state.messages.append({"role": "assistant", "content": answer, "mode": "arxiv"})
            st.info("✓ Answer added to Research Chat tab!")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#3a4060;font-size:12px;font-family:IBM Plex Mono,monospace;padding:1rem 0'>
  Research Paper Agent · Streamlit Cloud · Powered by Claude & arXiv
</div>
""", unsafe_allow_html=True)
