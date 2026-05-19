# 🔬 Research Paper Agent — Streamlit Cloud

AI-powered research assistant. Upload PDFs, chat, summarize, compare, critique, and search arXiv live.

## Run Locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

---

## Files

```
research_agent/
├── app.py               ← Single-file app (all logic inside)
├── requirements.txt     ← 4 dependencies only
└── .streamlit/
    └── secrets.toml     ← API key (never commit this!)
```
