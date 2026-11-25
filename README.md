# Agentic RAG System

An agentic Retrieval-Augmented Generation (RAG) assistant that plans before answering, decides when and how to retrieve information, uses tools, and answers with cited sources from your own documents.

## Standard RAG vs Agentic RAG

| Standard RAG | Agentic RAG (this project) |
|---|---|
| One retrieval, one answer | Plans, then retrieves as many times as needed |
| Fixed query | Rewrites the query when results are weak |
| No tools | Search, summarize, data analysis, calculator |
| Struggles with multi-step questions | Breaks complex questions into steps |

## Features

- Knowledge base pipeline: text extraction, cleaning, chunking, embeddings, ChromaDB storage
- Planner step that breaks the question into sub-steps
- Agent loop that chooses tools, refines queries, and compares documents
- Tools: `search_knowledge_base`, `list_documents`, `summarize_document`, `analyze_data` (CSV/XLSX), `calculate`
- Answers with citations such as `(file.pdf, p.3)` and a Sources list
- Arabic and English support (multilingual embeddings)
- Gradio UI with an Agent trace panel showing the plan, tool calls, and retrieved evidence

Supported files: PDF, DOCX, TXT, MD, CSV, XLSX.

## Run on Google Colab (easiest)

1. Open `Agentic_RAG.ipynb` in Google Colab.
2. Get a free API key from [console.groq.com](https://console.groq.com).
3. In Colab, click the key icon in the sidebar, add a secret named `LLM_API_KEY`, and turn on Notebook access.
4. Run the cells in order and open the public Gradio link.

## Run locally

```bash
pip install -r requirements.txt
export LLM_API_KEY="your_key"      # Windows: set LLM_API_KEY=your_key
python app.py
```

Put your documents in `knowledge_base/` or upload them in the app, then click **Upload & rebuild index**.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | none | Key for any OpenAI-compatible provider |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Provider endpoint |
| `LLM_MODEL` | auto-detected | Model name (optional) |

## Tech stack

Python, Gradio, ChromaDB, sentence-transformers, OpenAI-compatible LLM API (Groq by default), pandas, pypdf, python-docx.
