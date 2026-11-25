"""
Agentic RAG System
Pipeline : extract -> clean -> chunk -> embed -> store (ChromaDB)
Agent    : plan -> (search / summarize / analyze / calculate)* -> answer with sources
UI       : Gradio

Setup:
    pip install -r requirements.txt
    export LLM_API_KEY="your_key"
    python app.py
"""
import ast
import json
import math
import operator
import os
import re
from pathlib import Path

import chromadb
import gradio as gr
import pandas as pd
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

KB_DIR = Path("knowledge_base")
DB_DIR = "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Arabic + English
CHUNK_WORDS, OVERLAP_WORDS = 180, 40
MAX_STEPS = 8

llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "missing"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
)


def pick_model():
    """Use LLM_MODEL if set and available, otherwise pick a working model from the provider."""
    wanted = os.getenv("LLM_MODEL")
    try:
        ids = [m.id for m in llm.models.list().data]
    except Exception:
        return wanted or "openai/gpt-oss-120b"
    if wanted in ids:
        return wanted
    for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile",
              "meta-llama/llama-4-maverick-17b-128e-instruct",
              "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"]:
        if m in ids:
            return m
    skip = ("whisper", "guard", "tts", "orpheus", "compound", "safeguard", "embed")
    ok = [i for i in ids if not any(s in i for s in skip)]
    return (ok or ids)[0]


LLM_MODEL = pick_model()
print("Using model:", LLM_MODEL)

KB_DIR.mkdir(exist_ok=True)
embedder = SentenceTransformer(EMBED_MODEL)
chroma = chromadb.PersistentClient(path=DB_DIR)
def get_collection():
    try:
        return chroma.get_or_create_collection("knowledge_base", configuration={"hnsw": {"space": "cosine"}})
    except Exception:
        return chroma.get_or_create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})


collection = get_collection()


def extract(path: Path):
    """Return list of (page_number, text)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
    if ext == ".docx":
        return [(1, "\n".join(p.text for p in Document(str(path)).paragraphs))]
    if ext in (".txt", ".md"):
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    if ext in (".csv", ".xlsx"):  # index a schema description so the agent can find tables
        df = load_table(path)
        desc = (f"Table file {path.name}. Columns: {', '.join(map(str, df.columns))}. "
                f"{len(df)} rows. Sample rows:\n{df.head(5).to_string()}")
        return [(1, desc)]
    return []


def clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)          # fix hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece.split()) >= 15 or (i == 0 and piece):
            yield piece
        if i + CHUNK_WORDS >= len(words):
            break


def build_index():
    """(Re)build the knowledge base from every file in KB_DIR."""
    global collection
    chroma.delete_collection("knowledge_base")
    collection = get_collection()
    ids, docs, metas = [], [], []
    for f in sorted(KB_DIR.iterdir()):
        for page, raw in extract(f):
            for j, c in enumerate(chunk(clean(raw))):
                ids.append(f"{f.name}-p{page}-c{j}")
                docs.append(c)
                metas.append({"source": f.name, "page": page})
    if not docs:
        return "No documents found in knowledge_base/."
    embs = embedder.encode(docs, batch_size=64, normalize_embeddings=True).tolist()
    for i in range(0, len(ids), 500):
        collection.add(ids=ids[i:i + 500], documents=docs[i:i + 500],
                       embeddings=embs[i:i + 500], metadatas=metas[i:i + 500])
    return f"Indexed {len(docs)} chunks from {len({m['source'] for m in metas})} files."


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def search_knowledge_base(query: str, top_k: int = 5, source: str = ""):
    if collection.count() == 0:
        return "Knowledge base is empty."
    q = embedder.encode([query], normalize_embeddings=True).tolist()
    where = {"source": source} if source else None
    res = collection.query(query_embeddings=q, n_results=min(int(top_k), 10), where=where)
    out = []
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
        out.append(f"[{i}] ({meta['source']}, p.{meta['page']}) relevance={1 - dist:.2f}\n{doc}")
    return "\n\n".join(out) or "No results."


def list_documents():
    names = sorted({m["source"] for m in collection.get()["metadatas"]})
    return "\n".join(names) or "Knowledge base is empty."


def summarize_document(source: str, focus: str = ""):
    got = collection.get(where={"source": source})
    if not got["documents"]:
        return f"Document '{source}' not found. Use list_documents."
    chunks = got["documents"]
    stride = max(1, len(chunks) // 20)                 # sample evenly, max ~20 chunks
    text = "\n".join(chunks[::stride])[:14000]
    r = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.1,
        messages=[{"role": "user", "content":
                   f"Summarize the following document in a concise structured way. "
                   f"{'Focus on: ' + focus if focus else ''}\n\n{text}"}])
    return r.choices[0].message.content


def analyze_data(file: str, operation: str, column: str = "", group_by: str = "", agg: str = "mean"):
    path = KB_DIR / file
    if not path.exists() or path.suffix.lower() not in (".csv", ".xlsx"):
        return f"'{file}' is not a CSV/XLSX file in the knowledge base."
    df = load_table(path)
    try:
        if operation == "columns":
            return df.dtypes.to_string()
        if operation == "describe":
            return df.describe(include="all").to_string()
        if operation == "head":
            return df.head(10).to_string()
        if operation == "value_counts":
            return df[column].value_counts().head(20).to_string()
        if operation == "aggregate":
            if group_by:
                return df.groupby(group_by)[column].agg(agg).sort_values(ascending=False).head(20).to_string()
            return str(df[column].agg(agg))
        return "Unknown operation. Use: columns, describe, head, value_counts, aggregate."
    except Exception as e:
        return f"Analysis error: {e}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "abs": abs, "round": round}


def _eval(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
        return _FUNCS[n.func.id](*[_eval(a) for a in n.args])
    raise ValueError("Unsupported expression")


def calculate(expression: str):
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculation error: {e}"


TOOL_FUNCS = {
    "search_knowledge_base": search_knowledge_base,
    "list_documents": list_documents,
    "summarize_document": summarize_document,
    "analyze_data": analyze_data,
    "calculate": calculate,
}


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S = {"type": "string"}
TOOLS = [
    _tool("search_knowledge_base",
          "Semantic search over the documents. Returns chunks with source, page and relevance. "
          "Optionally restrict to one file with `source`.",
          {"query": S, "top_k": {"type": "integer"}, "source": S}, ["query"]),
    _tool("list_documents", "List all files in the knowledge base.", {}),
    _tool("summarize_document", "Summarize one whole document, optionally with a focus.",
          {"source": S, "focus": S}, ["source"]),
    _tool("analyze_data",
          "Run analysis on a CSV/XLSX file. operation: columns | describe | head | value_counts | aggregate. "
          "For aggregate give column, optional group_by, and agg (mean/sum/min/max/count/median).",
          {"file": S, "operation": S, "column": S, "group_by": S, "agg": S}, ["file", "operation"]),
    _tool("calculate", "Evaluate an arithmetic expression exactly (+ - * / ** % sqrt log abs round).",
          {"expression": S}, ["expression"]),
]


PLANNER_PROMPT = (
    "You are the planning step of a RAG agent. Break the user's question into a short numbered plan "
    "(max 5 steps). For each step say which tool fits: search_knowledge_base, summarize_document, "
    "analyze_data, calculate. Be brief. Output only the plan."
)

AGENT_PROMPT = (
    "You are a knowledge-base assistant. Follow the plan, using tools as needed.\n"
    "Rules:\n"
    "- Ground every factual claim in tool results. Never invent facts.\n"
    "- If search results look weak (relevance < 0.35) or incomplete, rewrite the query "
    "(synonyms, narrower or broader wording) and search again.\n"
    "- For comparisons, search each item separately (use the `source` filter when useful).\n"
    "- Use `calculate` for any arithmetic; do not compute in your head.\n"
    "- Final answer: clear, in the user's language, with citations like (file.pdf, p.3) and a "
    "'Sources' list at the end. If the knowledge base does not contain the answer, say so."
)


def chat(messages, **kw):
    return llm.chat.completions.create(model=LLM_MODEL, temperature=0.1, messages=messages, **kw)


def run_agent(question: str):
    trace = []
    plan = chat([{"role": "system", "content": PLANNER_PROMPT},
                 {"role": "user", "content": question}]).choices[0].message.content
    trace.append(f"**Plan**\n\n{plan}")

    messages = [{"role": "system", "content": AGENT_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPlan:\n{plan}"}]
    for step in range(1, MAX_STEPS + 1):
        msg = chat(messages, tools=TOOLS, tool_choice="auto").choices[0].message
        if not msg.tool_calls:
            return msg.content, trace
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": t.id, "type": "function",
                                         "function": {"name": t.function.name,
                                                      "arguments": t.function.arguments}}
                                        for t in msg.tool_calls]})
        for t in msg.tool_calls:
            try:
                args = json.loads(t.function.arguments or "{}")
                result = TOOL_FUNCS[t.function.name](**args)
            except Exception as e:
                args, result = {}, f"Tool error: {e}"
            trace.append(f"**Step {step}: `{t.function.name}`** `{json.dumps(args, ensure_ascii=False)}`\n\n"
                         f"```\n{str(result)[:1200]}\n```")
            messages.append({"role": "tool", "tool_call_id": t.id, "content": str(result)[:6000]})

    messages.append({"role": "user", "content": "Step limit reached. Give the best final answer now."})
    return chat(messages).choices[0].message.content, trace


def ask(question):
    if not question.strip():
        return "", ""
    try:
        answer, trace = run_agent(question)
    except Exception as e:
        return f"Error: {e}", ""
    return answer, "\n\n---\n\n".join(trace)


def upload_and_index(files):
    for f in files or []:
        dest = KB_DIR / Path(f).name
        dest.write_bytes(Path(f).read_bytes())
    return build_index()


with gr.Blocks(title="Agentic RAG") as demo:
    gr.Markdown("# Agentic RAG Assistant\nAsk complex, multi-step questions about your documents.")
    with gr.Row():
        files = gr.File(label="Add documents (PDF, DOCX, TXT, MD, CSV, XLSX)",
                        file_count="multiple", type="filepath")
        status = gr.Textbox(label="Index status", interactive=False)
    with gr.Row():
        gr.Button("Upload & rebuild index").click(upload_and_index, files, status)
        gr.Button("Rebuild from folder").click(lambda: build_index(), None, status)
    q = gr.Textbox(label="Question", lines=2)
    go = gr.Button("Ask", variant="primary")
    ans = gr.Markdown()
    with gr.Accordion("Agent trace (plan, tool calls, retrieved evidence)", open=False):
        tr = gr.Markdown()
    go.click(ask, q, [ans, tr])
    q.submit(ask, q, [ans, tr])

if __name__ == "__main__":
    if collection.count() == 0 and any(KB_DIR.iterdir()):
        print(build_index())
    demo.launch()
