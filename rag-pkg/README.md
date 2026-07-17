# rag-pkg — Pocket Crockett RAG MVP

Local retrieval-augmented chat over the Pipeline A corpus in `text/rag_chunks/`.

- **Embeddings:** `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- **Vector store:** `sqlite-vec` → `index/survive.sqlite`
- **LLM:** [Ollama](https://ollama.com) on localhost

## Setup

```bash
cd rag-pkg
uv sync
```

Install and start [Ollama](https://ollama.com), then pull a small instruct model:

```bash
ollama pull ministral-3:3b
# or: ollama pull qwen3:8b
```

## Build the index

```bash
uv run python build-database.py
```

Reads every `../text/rag_chunks/*.jsonl` chunk (~2.2k), embeds `text`, and writes `index/survive.sqlite` (gitignored).

## Ask a question

```bash
uv run python ask.py "How do I disinfect drinking water in an emergency?"
```

Useful flags:

```bash
# Retrieval only (debug ranking, no LLM)
uv run python ask.py --retrieve-only "How do I build a debris hut?"

# Different Ollama model
uv run python ask.py --model qwen3:8b "What are the signs of heat exhaustion?"

# Env overrides
OLLAMA_MODEL=qwen3:8b OLLAMA_HOST=http://127.0.0.1:11434 uv run python ask.py "..."
```

## Safety behavior (MVP)

- Prompt instructs the model to answer **only** from retrieved sources and cite `[1]`, `[2]`, …
- High-trust chunks are preferred when ranking (`trust_tier`)
- Low-trust / `edibility_claim_review` hits are labeled in context; they are **not** edibility authority

## Layout

```text
rag-pkg/
  build-database.py   # JSONL → sqlite-vec index
  ask.py              # retrieve → prompt → Ollama
  index/survive.sqlite  # generated, gitignored
  pyproject.toml
```
