#!/usr/bin/env python3
"""Retrieve from the local RAG index and answer via Ollama."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).resolve().parent / "index" / "survive.sqlite"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ministral-3:3b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_K = 5


def open_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise SystemExit(
            f"Missing index at {path}. Run: uv run python build-database.py"
        )
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def embedding_model_name(db: sqlite3.Connection) -> str:
    row = db.execute(
        "SELECT value FROM index_meta WHERE key = 'embedding_model'"
    ).fetchone()
    return row["value"] if row else DEFAULT_MODEL


def retrieve(
    db: sqlite3.Connection,
    model: SentenceTransformer,
    question: str,
    k: int,
    prefer_high_trust: bool,
) -> list[dict]:
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    fetch_k = k * 4 if prefer_high_trust else k
    hits = db.execute(
        """
        SELECT
          c.chunk_id,
          c.text,
          c.source_title,
          c.source_id,
          c.trust_tier,
          c.url,
          c.page,
          c.edibility_claim_review,
          v.distance
        FROM chunk_vec AS v
        JOIN chunks AS c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_vec.tolist()), fetch_k),
    ).fetchall()
    rows = [dict(h) for h in hits]
    if prefer_high_trust:
        rows.sort(key=lambda r: (0 if r["trust_tier"] == "high" else 1, r["distance"]))
        rows = rows[:k]
    return rows


def build_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        cite = f"[{i}] {h['source_title']}"
        if h["page"] is not None:
            cite += f", p.{h['page']}"
        cite += f" ({h['url']})"
        notes: list[str] = []
        if h["trust_tier"] == "low":
            notes.append("LOW TRUST")
        if h.get("edibility_claim_review"):
            notes.append("NOT edibility authority — historical claim only")
        note = f" [{'; '.join(notes)}]" if notes else ""
        blocks.append(f"{cite}{note}\n{h['text']}")

    context = "\n\n---\n\n".join(blocks)
    system = (
        "You are Pocket Crockett, an offline survival assistant. "
        "Answer ONLY using the provided SOURCES. Cite them as [1], [2], etc. "
        "If the sources are insufficient, say you do not know. "
        "Prefer high-trust sources. Never treat low-trust or "
        "needs_human_review material as edibility or medical authority. "
        "Be calm, practical, and concise."
    )
    user = f"SOURCES:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    return system, user


def ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Ollama request failed ({base_url}). "
            f"Is Ollama running? Try: ollama serve\n{exc}"
        ) from exc
    message = body.get("message") or {}
    content = message.get("content")
    if not content:
        raise SystemExit(f"Unexpected Ollama response: {body!r}")
    return content.strip()


def print_sources(hits: list[dict]) -> None:
    print("\nSOURCES")
    print("-------")
    for i, h in enumerate(hits, start=1):
        page = f", p.{h['page']}" if h["page"] is not None else ""
        print(
            f"[{i}] dist={h['distance']:.4f}  {h['chunk_id']}  "
            f"[{h['trust_tier']}]  {h['source_title']}{page}"
        )
        print(f"    {h['url']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Pocket Crockett using local RAG + Ollama"
    )
    parser.add_argument("question", nargs="?", help="Survival / preparedness question")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="Top-k chunks (default 5)")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print retrieved chunks without calling the LLM",
    )
    parser.add_argument(
        "--no-prefer-high-trust",
        action="store_true",
        help="Disable high-trust preference when ranking hits",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        question = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
    if not question:
        parser.error("question required (arg or stdin)")

    db = open_db(DB_PATH)
    emb_name = embedding_model_name(db)
    print(f"Index: {DB_PATH}")
    print(f"Embedder: {emb_name}")
    print(f"Question: {question}")

    model = SentenceTransformer(emb_name)
    hits = retrieve(
        db,
        model,
        question,
        k=args.k,
        prefer_high_trust=not args.no_prefer_high_trust,
    )
    db.close()

    if not hits:
        raise SystemExit("No retrieval hits.")

    print_sources(hits)

    if args.retrieve_only:
        print("\n--- retrieve-only: skipping LLM ---")
        for i, h in enumerate(hits, start=1):
            print(f"\n[{i}] {h['chunk_id']}\n{h['text'][:500]}...")
        return

    system, user = build_prompt(question, hits)
    print(f"\nOllama model: {args.model}")
    answer = ollama_chat(
        base_url=args.ollama_url,
        model=args.model,
        system=system,
        user=user,
    )
    print("\nANSWER")
    print("------")
    print(answer)


if __name__ == "__main__":
    main()
