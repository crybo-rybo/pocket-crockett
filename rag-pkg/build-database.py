#!/usr/bin/env python3
"""Build a local sqlite-vec RAG index from text/rag_chunks/*.jsonl."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "text" / "rag_chunks"
DB_PATH = Path(__file__).resolve().parent / "index" / "survive.sqlite"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(chunks_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "chunk_id" not in chunk or "text" not in chunk:
                    raise ValueError(f"Bad chunk in {path}:{line_no}")
                rows.append(chunk)
    return rows


def main() -> None:
    if not CHUNKS_DIR.is_dir():
        raise SystemExit(f"Missing chunks dir: {CHUNKS_DIR}")

    chunks = load_chunks(CHUNKS_DIR)
    if not chunks:
        raise SystemExit(f"No chunks found under {CHUNKS_DIR}")

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_DIR}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    dim = int(embeddings.shape[1])
    print(f"Embedded with {MODEL_NAME} → dim={dim}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute(
        """
        CREATE TABLE chunks (
          rowid INTEGER PRIMARY KEY,
          chunk_id TEXT NOT NULL UNIQUE,
          text TEXT NOT NULL,
          source_title TEXT,
          source_id TEXT,
          license TEXT,
          topic TEXT,
          page INTEGER,
          trust_tier TEXT,
          url TEXT,
          edibility_claim_review TEXT
        )
        """
    )
    db.execute(
        f"""
        CREATE VIRTUAL TABLE chunk_vec USING vec0(
          embedding float[{dim}]
        )
        """
    )
    db.execute(
        """
        CREATE TABLE index_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )
    db.executemany(
        "INSERT INTO index_meta(key, value) VALUES (?, ?)",
        [
            ("embedding_model", MODEL_NAME),
            ("embedding_dim", str(dim)),
            ("normalize_embeddings", "true"),
            ("chunk_count", str(len(chunks))),
            ("source", str(CHUNKS_DIR)),
        ],
    )

    for i, (chunk, vec) in enumerate(zip(chunks, embeddings), start=1):
        db.execute(
            """
            INSERT INTO chunks (
              rowid, chunk_id, text, source_title, source_id, license,
              topic, page, trust_tier, url, edibility_claim_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                i,
                chunk["chunk_id"],
                chunk["text"],
                chunk.get("source_title"),
                chunk.get("source_id"),
                chunk.get("license"),
                json.dumps(chunk.get("topic") or []),
                chunk.get("page"),
                chunk.get("trust_tier"),
                chunk.get("url"),
                chunk.get("edibility_claim_review"),
            ),
        )
        db.execute(
            "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, ?)",
            (i, sqlite_vec.serialize_float32(vec.tolist())),
        )

    db.commit()
    db.close()
    print(f"Wrote {len(chunks)} vectors → {DB_PATH}")


if __name__ == "__main__":
    main()
