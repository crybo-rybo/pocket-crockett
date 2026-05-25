#!/usr/bin/env python3
import csv
import json
import re
import shutil
import sys
import textwrap
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "text"
RAW_DIR = TEXT_DIR / "raw"
CLEAN_DIR = TEXT_DIR / "clean"
CHUNK_DIR = TEXT_DIR / "rag_chunks"
REPORT_DIR = TEXT_DIR / "reports"
SOURCE_PLAN = TEXT_DIR / "source_plan.csv"
MANIFEST = TEXT_DIR / "MANIFEST.csv"
COVERAGE_LOG = TEXT_DIR / "coverage_log.csv"
SKIPPED = TEXT_DIR / "skipped_sources.csv"
REPORT = REPORT_DIR / "coverage_provenance_report.md"

MANIFEST_FIELDS = [
    "source_id",
    "source",
    "title",
    "license",
    "trust_tier",
    "url",
    "date_acquired",
    "raw_path",
    "clean_path",
    "rag_chunks_path",
    "topics",
    "source_type",
    "ocr_quality",
    "status",
    "license_evidence_url",
    "license_evidence",
    "notes",
]


def ensure_dirs():
    for path in [RAW_DIR, CLEAN_DIR, CHUNK_DIR, REPORT_DIR, TEXT_DIR / "finetune"]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Pocket-Crockett-corpus-builder/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) < 1024:
        raise RuntimeError(f"Downloaded response for {url} was unexpectedly small ({len(data)} bytes)")
    dest.write_bytes(data)


def normalize_text(text):
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip() + "\n"


def strip_gutenberg_boilerplate(text):
    start_match = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    end_match = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*", text, re.I | re.S)
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
    return text


def clean_wikitext(text):
    text = re.sub(r"\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}", " ", text)
    text = re.sub(r"\[\[File:[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[Image:[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("'''", "").replace("''", "")
    return text


def extract_pdf(raw_path):
    reader = PdfReader(str(raw_path))
    pages = []
    extracted_chars = 0
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        extracted_chars += len(page_text.strip())
        pages.append(f"\n\n[page {idx}]\n\n{page_text}")
    if not reader.pages:
        quality = "unusable"
    else:
        chars_per_page = extracted_chars / len(reader.pages)
        if chars_per_page < 100:
            quality = "unusable"
        elif chars_per_page < 500:
            quality = "noisy"
        else:
            quality = "clean"
    return normalize_text("\n".join(pages)), quality


def clean_source(row):
    raw_path = RAW_DIR / row["raw_filename"]
    clean_path = CLEAN_DIR / row["clean_filename"]
    source_type = row["source_type"]
    if source_type == "pdf":
        text, quality = extract_pdf(raw_path)
    else:
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        if source_type == "text":
            text = strip_gutenberg_boilerplate(text)
        elif source_type == "wikitext":
            text = clean_wikitext(text)
        text = normalize_text(text)
        quality = "clean" if len(text) >= 1000 else "noisy"
    clean_path.write_text(text, encoding="utf-8")
    return clean_path, quality


def iter_page_sections(text):
    parts = re.split(r"\n\s*\[page (\d+)\]\s*\n", text)
    if len(parts) == 1:
        yield None, text
        return
    preface = parts[0].strip()
    if preface:
        yield None, preface
    for i in range(1, len(parts), 2):
        page = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        yield page, body


def split_paragraphs(text):
    paragraphs = []
    for page, body in iter_page_sections(text):
        for para in re.split(r"\n\s*\n", body):
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) >= 40:
                paragraphs.append((page, para))
    return paragraphs


def chunk_text(row, clean_path, quality):
    chunk_path = CHUNK_DIR / f"{row['source_id']}.jsonl"
    if quality == "unusable":
        chunk_path.write_text("", encoding="utf-8")
        return chunk_path, 0

    paragraphs = split_paragraphs(clean_path.read_text(encoding="utf-8", errors="replace"))
    chunks = []
    current = []
    current_pages = []
    current_words = 0

    def flush():
        nonlocal current, current_pages, current_words
        if not current:
            return
        chunk_text_value = "\n\n".join(current).strip()
        if len(chunk_text_value) < 120:
            current = []
            current_pages = []
            current_words = 0
            return
        chunks.append({
            "text": chunk_text_value,
            "source_title": row["title"],
            "source_id": row["source_id"],
            "license": row["license"],
            "topic": [t for t in row["topics"].split(";") if t],
            "page": min([p for p in current_pages if p is not None], default=None),
            "trust_tier": row["trust_tier"],
            "url": row["url"],
        })
        overlap = current[-1:]
        overlap_pages = current_pages[-1:]
        current = overlap.copy()
        current_pages = overlap_pages.copy()
        current_words = sum(len(p.split()) for p in current)

    for page, para in paragraphs:
        words = len(para.split())
        if current_words and current_words + words > 700:
            flush()
        current.append(para)
        current_pages.append(page)
        current_words += words
        if current_words >= 500:
            flush()
    if current_words:
        flush()

    with chunk_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = f"{row['source_id']}:{idx:05d}"
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunk_path, len(chunks)


def summarize_coverage(manifest_rows):
    coverage = defaultdict(lambda: Counter())
    for row in manifest_rows:
        if row["status"] != "acquired":
            continue
        for topic in row["topics"].split(";"):
            if topic:
                coverage[topic][row["trust_tier"]] += 1
    fields = ["topic", "high_sources", "low_sources", "total_sources", "status"]
    rows = []
    all_topics = [
        "shelter",
        "water",
        "fire",
        "navigation",
        "first_aid",
        "foraging",
        "gardening",
        "farming",
        "animal_husbandry",
        "food_preservation",
        "pre_industrial_trades",
    ]
    for topic in all_topics:
        high = coverage[topic]["high"]
        low = coverage[topic]["low"]
        rows.append({
            "topic": topic,
            "high_sources": high,
            "low_sources": low,
            "total_sources": high + low,
            "status": "covered" if high > 0 else "gap: zero high-trust coverage",
        })
    write_csv(COVERAGE_LOG, rows, fields)
    return rows


def write_report(manifest_rows, coverage_rows):
    acquired = [r for r in manifest_rows if r["status"] == "acquired"]
    skipped = read_csv(SKIPPED) if SKIPPED.exists() else []
    trust_counts = Counter(r["trust_tier"] for r in acquired)
    ocr_counts = Counter(r["ocr_quality"] for r in acquired)
    total_chunks = sum(int(r.get("chunk_count", 0)) for r in acquired)
    gaps = [r for r in coverage_rows if r["status"].startswith("gap")]

    acquired_lines = "\n".join(
        f"- `{r['source_id']}` - {r['title']} ({r['license']}, {r['trust_tier']}, {r['ocr_quality']}, {r.get('chunk_count', 0)} chunks)"
        for r in acquired
    )
    skipped_lines = "\n".join(
        f"- {r['source_category']} / {r['title']}: {r['status']} - {r['reason']}"
        for r in skipped
    )
    coverage_lines = "\n".join(
        f"- {r['topic']}: {r['high_sources']} high, {r['low_sources']} low - {r['status']}"
        for r in coverage_rows
    )
    license_lines = "\n".join(
        f"- `{r['source_id']}`: {r['license']} evidence: {r['license_evidence_url']}"
        for r in acquired
    )
    gap_text = "None" if not gaps else "\n".join(f"- {r['topic']}" for r in gaps)

    report = f"""# Pocket Crockett Pipeline A Coverage & Provenance Report

Generated: 2026-05-25

## Summary

- Acquired sources: {len(acquired)}
- Skipped source categories/items: {len(skipped)}
- Total RAG chunks: {total_chunks}
- Trust tiers: {dict(trust_counts)}
- OCR/text quality: {dict(ocr_counts)}
- Topics with zero high-trust coverage: {len(gaps)}

## Acquired Sources

{acquired_lines}

## Skipped Sources

{skipped_lines}

## Topic Coverage

{coverage_lines}

## Zero High-Trust Coverage Gaps

{gap_text}

## License Evidence

{license_lines}

## Notes

- Hesperian was not imported because its policy requires written permission for digital use.
- FAO was not imported in this pass because no item-specific approved license was selected.
- Survivor Library was treated as an index; initial corpus items were acquired from primary or item-level public-domain hosts instead.
- No source with unknown or unverifiable license was included in the manifest as acquired.
- Nothing flagged `unusable` was included in RAG chunks.
"""
    REPORT.write_text(report, encoding="utf-8")


def main():
    ensure_dirs()
    plan_rows = read_csv(SOURCE_PLAN)
    manifest_rows = []
    for row in plan_rows:
        raw_path = RAW_DIR / row["raw_filename"]
        print(f"Downloading {row['source_id']}...", file=sys.stderr)
        download(row["url"], raw_path)
        clean_path, quality = clean_source(row)
        chunk_path, chunk_count = chunk_text(row, clean_path, quality)
        manifest_row = {
            "source_id": row["source_id"],
            "source": row["source"],
            "title": row["title"],
            "license": row["license"],
            "trust_tier": row["trust_tier"],
            "url": row["url"],
            "date_acquired": row["date_acquired"],
            "raw_path": str(raw_path.relative_to(ROOT)),
            "clean_path": str(clean_path.relative_to(ROOT)),
            "rag_chunks_path": str(chunk_path.relative_to(ROOT)),
            "topics": row["topics"],
            "source_type": row["source_type"],
            "ocr_quality": quality,
            "status": "acquired" if quality != "unusable" else "acquired-unusable",
            "license_evidence_url": row["license_evidence_url"],
            "license_evidence": row["license_evidence"],
            "notes": row["notes"],
            "chunk_count": chunk_count,
        }
        manifest_rows.append(manifest_row)

    write_csv(MANIFEST, manifest_rows, MANIFEST_FIELDS + ["chunk_count"])
    coverage_rows = summarize_coverage(manifest_rows)
    write_report(manifest_rows, coverage_rows)
    print(f"Wrote {MANIFEST}", file=sys.stderr)
    print(f"Wrote {COVERAGE_LOG}", file=sys.stderr)
    print(f"Wrote {REPORT}", file=sys.stderr)


if __name__ == "__main__":
    main()
