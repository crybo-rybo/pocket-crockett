#!/usr/bin/env python3
"""Build and remediate the Pocket Crockett Pipeline A text corpus.

This replaces the earlier one-off scripts under ``scripts/`` with a stable
CLI and shared helpers for download, cleaning, chunking, and reporting.

Typical use:

    python3 tools/text_pipeline.py baseline
    python3 tools/text_pipeline.py remediate-round2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "text"
RAW_DIR = TEXT_DIR / "raw"
CLEAN_DIR = TEXT_DIR / "clean"
CHUNK_DIR = TEXT_DIR / "rag_chunks"
REPORT_DIR = TEXT_DIR / "reports"

SOURCE_PLAN = TEXT_DIR / "source_plan.csv"
ROUND2_SOURCE_PLAN = TEXT_DIR / "source_plan_round2.csv"
MANIFEST = TEXT_DIR / "MANIFEST.csv"
COVERAGE_LOG = TEXT_DIR / "coverage_log.csv"
SKIPPED = TEXT_DIR / "skipped_sources.csv"

BASELINE_REPORT = REPORT_DIR / "coverage_provenance_report.md"
ROUND2_REPORT = REPORT_DIR / "coverage_provenance_report_round2.md"
SKEW_REPORT = REPORT_DIR / "corpus_skew_round2.csv"
TRUST_AUDIT = REPORT_DIR / "trust_tier_audit_round2.csv"
OCR_AUDIT = REPORT_DIR / "ocr_spotcheck_round2.csv"
FORAGING_REVIEW = REPORT_DIR / "foraging_claims_needing_human_review_round2.jsonl"

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
    "chunk_count",
]

TOPICS = [
    "shelter",
    "water",
    "fire",
    "navigation",
    "first_aid",
    "low_resource_medicine",
    "wound_care",
    "foraging",
    "edible_plants",
    "gardening",
    "farming",
    "animal_husbandry",
    "food_preservation",
    "pre_industrial_trades",
]

ROUND2_SOURCES = [
    {
        "source_id": "us_military_fm_4_25_11_first_aid",
        "source": "Departments of the Army, Navy, Air Force, and Marine Corps",
        "title": "FM 4-25.11 (FM 21-11) First Aid (Change 1)",
        "license": "public-domain-usgov",
        "trust_tier": "high",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/01/FM_4-25.11_%28FM_21-11%29_First_Aid_%28Change_1%29.pdf",
        "raw_filename": "us_military_fm_4_25_11_first_aid.pdf",
        "clean_filename": "us_military_fm_4_25_11_first_aid.txt",
        "topics": "first_aid;low_resource_medicine;wound_care",
        "source_type": "pdf",
        "license_evidence_url": "https://commons.wikimedia.org/wiki/File:FM_4-25.11_(FM_21-11)_First_Aid_(Change_1).pdf",
        "license_evidence": "Wikimedia Commons identifies this as a work of U.S. Army/Navy/Air Force/Marine Corps personnel, public domain as U.S. federal government work; document states approved for public release.",
        "notes": "Dedicated high-trust first-aid and low-resource emergency care source added in remediation round 2.",
        "date_acquired": "2026-05-25",
    },
    {
        "source_id": "usda_food_plants_north_american_indians",
        "source": "USDA Bureau of Chemistry and Soils",
        "title": "Food Plants of the North American Indians",
        "license": "public-domain-usgov",
        "trust_tier": "low",
        "url": "https://archive.org/download/foodplantsofnort237yano/foodplantsofnort237yano_djvu.txt",
        "raw_filename": "usda_food_plants_north_american_indians_ocr.txt",
        "clean_filename": "usda_food_plants_north_american_indians.txt",
        "topics": "foraging;ethnobotany;edible_plants",
        "source_type": "text",
        "license_evidence_url": "https://commons.wikimedia.org/wiki/File:Food_plants_of_the_North_American_Indians_(IA_foodplantsofnort237yano).pdf",
        "license_evidence": "Wikimedia Commons identifies this USDA Miscellaneous Publication as public domain under 17 USC 105.",
        "notes": "Historical USDA ethnobotanical compilation. All edibility/safety claims are flagged needs_human_review before use in the vision edibility table.",
        "date_acquired": "2026-05-25",
    },
    {
        "source_id": "appropedia_rainwater_harvesting_wayback",
        "source": "Appropedia via Internet Archive Wayback Machine",
        "title": "Rainwater Harvesting",
        "license": "CC-BY-SA-3.0",
        "trust_tier": "low",
        "url": "https://web.archive.org/web/20240225191326/https://www.appropedia.org/Rainwater_harvesting",
        "raw_filename": "appropedia_rainwater_harvesting_wayback.html",
        "clean_filename": "appropedia_rainwater_harvesting_wayback.txt",
        "topics": "water;rainwater_harvesting",
        "source_type": "html",
        "license_evidence_url": "https://web.archive.org/web/20240225191326/https://www.appropedia.org/Rainwater_harvesting",
        "license_evidence": "Archived page metadata includes license content=\"CC-BY-SA-3.0\".",
        "notes": "Recovered from Wayback after live Appropedia acquisition was blocked by Cloudflare in round 1.",
        "date_acquired": "2026-05-25",
    },
    {
        "source_id": "epa_emergency_disinfection_drinking_water",
        "source": "U.S. Environmental Protection Agency",
        "title": "Emergency Disinfection of Drinking Water",
        "license": "public-domain-usgov",
        "trust_tier": "high",
        "url": "https://www.epa.gov/ground-water-and-drinking-water/emergency-disinfection-drinking-water",
        "raw_filename": "epa_emergency_disinfection_drinking_water.html",
        "clean_filename": "epa_emergency_disinfection_drinking_water.txt",
        "topics": "water;purification;emergency_preparedness",
        "source_type": "html",
        "license_evidence_url": "https://www.epa.gov/ground-water-and-drinking-water/emergency-disinfection-drinking-water",
        "license_evidence": "Official .gov EPA page; U.S. federal government work.",
        "notes": "Dedicated high-trust water disinfection source added in remediation round 2.",
        "date_acquired": "2026-05-25",
    },
]
ROUND2_SOURCE_IDS = {source["source_id"] for source in ROUND2_SOURCES}

CLAIM_PATTERNS = re.compile(
    r"\b(edible|eaten|eat|food|foods|cooked|raw|boiled|dried|roasted|poison|poisonous|toxic|toxicity|medicinal|medicine|poultice|tea|root|berries|leaves|seeds|fruit)\b",
    re.I,
)


def ensure_dirs() -> None:
    for path in [RAW_DIR, CLEAN_DIR, CHUNK_DIR, REPORT_DIR, TEXT_DIR / "finetune"]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Pocket-Crockett-corpus-builder/0.3"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) < 1024:
        raise RuntimeError(f"Downloaded response for {url} was unexpectedly small ({len(data)} bytes)")
    dest.write_bytes(data)


def normalize_text(text: str) -> str:
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


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    end_match = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*", text, re.I | re.S)
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
    return text


def clean_wikitext(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}", " ", text)
    text = re.sub(r"\[\[File:[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[Image:[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return text.replace("'''", "").replace("''", "")


def extract_pdf(raw_path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(raw_path))
    pages = []
    extracted_chars = 0
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        extracted_chars += len(page_text.strip())
        pages.append(f"\n\n[page {idx}]\n\n{page_text}")
    chars_per_page = extracted_chars / len(reader.pages) if reader.pages else 0
    if chars_per_page < 100:
        quality = "unusable"
    elif chars_per_page < 500:
        quality = "noisy"
    else:
        quality = "clean"
    return normalize_text("\n".join(pages)), quality


def clean_html(raw_path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "nav", "footer", "header", "form"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find(id="mw-content-text")
        or soup.find(class_="mw-parser-output")
        or soup.find("article")
        or soup.body
        or soup
    )
    lines = []
    for element in main.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        text = element.get_text(" ", strip=True)
        if text and not text.startswith("Retrieved from"):
            lines.append(text)
    return normalize_text("\n\n".join(lines))


def clean_source(row: dict[str, str]) -> tuple[Path, str]:
    raw_path = RAW_DIR / row["raw_filename"]
    clean_path = CLEAN_DIR / row["clean_filename"]
    source_type = row["source_type"]
    if source_type == "pdf":
        text, quality = extract_pdf(raw_path)
    elif source_type == "html":
        text = clean_html(raw_path)
        quality = "clean" if len(text) >= 1000 else "noisy"
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


def iter_page_sections(text: str):
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


def split_paragraphs(text: str) -> list[tuple[int | None, str]]:
    paragraphs = []
    for page, body in iter_page_sections(text):
        for para in re.split(r"\n\s*\n", body):
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) >= 40:
                paragraphs.append((page, para))
    return paragraphs


def chunk_text(row: dict[str, str], clean_path: Path, quality: str) -> tuple[Path, int]:
    chunk_path = CHUNK_DIR / f"{row['source_id']}.jsonl"
    if quality == "unusable":
        chunk_path.write_text("", encoding="utf-8")
        return chunk_path, 0

    paragraphs = split_paragraphs(clean_path.read_text(encoding="utf-8", errors="replace"))
    chunks = []
    current: list[str] = []
    current_pages: list[int | None] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_pages, current_words
        if not current:
            return
        chunk_text_value = "\n\n".join(current).strip()
        if len(chunk_text_value) < 120:
            current = []
            current_pages = []
            current_words = 0
            return
        chunk = {
            "text": chunk_text_value,
            "source_title": row["title"],
            "source_id": row["source_id"],
            "license": row["license"],
            "topic": [t for t in row["topics"].split(";") if t],
            "page": min([p for p in current_pages if p is not None], default=None),
            "trust_tier": row["trust_tier"],
            "url": row["url"],
        }
        if row["source_id"] == "usda_food_plants_north_american_indians":
            chunk["edibility_claim_review"] = "needs_human_review"
        chunks.append(chunk)
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


def manifest_row_from_source(source: dict[str, str], clean_path: Path, chunk_path: Path, quality: str, chunk_count: int) -> dict[str, str | int]:
    return {
        "source_id": source["source_id"],
        "source": source["source"],
        "title": source["title"],
        "license": source["license"],
        "trust_tier": source["trust_tier"],
        "url": source["url"],
        "date_acquired": source["date_acquired"],
        "raw_path": str((RAW_DIR / source["raw_filename"]).relative_to(ROOT)),
        "clean_path": str(clean_path.relative_to(ROOT)),
        "rag_chunks_path": str(chunk_path.relative_to(ROOT)),
        "topics": source["topics"],
        "source_type": source["source_type"],
        "ocr_quality": quality,
        "status": "acquired" if quality != "unusable" else "acquired-unusable",
        "license_evidence_url": source["license_evidence_url"],
        "license_evidence": source["license_evidence"],
        "notes": source["notes"],
        "chunk_count": chunk_count,
    }


def summarize_coverage(manifest_rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    coverage = defaultdict(lambda: Counter())
    for row in manifest_rows:
        if row["status"] != "acquired":
            continue
        for topic in row["topics"].split(";"):
            if topic:
                coverage[topic][row["trust_tier"]] += 1
    rows = []
    for topic in TOPICS:
        high = coverage[topic]["high"]
        low = coverage[topic]["low"]
        rows.append(
            {
                "topic": topic,
                "high_sources": high,
                "low_sources": low,
                "total_sources": high + low,
                "status": "covered" if high > 0 else "gap: zero high-trust coverage",
            }
        )
    write_csv(COVERAGE_LOG, rows, ["topic", "high_sources", "low_sources", "total_sources", "status"])
    return rows


def write_baseline_report(manifest_rows: list[dict[str, str]], coverage_rows: list[dict]) -> None:
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
        f"- {r['source_category']} / {r['title']}: {r['status']} - {r['reason']}" for r in skipped
    )
    coverage_lines = "\n".join(
        f"- {r['topic']}: {r['high_sources']} high, {r['low_sources']} low - {r['status']}" for r in coverage_rows
    )
    license_lines = "\n".join(
        f"- `{r['source_id']}`: {r['license']} evidence: {r['license_evidence_url']}" for r in acquired
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
    BASELINE_REPORT.write_text(report, encoding="utf-8")


def write_foraging_review() -> int:
    chunk_path = CHUNK_DIR / "usda_food_plants_north_american_indians.jsonl"
    records = []
    if not chunk_path.exists():
        FORAGING_REVIEW.write_text("", encoding="utf-8")
        return 0
    for line in chunk_path.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        for idx, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()
            if len(sentence) >= 40 and CLAIM_PATTERNS.search(sentence):
                records.append(
                    {
                        "claim_id": f"{chunk['chunk_id']}:claim:{idx:03d}",
                        "chunk_id": chunk["chunk_id"],
                        "source_id": chunk["source_id"],
                        "source_title": chunk["source_title"],
                        "page": chunk.get("page"),
                        "claim_text": sentence,
                        "review_status": "needs_human_review",
                        "review_reason": "Historical foraging/ethnobotanical food-use claim; must be reconciled with the vision pipeline edibility table before use.",
                    }
                )
    with FORAGING_REVIEW.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def update_wikibooks_tier(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    audit_rows = []
    for row in manifest_rows:
        if row["source_id"] == "wikibooks_gardening" and row["trust_tier"] != "low":
            old = row["trust_tier"]
            row["trust_tier"] = "low"
            row["notes"] = (
                row.get("notes", "")
                + " Round 2 trust-tier audit: reclassified from high to low because this is a community-authored youth-honors page, not a government/medical/extension authority."
            ).strip()
            audit_rows.append(
                {
                    "source_id": row["source_id"],
                    "old_trust_tier": old,
                    "new_trust_tier": "low",
                    "reason": "Community-authored Wikibooks page should not share the same trust tier as USDA/Army/FEMA/EPA sources.",
                }
            )
            chunk_path = ROOT / row["rag_chunks_path"]
            updated = []
            for line in chunk_path.read_text(encoding="utf-8").splitlines():
                chunk = json.loads(line)
                chunk["trust_tier"] = "low"
                updated.append(json.dumps(chunk, ensure_ascii=False))
            chunk_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    if not audit_rows:
        audit_rows.append(
            {
                "source_id": "wikibooks_gardening",
                "old_trust_tier": "high",
                "new_trust_tier": "low",
                "reason": "Community-authored Wikibooks page should not share the same trust tier as USDA/Army/FEMA/EPA sources. Reclassification already applied.",
            }
        )
    write_csv(TRUST_AUDIT, audit_rows, ["source_id", "old_trust_tier", "new_trust_tier", "reason"])
    return audit_rows


def fm_21_76_ocr_audit(manifest_rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    row = next(r for r in manifest_rows if r["source_id"] == "us_army_fm_21_76")
    text = (ROOT / row["clean_path"]).read_text(encoding="utf-8", errors="replace")
    sample_checks = [
        ("page markers", text.count("[page ") >= 200),
        ("survival heading readable", "SURVIVAL" in text[:10000].upper()),
        ("plant section readable", "edible" in text.lower() and "poisonous" in text.lower()),
        ("low replacement character rate", text.count("\ufffd") == 0),
    ]
    word_count = len(text.split())
    long_garble_tokens = len(re.findall(r"\b[A-Za-z]{35,}\b", text))
    quality = "clean" if word_count > 50000 and long_garble_tokens < 250 and all(v for _, v in sample_checks) else "noisy"
    old = row["ocr_quality"]
    row["ocr_quality"] = quality
    audit = [
        {
            "source_id": row["source_id"],
            "old_ocr_quality": old,
            "new_ocr_quality": quality,
            "word_count": word_count,
            "long_garble_tokens": long_garble_tokens,
            "checks": "; ".join(f"{name}={value}" for name, value in sample_checks),
            "notes": "Spot-check found readable headings and plant/first-aid content with no replacement characters; retained clean flag."
            if quality == "clean"
            else "Spot-check found enough artifacts to reclassify as noisy.",
        }
    ]
    write_csv(OCR_AUDIT, audit, ["source_id", "old_ocr_quality", "new_ocr_quality", "word_count", "long_garble_tokens", "checks", "notes"])
    return audit


def write_skew_report(manifest_rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    counts = Counter()
    for row in manifest_rows:
        counts[row["trust_tier"]] += int(row.get("chunk_count") or 0)
    total = sum(counts.values())
    rows = [
        {
            "trust_tier": tier,
            "chunk_count": count,
            "percentage": f"{(count / total * 100):.2f}" if total else "0.00",
        }
        for tier, count in sorted(counts.items())
    ]
    write_csv(SKEW_REPORT, rows, ["trust_tier", "chunk_count", "percentage"])
    return rows


def write_round2_report(
    manifest_rows: list[dict[str, str]],
    added_rows: list[dict[str, str]],
    coverage_rows: list[dict],
    skew_rows: list[dict],
    trust_audit_rows: list[dict],
    ocr_audit_rows: list[dict],
    claim_count: int,
) -> None:
    baseline_sources = 13
    total_chunks = sum(int(r.get("chunk_count") or 0) for r in manifest_rows)
    added_lines = "\n".join(
        f"- `{r['source_id']}` - {r['title']} ({r['license']}, {r['trust_tier']}, {r['ocr_quality']}, {r['chunk_count']} chunks)"
        for r in added_rows
    )
    trust_lines = "\n".join(
        f"- `{r['source_id']}`: {r['old_trust_tier']} -> {r['new_trust_tier']} - {r['reason']}" for r in trust_audit_rows
    )
    ocr_lines = "\n".join(
        f"- `{r['source_id']}`: {r['old_ocr_quality']} -> {r['new_ocr_quality']}; {r['notes']}" for r in ocr_audit_rows
    )
    coverage_lines = "\n".join(
        f"- {r['topic']}: {r['high_sources']} high, {r['low_sources']} low - {r['status']}" for r in coverage_rows
    )
    skew_lines = "\n".join(f"- {r['trust_tier']}: {r['chunk_count']} chunks ({r['percentage']}%)" for r in skew_rows)
    license_lines = "\n".join(
        f"- `{r['source_id']}`: {r['license']} evidence: {r['license_evidence_url']}" for r in added_rows
    )

    report = f"""# Pocket Crockett Pipeline A Coverage & Provenance Report - Round 2 Remediation

Generated: 2026-05-25

## Baseline Diff

- Baseline report date: 2026-05-25
- Baseline acquired sources: {baseline_sources}
- Round 2 sources added: {len(added_rows)}
- Total acquired sources after round 2: {len([r for r in manifest_rows if r['status'] == 'acquired'])}
- Total RAG chunks after round 2: {total_chunks}
- Foraging claim records flagged `needs_human_review`: {claim_count}

## Sources Added

{added_lines}

## Trust-Tier Audit

{trust_lines}

## FM 21-76 OCR Spot-Check

{ocr_lines}

## Topic Coverage After Round 2

{coverage_lines}

## Chunk Volume By Trust Tier

{skew_lines}

Recommendation: keep low-trust historical/community material available for recall, but down-weight it during retrieval and cap per-source contribution for oversized low-trust books. In particular, cap or dampen `gutenberg_leather_manufacture`, `gutenberg_manual_gardening`, and the historical USDA foraging source so they cannot dominate answers over modern government and medical material.

## Foraging Safety Coordination

- New foraging source: `usda_food_plants_north_american_indians`.
- Every chunk from this source carries `edibility_claim_review: needs_human_review`.
- Extracted claim-level review records are in `{FORAGING_REVIEW.relative_to(ROOT)}`.
- These claims must not be treated as final edibility authority. They require reconciliation with the vision pipeline's species edibility table.

## License Evidence For New Sources

{license_lines}

## Recommended Human Follow-Ups

- Pursue written digital-use permission from Hesperian if the project still wants `Where There Is No Doctor` or related Hesperian texts.
- Review all round-2 foraging claims before any species-level edibility table import.
- Add retrieval-time weighting/capping for low-trust and oversized sources.
- Later pass: address the deferred high-trust gap for `pre_industrial_trades`.
"""
    ROUND2_REPORT.write_text(report, encoding="utf-8")


def run_baseline(_args: argparse.Namespace) -> int:
    ensure_dirs()
    plan_rows = read_csv(SOURCE_PLAN)
    manifest_rows = []
    for row in plan_rows:
        raw_path = RAW_DIR / row["raw_filename"]
        print(f"Downloading {row['source_id']}...", file=sys.stderr)
        download(row["url"], raw_path)
        clean_path, quality = clean_source(row)
        chunk_path, chunk_count = chunk_text(row, clean_path, quality)
        manifest_rows.append(manifest_row_from_source(row, clean_path, chunk_path, quality, chunk_count))

    write_csv(MANIFEST, manifest_rows, MANIFEST_FIELDS)
    coverage_rows = summarize_coverage(manifest_rows)
    write_baseline_report(manifest_rows, coverage_rows)
    print(f"Wrote {MANIFEST}", file=sys.stderr)
    print(f"Wrote {COVERAGE_LOG}", file=sys.stderr)
    print(f"Wrote {BASELINE_REPORT}", file=sys.stderr)
    return 0


def run_remediate_round2(_args: argparse.Namespace) -> int:
    ensure_dirs()
    manifest_rows = read_csv(MANIFEST)
    existing_ids = {row["source_id"] for row in manifest_rows}
    existing_hashes = {
        sha256(ROOT / row["raw_path"])
        for row in manifest_rows
        if row.get("raw_path") and (ROOT / row["raw_path"]).exists()
    }

    write_csv(ROUND2_SOURCE_PLAN, ROUND2_SOURCES, list(ROUND2_SOURCES[0].keys()))

    added_rows = []
    for source in ROUND2_SOURCES:
        if source["source_id"] in existing_ids:
            continue
        raw_path = RAW_DIR / source["raw_filename"]
        download(source["url"], raw_path)
        raw_hash = sha256(raw_path)
        if raw_hash in existing_hashes:
            raw_path.unlink(missing_ok=True)
            continue
        existing_hashes.add(raw_hash)
        clean_path, quality = clean_source(source)
        chunk_path, chunk_count = chunk_text(source, clean_path, quality)
        manifest_row = manifest_row_from_source(source, clean_path, chunk_path, quality, chunk_count)
        manifest_rows.append(manifest_row)
        added_rows.append(manifest_row)

    trust_audit_rows = update_wikibooks_tier(manifest_rows)
    ocr_audit_rows = fm_21_76_ocr_audit(manifest_rows)
    claim_count = write_foraging_review()
    write_csv(MANIFEST, manifest_rows, MANIFEST_FIELDS)
    coverage_rows = summarize_coverage(manifest_rows)
    skew_rows = write_skew_report(manifest_rows)
    round2_rows = [row for row in manifest_rows if row["source_id"] in ROUND2_SOURCE_IDS]
    write_round2_report(manifest_rows, round2_rows, coverage_rows, skew_rows, trust_audit_rows, ocr_audit_rows, claim_count)

    print(f"Added {len(added_rows)} sources")
    print(f"Flagged {claim_count} foraging claim records")
    print(f"Wrote {ROUND2_REPORT}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and remediate the Pipeline A text corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Rebuild Pipeline A from text/source_plan.csv.")
    baseline.set_defaults(func=run_baseline)

    round2 = subparsers.add_parser("remediate-round2", help="Apply the round-2 text-corpus remediation pass.")
    round2.set_defaults(func=run_remediate_round2)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
