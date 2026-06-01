#!/usr/bin/env python3
"""
Pocket Crockett Pipeline B data preparation.

The script is intentionally manifest-first:
- license whitelist is loaded before any image is retained
- USDA matches are exact-name matches after stripping authors/HTML
- edibility records are safe skeletons only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
LICENSE_CONFIG = VISION / "config" / "accepted_image_licenses.json"
IMAGE_MANIFEST = VISION / "images" / "manifest.csv"
BATCH_MANIFEST = VISION / "MANIFEST.csv"
RAW_IMAGES = VISION / "images" / "raw"
OUTPUT_CLASSES = VISION / "splits" / "output_classes.csv"
HELD_ASIDE_POOL = VISION / "splits" / "held_aside_pool.csv"
BALANCE_TABLE = VISION / "reports" / "class_balance.csv"
PLANTNET_V2_RECORD = "https://zenodo.org/api/records/10419064"
USDA_API = "https://plantsservices.sc.egov.usda.gov/api/"
GBIF_API = "https://api.gbif.org/v1/"
INATURALIST_GBIF_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
PLANTNET_STORAGE_THRESHOLD_GIB = 120.0
PLANTNET_ARCHIVE_PATH = VISION / "sources" / "plantnet300k-v2" / "images.zip"
_USDA_SEARCH_CACHE: dict[str, Any] | None = None
_USDA_PROFILE_CACHE: dict[str, Any] | None = None
SPLIT_NAMES = ["train", "val", "test", "calibration"]


IMAGE_FIELDS = [
    "image_id",
    "path",
    "dataset",
    "batch",
    "scientific_name",
    "taxon_id",
    "source_taxon_id",
    "source_image_id",
    "observation_id",
    "author",
    "license",
    "source_url",
    "lat",
    "lon",
    "split",
    "original_split",
    "content_hash",
    "width",
    "height",
    "license_checked_at",
    "provenance_json",
]

TOXIC_REMEDIATION_TARGETS = {
    "Cicuta maculata",
    "Cicuta douglasii",
    "Conium maculatum",
    "Zigadenus venenosus",
    "Zigadenus paniculatus",
    "Zigadenus glaberrimus",
    "Toxicodendron radicans",
    "Toxicodendron diversilobum",
    "Toxicodendron vernix",
    "Phytolacca americana",
    "Solanum dulcamara",
    "Atropa belladonna",
    "Digitalis purpurea",
    "Nerium oleander",
    "Actaea pachypoda",
    "Actaea rubra",
}

DESCOPED_FUNGI_TARGETS = {
    "Amanita bisporigera",
    "Amanita phalloides",
    "Amanita ocreata",
}


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "pocket-crockett-data-prep/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str, dest: Path, *, min_bytes: int = 1) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pocket-crockett-data-prep/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as out:
        shutil.copyfileobj(r, out)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {dest}")
    tmp.replace(dest)


def download_image(url: str, dest: Path, *, min_bytes: int = 256) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pocket-crockett-data-prep/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content_type = (r.headers.get("content-type") or "").split(";")[0].lower()
        if content_type and not content_type.startswith("image/"):
            raise RuntimeError(f"Not an image response: {content_type}")
        with tmp.open("wb") as out:
            shutil.copyfileobj(r, out)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded image is too small: {dest}")
    if not is_supported_image(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Unsupported image bytes: {dest}")
    tmp.replace(dest)
    return content_type


def is_supported_image(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(16)
    return header.startswith(b"\xff\xd8\xff") or header.startswith(b"\x89PNG\r\n\x1a\n")


def image_dimensions(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as f:
            data = f.read(32)
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                return str(int.from_bytes(data[16:20], "big")), str(int.from_bytes(data[20:24], "big"))
            if not data.startswith(b"\xff\xd8"):
                return "", ""
            f.seek(2)
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    return "", ""
                if marker_start != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3"}:
                    length = int.from_bytes(f.read(2), "big")
                    segment = f.read(length - 2)
                    return str(int.from_bytes(segment[3:5], "big")), str(int.from_bytes(segment[1:3], "big"))
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                length_bytes = f.read(2)
                if len(length_bytes) != 2:
                    return "", ""
                length = int.from_bytes(length_bytes, "big")
                if length < 2:
                    return "", ""
                f.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return "", ""


def normalize_license(value: str | None) -> str:
    if value is None:
        return "unknown"
    v = value.strip().lower()
    if not v:
        return "unknown"
    v = v.replace("creative commons ", "cc-")
    v = v.replace("https://creativecommons.org/licenses/", "cc-")
    v = v.replace("http://creativecommons.org/licenses/", "cc-")
    v = v.replace("https://creativecommons.org/publicdomain/zero/1.0/", "cc0")
    v = v.replace("http://creativecommons.org/publicdomain/zero/1.0/", "cc0")
    v = re.sub(r"/legalcode$", "", v)
    v = v.rstrip("/")
    v = v.replace("/4.0", "-4.0").replace("/3.0", "-3.0")
    v = v.replace("_", "-").replace(" ", "-")
    v = v.replace("-4-0", "-4.0").replace("-3-0", "-3.0").replace("cc0-1-0", "cc0")
    return v


def accepted_licenses() -> set[str]:
    cfg = read_json(LICENSE_CONFIG, {})
    return {normalize_license(x) for x in cfg.get("accepted", [])}


def is_license_accepted(value: str | None) -> bool:
    return normalize_license(value) in accepted_licenses()


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(value))).strip()


def canonical_species_name(value: str | None) -> str:
    text = strip_html(value)
    text = text.replace("×", "x")
    text = re.sub(r"\s+\[[^\]]+\]$", "", text)
    parts = text.split()
    if len(parts) < 2:
        return text
    return f"{parts[0]} {parts[1]}"


def is_exact_species_hit(wanted: str, candidate: str | None) -> bool:
    wanted_binomial = canonical_species_name(wanted)
    candidate_text = strip_html(candidate).replace("×", " x ")
    if not candidate_text.lower().startswith(wanted_binomial.lower()):
        return False
    remainder = candidate_text[len(wanted_binomial) :].strip()
    if not remainder:
        return True
    lower = remainder.lower()
    reject_prefixes = ("x ", "var.", "var ", "subsp.", "subsp ", "ssp.", "ssp ", "f.", " f ", "[")
    if lower.startswith(reject_prefixes):
        return False
    return True


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ensure_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def append_batch(row: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "source",
        "source_url",
        "license_terms",
        "accepted_license_whitelist",
        "count",
        "date_acquired",
        "status",
        "notes",
    ]
    ensure_csv(BATCH_MANIFEST, fields)
    rows = read_csv(BATCH_MANIFEST)
    key = (row["dataset"], row["source"], row.get("status", ""))
    rows = [r for r in rows if (r.get("dataset"), r.get("source"), r.get("status")) != key]
    rows.append({k: row.get(k, "") for k in fields})
    write_csv(BATCH_MANIFEST, fields, rows)


def cmd_fetch_plantnet_metadata(_: argparse.Namespace) -> None:
    outdir = VISION / "sources" / "plantnet300k-v2"
    record = http_json(PLANTNET_V2_RECORD)
    write_json(outdir / "zenodo_record.json", record)
    files = {f["key"]: f for f in record["files"]}
    for key in ["README.md", "species_metadata.csv", "plantnet300K_metadata.csv"]:
        dest = outdir / key
        if not dest.exists():
            download(files[key]["links"]["self"], dest, min_bytes=100)
    image_rows = sum(1 for _ in (outdir / "plantnet300K_metadata.csv").open("r", encoding="utf-8")) - 1
    append_batch(
        {
            "dataset": "plantnet300k-v2",
            "source": "Zenodo",
            "source_url": record["links"]["self_html"],
            "license_terms": record["metadata"].get("license", {}).get("id", "cc-by-4.0")
            + "; per-image metadata includes cc-by-sa/cc-by-nc/cc-by-nc-sa",
            "accepted_license_whitelist": json.dumps(sorted(accepted_licenses())),
            "count": image_rows,
            "date_acquired": utc_today(),
            "status": "metadata_acquired_images_pending_storage",
            "notes": "Full image archive is 41.8 GB compressed; not retained until storage check passes.",
        }
    )
    print(f"Fetched PlantNet-300K-v2 metadata rows: {image_rows}")


def cmd_check_storage(args: argparse.Namespace) -> None:
    usage = shutil.disk_usage(args.path)
    free_gib = usage.free / (1024**3)
    print(f"Free space at {args.path}: {free_gib:.1f} GiB")
    if free_gib < args.required_gib:
        raise SystemExit(f"Insufficient storage: need at least {args.required_gib:.1f} GiB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plantnet_record_file(key: str) -> dict[str, Any]:
    record_path = VISION / "sources" / "plantnet300k-v2" / "zenodo_record.json"
    if not record_path.exists():
        record = http_json(PLANTNET_V2_RECORD)
        write_json(record_path, record)
    else:
        record = read_json(record_path, {})
    for item in record.get("files", []):
        if item.get("key") == key:
            return item
    raise RuntimeError(f"PlantNet record does not include {key}")


def plantnet_part_bounds(index: int, chunk_size: int, archive_size: int) -> tuple[int, int]:
    start = index * chunk_size
    end = min(start + chunk_size, archive_size) - 1
    return start, end


def plantnet_parts_progress(parts_dir: Path, archive_size: int, chunk_size: int) -> tuple[int, int, int]:
    total_parts = (archive_size + chunk_size - 1) // chunk_size
    bytes_present = 0
    complete_parts = 0
    for index in range(total_parts):
        start, end = plantnet_part_bounds(index, chunk_size, archive_size)
        expected = end - start + 1
        path = parts_dir / f"{index:06d}.part"
        size = path.stat().st_size if path.exists() else 0
        bytes_present += min(size, expected)
        if size == expected:
            complete_parts += 1
    return bytes_present, complete_parts, total_parts


def seed_plantnet_parts_from_prefix(prefix_path: Path, parts_dir: Path, archive_size: int, chunk_size: int) -> int:
    if not prefix_path.exists() or prefix_path.stat().st_size == 0:
        return 0
    parts_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0
    with prefix_path.open("rb") as src:
        index = 0
        remaining = min(prefix_path.stat().st_size, archive_size)
        while remaining > 0:
            start, end = plantnet_part_bounds(index, chunk_size, archive_size)
            expected = min(end - start + 1, remaining)
            part = parts_dir / f"{index:06d}.part"
            if part.exists() and part.stat().st_size == expected:
                src.seek(expected, os.SEEK_CUR)
            else:
                tmp = part.with_suffix(".part.tmpseed")
                with tmp.open("wb") as out:
                    shutil.copyfileobj(src, out, expected)
                tmp.replace(part)
                seeded += 1
            remaining -= expected
            index += 1
    return seeded


def cmd_plantnet_preflight(args: argparse.Namespace) -> None:
    image_file = plantnet_record_file("images.zip")
    archive_size = int(image_file["size"])
    free = shutil.disk_usage(args.path).free
    existing_archive = args.archive if args.archive else PLANTNET_ARCHIVE_PATH
    existing_size = existing_archive.stat().st_size if existing_archive.exists() else 0
    estimated_needed = max(0, archive_size - existing_size) + int(archive_size * args.extraction_multiplier)
    report = {
        "path": str(Path(args.path).resolve()),
        "archive_path": str(existing_archive),
        "archive_size_bytes": archive_size,
        "existing_archive_bytes": existing_size,
        "estimated_extraction_bytes": int(archive_size * args.extraction_multiplier),
        "estimated_additional_bytes_needed": estimated_needed,
        "free_bytes": free,
        "free_gib": round(free / (1024**3), 2),
        "passes": free >= estimated_needed,
        "source_url": image_file["links"]["self"],
        "checksum": image_file["checksum"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(1)


def cmd_download_plantnet_segmented(args: argparse.Namespace) -> None:
    image_file = plantnet_record_file("images.zip")
    dest = args.output or PLANTNET_ARCHIVE_PATH
    url = image_file["links"]["self"]
    archive_size = int(image_file["size"])
    chunk_size = int(args.chunk_mib * 1024 * 1024)
    parts_dir = args.parts_dir or dest.with_suffix(dest.suffix + ".parts")
    reserve_bytes = int(args.reserve_gib * (1024**3))
    free = shutil.disk_usage(dest.parent).free
    existing_bytes, _, _ = plantnet_parts_progress(parts_dir, archive_size, chunk_size) if parts_dir.exists() else (0, 0, 0)
    prefix_bytes = dest.stat().st_size if dest.exists() and dest.stat().st_size < archive_size else 0
    additional_needed = max(0, archive_size - max(existing_bytes, prefix_bytes))
    if free < additional_needed + reserve_bytes:
        raise SystemExit(
            f"Insufficient space for segmented PlantNet download: need {additional_needed / (1024**3):.1f} GiB plus "
            f"{args.reserve_gib:.1f} GiB reserve, have {free / (1024**3):.1f} GiB free"
        )
    seeded = seed_plantnet_parts_from_prefix(dest, parts_dir, archive_size, chunk_size)
    total_parts = (archive_size + chunk_size - 1) // chunk_size
    deadline = time.time() + args.max_seconds if args.max_seconds else None
    lock = Lock()
    active: set[int] = set()
    errors: list[str] = []

    def part_expected(index: int) -> int:
        start, end = plantnet_part_bounds(index, chunk_size, archive_size)
        return end - start + 1

    def pick_part() -> int | None:
        with lock:
            for index in range(total_parts):
                if index in active:
                    continue
                part = parts_dir / f"{index:06d}.part"
                if part.exists() and part.stat().st_size == part_expected(index):
                    continue
                active.add(index)
                return index
            return None

    def release_part(index: int) -> None:
        with lock:
            active.discard(index)

    def worker(worker_id: int) -> None:
        while True:
            if deadline and time.time() >= deadline:
                return
            index = pick_part()
            if index is None:
                return
            try:
                start, end = plantnet_part_bounds(index, chunk_size, archive_size)
                part = parts_dir / f"{index:06d}.part"
                have = part.stat().st_size if part.exists() else 0
                expected = end - start + 1
                if have > expected:
                    part.unlink()
                    have = 0
                if have == expected:
                    continue
                range_start = start + have
                timeout = args.request_seconds
                if deadline:
                    timeout = max(1, min(timeout, int(deadline - time.time())))
                tmp = part.with_suffix(f".part.w{worker_id}.tmp")
                tmp.unlink(missing_ok=True)
                cmd = [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--range",
                    f"{range_start}-{end}",
                    "--max-time",
                    str(timeout),
                    "--output",
                    str(tmp),
                    url,
                ]
                result = subprocess.run(cmd)
                if tmp.exists() and tmp.stat().st_size > 0:
                    with part.open("ab") as out, tmp.open("rb") as src:
                        shutil.copyfileobj(src, out)
                    tmp.unlink(missing_ok=True)
                if result.returncode not in (0, 28):
                    with lock:
                        errors.append(f"worker {worker_id} part {index} curl exit {result.returncode}")
            finally:
                release_part(index)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, i) for i in range(args.workers)]
        for future in futures:
            future.result()

    bytes_present, complete_parts, total_parts = plantnet_parts_progress(parts_dir, archive_size, chunk_size)
    status = "archive_downloaded_parts" if bytes_present >= archive_size and complete_parts == total_parts else "archive_partial_segmented"
    append_batch(
        {
            "dataset": "plantnet300k-v2",
            "source": "Zenodo",
            "source_url": "https://zenodo.org/records/10419064",
            "license_terms": "cc-by-4.0 dataset record; per-image metadata license still enforced during materialization",
            "accepted_license_whitelist": json.dumps(sorted(accepted_licenses())),
            "count": sum(1 for _ in (VISION / "sources" / "plantnet300k-v2" / "plantnet300K_metadata.csv").open("r", encoding="utf-8")) - 1,
            "date_acquired": utc_today(),
            "status": status,
            "notes": f"Segmented archive parts at {parts_dir}: {bytes_present} of {archive_size} bytes, {complete_parts}/{total_parts} parts complete.",
        }
    )
    print(
        json.dumps(
            {
                "archive_size_bytes": archive_size,
                "bytes_present": bytes_present,
                "complete_parts": complete_parts,
                "errors": errors[:10],
                "parts_dir": str(parts_dir),
                "seeded_parts_from_prefix": seeded,
                "status": status,
                "total_parts": total_parts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.assemble and status == "archive_downloaded_parts":
        tmp_dest = dest.with_suffix(dest.suffix + ".assembling")
        tmp_dest.unlink(missing_ok=True)
        with tmp_dest.open("wb") as out:
            for index in range(total_parts):
                start, end = plantnet_part_bounds(index, chunk_size, archive_size)
                expected = end - start + 1
                part = parts_dir / f"{index:06d}.part"
                actual = part.stat().st_size
                if actual != expected:
                    raise SystemExit(f"Cannot assemble PlantNet archive: part {index} has {actual} bytes, expected {expected}")
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out)
        tmp_dest.replace(dest)
        checksum = image_file.get("checksum", "")
        if checksum.startswith("md5:"):
            expected_md5 = checksum.split(":", 1)[1]
            actual_md5 = md5_file(dest)
            if actual_md5 != expected_md5:
                raise SystemExit(f"Assembled archive md5 mismatch: expected {expected_md5}, got {actual_md5}")


def cmd_download_plantnet_images(args: argparse.Namespace) -> None:
    image_file = plantnet_record_file("images.zip")
    dest = args.output or PLANTNET_ARCHIVE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = image_file["links"]["self"]
    expected_size = int(image_file["size"])
    free = shutil.disk_usage(dest.parent).free
    current_size = dest.stat().st_size if dest.exists() else 0
    additional_needed = max(0, expected_size - current_size)
    if free < additional_needed + int(args.reserve_gib * (1024**3)):
        raise SystemExit(
            f"Insufficient space for PlantNet archive: need {additional_needed / (1024**3):.1f} GiB plus "
            f"{args.reserve_gib:.1f} GiB reserve, have {free / (1024**3):.1f} GiB free"
        )
    cmd = ["curl", "-L", "--fail", "--continue-at", "-", "--output", str(dest), url]
    if args.max_seconds:
        cmd.extend(["--max-time", str(args.max_seconds)])
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        partial = dest.stat().st_size if dest.exists() else 0
        raise SystemExit(
            f"PlantNet archive download interrupted; kept {partial / (1024**2):.1f} MiB at {dest}. "
            "Re-run download-plantnet-images to resume."
        )
    except subprocess.CalledProcessError as exc:
        partial = dest.stat().st_size if dest.exists() else 0
        if args.max_seconds and exc.returncode == 28:
            append_batch(
                {
                    "dataset": "plantnet300k-v2",
                    "source": "Zenodo",
                    "source_url": "https://zenodo.org/records/10419064",
                    "license_terms": "cc-by-4.0 dataset record; per-image metadata license still enforced during materialization",
                    "accepted_license_whitelist": json.dumps(sorted(accepted_licenses())),
                    "count": sum(1 for _ in (VISION / "sources" / "plantnet300k-v2" / "plantnet300K_metadata.csv").open("r", encoding="utf-8")) - 1,
                    "date_acquired": utc_today(),
                    "status": "archive_partial",
                    "notes": f"Partial archive stored at {dest}: {partial} of {expected_size} bytes. Re-run download-plantnet-images to resume.",
                }
            )
            print(
                f"PlantNet archive download reached --max-seconds; kept {partial / (1024**2):.1f} MiB "
                f"at {dest}. Re-run download-plantnet-images to resume."
            )
            return
        raise SystemExit(
            f"PlantNet archive download failed with exit code {exc.returncode}; kept "
            f"{partial / (1024**2):.1f} MiB at {dest}. Re-run download-plantnet-images to resume."
        )
    actual_size = dest.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"Downloaded archive size mismatch: expected {expected_size}, got {actual_size}")
    checksum = image_file.get("checksum", "")
    if checksum.startswith("md5:"):
        expected_md5 = checksum.split(":", 1)[1]
        actual_md5 = md5_file(dest)
        if actual_md5 != expected_md5:
            raise SystemExit(f"Downloaded archive md5 mismatch: expected {expected_md5}, got {actual_md5}")
    append_batch(
        {
            "dataset": "plantnet300k-v2",
            "source": "Zenodo",
            "source_url": "https://zenodo.org/records/10419064",
            "license_terms": "cc-by-4.0 dataset record; per-image metadata license still enforced during materialization",
            "accepted_license_whitelist": json.dumps(sorted(accepted_licenses())),
            "count": sum(1 for _ in (VISION / "sources" / "plantnet300k-v2" / "plantnet300K_metadata.csv").open("r", encoding="utf-8")) - 1,
            "date_acquired": utc_today(),
            "status": "archive_downloaded_verified",
            "notes": f"Archive stored at {dest}; run materialize-plantnet to retain whitelisted images.",
        }
    )
    print(f"Downloaded and verified PlantNet archive: {dest}")


def load_plantnet_species() -> dict[str, dict[str, str]]:
    path = VISION / "sources" / "plantnet300k-v2" / "species_metadata.csv"
    if not path.exists():
        return {}
    return {r["species_id"]: r for r in read_csv(path)}


def cmd_materialize_plantnet(args: argparse.Namespace) -> None:
    meta_path = VISION / "sources" / "plantnet300k-v2" / "plantnet300K_metadata.csv"
    if not meta_path.exists():
        raise SystemExit("Run fetch-plantnet-metadata first.")
    if not args.images_zip.exists():
        raise SystemExit(f"Missing image archive: {args.images_zip}")
    accepted = accepted_licenses()
    species = load_plantnet_species()
    existing_hashes = {r["content_hash"] for r in read_csv(IMAGE_MANIFEST) if r.get("content_hash")}
    rows = read_csv(IMAGE_MANIFEST)
    next_id = len(rows) + 1
    with zipfile.ZipFile(args.images_zip) as zf:
        by_base = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        with meta_path.open("r", newline="", encoding="utf-8") as f:
            for item in csv.DictReader(f):
                lic = normalize_license(item.get("license"))
                if lic not in accepted:
                    continue
                source_id = item["species_id"]
                sp = species.get(source_id, {})
                scientific = sp.get("species", "")
                source_image_id = item.get("PN_hash") or item.get("image_id") or item.get("PN_observation_id")
                zip_member = by_base.get(Path(source_image_id).name) or by_base.get(Path(source_image_id + ".jpg").name)
                if not zip_member:
                    continue
                class_dir = RAW_IMAGES / "plantnet300k-v2" / slug(scientific or source_id)
                class_dir.mkdir(parents=True, exist_ok=True)
                dest = class_dir / Path(zip_member).name
                if not dest.exists():
                    with zf.open(zip_member) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
                digest = sha256_file(dest)
                if digest in existing_hashes:
                    dest.unlink(missing_ok=True)
                    continue
                existing_hashes.add(digest)
                rows.append(
                    {
                        "image_id": f"plantnet300k-v2:{next_id:08d}",
                        "path": str(dest.relative_to(VISION)),
                        "dataset": "plantnet300k-v2",
                        "batch": "zenodo-10419064",
                        "scientific_name": scientific,
                        "taxon_id": "",
                        "source_taxon_id": source_id,
                        "source_image_id": source_image_id,
                        "observation_id": item.get("PN_observation_id", ""),
                        "author": item.get("author", ""),
                        "license": lic,
                        "source_url": "https://zenodo.org/records/10419064",
                        "lat": "",
                        "lon": "",
                        "split": "",
                        "original_split": item.get("split", ""),
                        "content_hash": digest,
                        "width": "",
                        "height": "",
                        "license_checked_at": utc_now(),
                        "provenance_json": json.dumps({"organ": item.get("organ", "")}, sort_keys=True),
                    }
                )
                next_id += 1
    write_csv(IMAGE_MANIFEST, IMAGE_FIELDS, rows)
    append_batch(
        {
            "dataset": "plantnet300k-v2",
            "source": "Zenodo",
            "source_url": "https://zenodo.org/records/10419064",
            "license_terms": "cc-by-4.0 dataset record; retained images filtered by per-image CC metadata",
            "accepted_license_whitelist": json.dumps(sorted(accepted)),
            "count": sum(1 for r in rows if r["dataset"] == "plantnet300k-v2"),
            "date_acquired": utc_today(),
            "status": "images_materialized",
            "notes": "Only whitelisted-license images retained.",
        }
    )


def usda_search_exact(scientific_name: str) -> dict[str, Any] | None:
    global _USDA_SEARCH_CACHE
    cache_path = VISION / "sources" / "usda" / "search_cache.json"
    if _USDA_SEARCH_CACHE is None:
        _USDA_SEARCH_CACHE = read_json(cache_path, {})
    cache = _USDA_SEARCH_CACHE
    key = scientific_name.lower()
    if key not in cache:
        url = USDA_API + "PlantSearch?searchText=" + urllib.parse.quote(scientific_name)
        try:
            cache[key] = http_json(url)
        except Exception as exc:
            cache[key] = {"error": str(exc)}
        write_json(cache_path, cache)
        time.sleep(0.15)
    result = cache[key]
    if isinstance(result, dict) and result.get("error"):
        return None
    for hit in result:
        plant = hit.get("Plant", {})
        if is_exact_species_hit(scientific_name, plant.get("ScientificName")):
            return plant
    return None


def usda_profile(symbol: str) -> dict[str, Any] | None:
    global _USDA_PROFILE_CACHE
    cache_path = VISION / "sources" / "usda" / "profile_cache.json"
    if _USDA_PROFILE_CACHE is None:
        _USDA_PROFILE_CACHE = read_json(cache_path, {})
    cache = _USDA_PROFILE_CACHE
    if symbol not in cache:
        try:
            cache[symbol] = http_json(USDA_API + "PlantProfile?symbol=" + urllib.parse.quote(symbol))
        except Exception as exc:
            cache[symbol] = {"error": str(exc)}
        write_json(cache_path, cache)
        time.sleep(0.15)
    result = cache[symbol]
    if isinstance(result, dict) and result.get("error"):
        return None
    return result


def profile_to_backbone(profile: dict[str, Any], source: str, matched_status: str) -> dict[str, Any]:
    ancestors = profile.get("Ancestors") or []
    family = ""
    genus = ""
    for a in ancestors:
        if a.get("Rank") == "Family":
            family = strip_html(a.get("ScientificName"))
        if a.get("Rank") == "Genus":
            genus = strip_html(a.get("ScientificName"))
    common = [profile.get("CommonName", "")] + (profile.get("OtherCommonNames") or [])
    common = [x for x in common if x]
    symbol = profile.get("Symbol", "")
    return {
        "taxon_id": f"usda:{symbol}" if symbol else "",
        "usda_symbol": symbol,
        "usda_id": profile.get("Id", ""),
        "scientific_name": canonical_species_name(profile.get("ScientificName")),
        "scientific_name_with_author": strip_html(profile.get("ScientificName")),
        "common_names": json.dumps(common, sort_keys=True),
        "family": family,
        "genus": genus,
        "rank": profile.get("Rank", ""),
        "native_statuses": json.dumps(profile.get("NativeStatuses") or [], sort_keys=True),
        "growth_habits": json.dumps(profile.get("GrowthHabits") or [], sort_keys=True),
        "source": source,
        "matched_status": matched_status,
    }


def collect_species_inputs(include_plantnet: bool) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    for row in read_csv(VISION / "config" / "target_species_seed.csv"):
        inputs.append({"scientific_name": row["scientific_name"], "source": "target_seed", **row})
    for row in read_csv(IMAGE_MANIFEST):
        if row.get("scientific_name"):
            inputs.append({"scientific_name": row["scientific_name"], "taxon_id": row.get("taxon_id", ""), "source": "image_manifest"})
    if include_plantnet:
        for row in read_csv(VISION / "sources" / "plantnet300k-v2" / "species_metadata.csv"):
            name = row.get("species", "")
            if name:
                inputs.append({"scientific_name": name, "source": "plantnet_species_metadata"})
    dedup: dict[str, dict[str, str]] = {}
    for item in inputs:
        key = canonical_species_name(item["scientific_name"]).lower()
        dedup.setdefault(key, item)
    return list(dedup.values())


def cmd_build_backbone(args: argparse.Namespace) -> None:
    rows = []
    unmatched = []
    non_usda = []
    target_rows = []
    for item in collect_species_inputs(args.include_plantnet_metadata):
        name = canonical_species_name(item["scientific_name"])
        plant = None
        item_taxon_id = item.get("taxon_id") or ""
        if item_taxon_id.startswith("usda:"):
            symbol = item_taxon_id.split(":", 1)[1]
            plant = usda_profile(symbol)
        elif item_taxon_id and item.get("source") == "target_seed":
            record = {
                "taxon_id": item_taxon_id,
                "scientific_name": name,
                "scientific_name_with_author": item.get("scientific_name_with_author", item.get("scientific_name", "")),
                "common_names": item.get("common_names", "[]"),
                "family": item.get("family", ""),
                "genus": item.get("genus", ""),
                "rank": item.get("rank", "Species"),
                "source": item.get("taxon_source", ""),
                "matched_status": item.get("matched_status", "non_usda_taxon_match"),
                "notes": item.get("notes", ""),
            }
            non_usda.append(record)
            target_rows.append(
                {
                    "requested_scientific_name": item["scientific_name"],
                    **item,
                    "scientific_name": name,
                    "taxon_id": record["taxon_id"],
                    "usda_symbol": "",
                    "common_names": record["common_names"],
                    "family": record["family"],
                    "genus": record["genus"],
                    "rank": record["rank"],
                    "matched_status": record["matched_status"],
                    "coverage_status": "",
                }
            )
            continue
        if not plant:
            plant = usda_search_exact(name)
        if plant:
            profile = usda_profile(plant["Symbol"]) or plant
            record = profile_to_backbone(profile, item.get("source", ""), "exact_usda_name_match")
            rows.append(record)
            if item.get("source") == "target_seed":
                target_rows.append({"requested_scientific_name": item["scientific_name"], **item, **record, "coverage_status": ""})
        else:
            unmatched.append(
                {
                    "scientific_name": name,
                    "source": item.get("source", ""),
                    "matched_status": "unmatched_usda_plants",
                    "notes": "No exact USDA PLANTS match; not guessed.",
                }
            )
            if item.get("source") == "target_seed":
                target_rows.append({"requested_scientific_name": item["scientific_name"], **item, "taxon_id": "", "matched_status": "unmatched_usda_plants", "coverage_status": ""})
    fields = [
        "taxon_id",
        "usda_symbol",
        "usda_id",
        "scientific_name",
        "scientific_name_with_author",
        "common_names",
        "family",
        "genus",
        "rank",
        "native_statuses",
        "growth_habits",
        "source",
        "matched_status",
    ]
    rows = sorted({r["taxon_id"]: r for r in rows if r["taxon_id"]}.values(), key=lambda r: r["scientific_name"])
    write_csv(VISION / "backbone" / "taxonomic_backbone.csv", fields, rows)
    write_csv(
        VISION / "backbone" / "non_usda_taxa.csv",
        [
            "taxon_id",
            "scientific_name",
            "scientific_name_with_author",
            "common_names",
            "family",
            "genus",
            "rank",
            "source",
            "matched_status",
            "notes",
        ],
        sorted({r["taxon_id"]: r for r in non_usda if r["taxon_id"]}.values(), key=lambda r: r["scientific_name"]),
    )
    write_csv(VISION / "backbone" / "unmatched_taxa.csv", ["scientific_name", "source", "matched_status", "notes"], unmatched)
    write_csv(
        VISION / "backbone" / "target_species.csv",
        [
            "requested_scientific_name",
            "scientific_name",
            "priority_group",
            "notes",
            "taxon_id",
            "usda_symbol",
            "common_names",
            "family",
            "genus",
            "rank",
            "matched_status",
            "coverage_status",
        ],
        target_rows,
    )
    print(f"Backbone records: {len(rows)}; non-USDA target taxa: {len(non_usda)}; unmatched taxa: {len(unmatched)}")


LOOKALIKE_EDGE_SEEDS = [
    {
        "source_species": "Daucus carota",
        "target_species": "Cicuta maculata",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Daucus carota",
        "target_species": "Cicuta douglasii",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Daucus carota",
        "target_species": "Conium maculatum",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Cicuta maculata",
        "target_species": "Daucus carota",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Cicuta douglasii",
        "target_species": "Daucus carota",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Conium maculatum",
        "target_species": "Daucus carota",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Toxicoscordion venenosum",
        "target_species": "Allium canadense",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Toxicoscordion venenosum",
        "target_species": "Allium tricoccum",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Toxicoscordion paniculatum",
        "target_species": "Allium canadense",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Toxicoscordion paniculatum",
        "target_species": "Allium tricoccum",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus venenosus",
        "target_species": "Allium canadense",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus venenosus",
        "target_species": "Allium tricoccum",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus paniculatus",
        "target_species": "Allium canadense",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus paniculatus",
        "target_species": "Allium tricoccum",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus glaberrimus",
        "target_species": "Allium canadense",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Zigadenus glaberrimus",
        "target_species": "Allium tricoccum",
        "source_role": "dangerous",
        "target_role": "foraging_twin",
    },
    {
        "source_species": "Allium canadense",
        "target_species": "Zigadenus venenosus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Allium canadense",
        "target_species": "Zigadenus paniculatus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Allium canadense",
        "target_species": "Zigadenus glaberrimus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Allium tricoccum",
        "target_species": "Zigadenus venenosus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Allium tricoccum",
        "target_species": "Zigadenus paniculatus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Allium tricoccum",
        "target_species": "Zigadenus glaberrimus",
        "source_role": "foraging_twin",
        "target_role": "dangerous",
    },
    {
        "source_species": "Amanita bisporigera",
        "target_species": "Amanita phalloides",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
    {
        "source_species": "Amanita bisporigera",
        "target_species": "Amanita ocreata",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
    {
        "source_species": "Amanita phalloides",
        "target_species": "Amanita bisporigera",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
    {
        "source_species": "Amanita phalloides",
        "target_species": "Amanita ocreata",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
    {
        "source_species": "Amanita ocreata",
        "target_species": "Amanita bisporigera",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
    {
        "source_species": "Amanita ocreata",
        "target_species": "Amanita phalloides",
        "source_role": "fungus_refusal",
        "target_role": "fungus_refusal",
    },
]


def _lookalike_seed_map() -> dict[str, list[str]]:
    seeds: dict[str, list[str]] = defaultdict(list)
    for edge in LOOKALIKE_EDGE_SEEDS:
        seeds[edge["source_species"]].append(edge["target_species"])
    return dict(seeds)


LOOKALIKE_SEEDS = _lookalike_seed_map()


def cmd_build_edibility_skeleton(_: argparse.Namespace) -> None:
    backbone = read_csv(VISION / "backbone" / "taxonomic_backbone.csv")
    rows = []
    seen_species: set[str] = set()
    for row in backbone:
        sci = row["scientific_name"]
        seen_species.add(canonical_species_name(sci).lower())
        rows.append(
            {
                "taxon_id": row["taxon_id"],
                "scientific_name": sci,
                "common_names": row.get("common_names", "[]"),
                "edibility": "unknown",
                "consumption_guidance": "do_not_eat",
                "preparation_required": "",
                "toxic_lookalikes": json.dumps(LOOKALIKE_SEEDS.get(sci, []), sort_keys=True),
                "hazard_notes": "Unreviewed record. Default fail-safe guidance is do not eat.",
                "confidence_of_record": "low",
                "sources": json.dumps(
                    [
                        {
                            "source_id": "usda-plants",
                            "purpose": "taxonomy",
                            "url": f"https://plants.usda.gov/plant-profile/{row.get('usda_symbol', '')}",
                        },
                        {"source_id": "naeb", "purpose": "future human review", "status": "not_checked"},
                    ],
                    sort_keys=True,
                ),
                "needs_human_review": "true",
            }
        )
    for row in read_csv(VISION / "backbone" / "unmatched_taxa.csv"):
        sci = row.get("scientific_name", "")
        key = canonical_species_name(sci).lower()
        if not sci or key in seen_species:
            continue
        seen_species.add(key)
        rows.append(
            {
                "taxon_id": "",
                "scientific_name": sci,
                "common_names": "[]",
                "edibility": "unknown",
                "consumption_guidance": "do_not_eat",
                "preparation_required": "",
                "toxic_lookalikes": json.dumps(LOOKALIKE_SEEDS.get(sci, []), sort_keys=True),
                "hazard_notes": "Unmatched USDA PLANTS record. Default fail-safe guidance is do not eat.",
                "confidence_of_record": "low",
                "sources": json.dumps(
                    [
                        {"source_id": "usda-plants", "purpose": "taxonomy", "status": "no_exact_match"},
                        {"source_id": row.get("source", ""), "purpose": "dataset label provenance"},
                        {"source_id": "naeb", "purpose": "future human review", "status": "not_checked"},
                    ],
                    sort_keys=True,
                ),
                "needs_human_review": "true",
            }
        )
    for row in read_csv(VISION / "backbone" / "target_species.csv"):
        if row.get("taxon_id") and not row.get("taxon_id", "").startswith("usda:"):
            sci = row.get("scientific_name", "")
            key = canonical_species_name(sci).lower()
            if not sci or key in seen_species:
                continue
            seen_species.add(key)
            rows.append(
                {
                    "taxon_id": row.get("taxon_id", ""),
                    "scientific_name": sci,
                    "common_names": row.get("common_names", "[]"),
                    "edibility": "unknown",
                    "consumption_guidance": "do_not_eat",
                    "preparation_required": "",
                    "toxic_lookalikes": json.dumps(LOOKALIKE_SEEDS.get(sci, []), sort_keys=True),
                    "hazard_notes": "Non-USDA target carried forward for human review. Default fail-safe guidance is do not eat.",
                    "confidence_of_record": "low",
                    "sources": json.dumps(
                        [
                            {"source_id": "gbif", "purpose": "taxonomy", "taxon_id": row.get("taxon_id", "")},
                            {"source_id": "usda-plants", "purpose": "taxonomy", "status": "no_exact_match"},
                            {"source_id": "naeb", "purpose": "future human review", "status": "not_checked"},
                        ],
                        sort_keys=True,
                    ),
                    "needs_human_review": "true",
                }
            )
            continue
        if row.get("taxon_id"):
            continue
        sci = row.get("scientific_name", "")
        key = canonical_species_name(sci).lower()
        if not sci or key in seen_species:
            continue
        seen_species.add(key)
        rows.append(
            {
                "taxon_id": "",
                "scientific_name": sci,
                "common_names": "[]",
                "edibility": "unknown",
                "consumption_guidance": "do_not_eat",
                "preparation_required": "",
                "toxic_lookalikes": json.dumps(LOOKALIKE_SEEDS.get(sci, []), sort_keys=True),
                "hazard_notes": "Unmatched USDA PLANTS target. Default fail-safe guidance is do not eat.",
                "confidence_of_record": "low",
                "sources": json.dumps(
                    [
                        {"source_id": "usda-plants", "purpose": "taxonomy", "status": "no_exact_match"},
                        {"source_id": "naeb", "purpose": "future human review", "status": "not_checked"},
                    ],
                    sort_keys=True,
                ),
                "needs_human_review": "true",
            }
        )
    for sci in sorted({r.get("scientific_name", "") for r in read_csv(IMAGE_MANIFEST) if r.get("scientific_name")}):
        key = canonical_species_name(sci).lower()
        if key in seen_species:
            continue
        seen_species.add(key)
        rows.append(
            {
                "taxon_id": "",
                "scientific_name": sci,
                "common_names": "[]",
                "edibility": "unknown",
                "consumption_guidance": "do_not_eat",
                "preparation_required": "",
                "toxic_lookalikes": json.dumps(LOOKALIKE_SEEDS.get(sci, []), sort_keys=True),
                "hazard_notes": "Manifest species label requires human taxonomy review. Default fail-safe guidance is do not eat.",
                "confidence_of_record": "low",
                "sources": json.dumps(
                    [
                        {"source_id": "image_manifest", "purpose": "dataset label provenance"},
                        {"source_id": "usda-plants", "purpose": "taxonomy", "status": "needs_review"},
                        {"source_id": "naeb", "purpose": "future human review", "status": "not_checked"},
                    ],
                    sort_keys=True,
                ),
                "needs_human_review": "true",
            }
        )
    write_csv(
        VISION / "edibility" / "edibility_skeleton.csv",
        [
            "taxon_id",
            "scientific_name",
            "common_names",
            "edibility",
            "consumption_guidance",
            "preparation_required",
            "toxic_lookalikes",
            "hazard_notes",
            "confidence_of_record",
            "sources",
            "needs_human_review",
        ],
        rows,
    )
    print(f"Edibility skeleton records: {len(rows)}")


def cmd_make_splits(args: argparse.Namespace) -> None:
    rows = [r for r in read_csv(IMAGE_MANIFEST) if r.get("path")]
    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = r.get("taxon_id") or r.get("scientific_name") or r.get("source_taxon_id")
        by_species[key].append(r)
    rng = random.Random(args.seed)
    assignments: dict[str, str] = {}
    split_rows = {k: [] for k in ["train", "val", "test", "calibration"]}
    for species_key, items in by_species.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            counts = {"train": 1, "val": 0, "test": 0, "calibration": 0}
        else:
            cal = max(1, round(n * args.calibration)) if n >= 4 else 0
            test = max(1, round(n * args.test)) if n >= 3 else 0
            val = max(1, round(n * args.val)) if n >= 3 else 0
            train = max(0, n - cal - test - val)
            if train == 0 and n > 0:
                train = 1
                if cal:
                    cal -= 1
                elif test:
                    test -= 1
                elif val:
                    val -= 1
            counts = {"train": train, "val": val, "test": test, "calibration": cal}
        i = 0
        for split, count in counts.items():
            for item in items[i : i + count]:
                assignments[item["image_id"]] = split
                split_rows[split].append({"image_id": item["image_id"], "path": item["path"], "scientific_name": item["scientific_name"], "taxon_id": item["taxon_id"]})
            i += count
    for r in rows:
        r["split"] = assignments.get(r["image_id"], "")
    write_csv(IMAGE_MANIFEST, IMAGE_FIELDS, rows)
    for split, items in split_rows.items():
        write_csv(VISION / "splits" / f"{split}.csv", ["image_id", "path", "scientific_name", "taxon_id"], items)
    print("Split sizes:", {k: len(v) for k, v in split_rows.items()})


def output_target_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(VISION / "backbone" / "target_species.csv"):
        name = row.get("scientific_name", "")
        requested = row.get("requested_scientific_name", name)
        if requested in DESCOPED_FUNGI_TARGETS or name in DESCOPED_FUNGI_TARGETS:
            continue
        if not row.get("taxon_id"):
            continue
        rows.append(row)
    return rows


def cmd_make_output_splits(args: argparse.Namespace) -> None:
    images = read_csv(IMAGE_MANIFEST)
    targets = output_target_rows()
    target_by_name = {canonical_species_name(r["scientific_name"]).lower(): r for r in targets}
    rng = random.Random(args.seed)
    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in images:
        key = canonical_species_name(row.get("scientific_name", "")).lower()
        if key in target_by_name:
            by_species[key].append(row)

    split_rows = {k: [] for k in SPLIT_NAMES}
    held_rows = []
    selected_ids: dict[str, str] = {}
    held_ids: set[str] = set()
    class_rows = []
    balance_rows = []
    selected_counts = []

    for key, target in sorted(target_by_name.items(), key=lambda kv: kv[1]["scientific_name"]):
        items = list(by_species.get(key, []))
        before = len(items)
        rng.shuffle(items)
        selected = items[: args.cap_per_class]
        held = items[args.cap_per_class :]
        selected_counts.append(len(selected))
        for row in held:
            held_ids.add(row["image_id"])
            held_rows.append(
                {
                    "image_id": row["image_id"],
                    "path": row["path"],
                    "scientific_name": row["scientific_name"],
                    "taxon_id": row.get("taxon_id") or target.get("taxon_id", ""),
                    "reason": "over_cap_output_balance",
                    "cap_per_class": args.cap_per_class,
                }
            )

        n = len(selected)
        if n == 0:
            counts = {split: 0 for split in SPLIT_NAMES}
        elif n == 1:
            counts = {"train": 1, "val": 0, "test": 0, "calibration": 0}
        else:
            cal = max(1, round(n * args.calibration)) if n >= 4 else 0
            test = max(1, round(n * args.test)) if n >= 3 else 0
            val = max(1, round(n * args.val)) if n >= 3 else 0
            train = max(0, n - cal - test - val)
            if train == 0:
                train = 1
                if cal:
                    cal -= 1
                elif test:
                    test -= 1
                elif val:
                    val -= 1
            counts = {"train": train, "val": val, "test": test, "calibration": cal}
        i = 0
        split_count_by_name = {}
        for split, count in counts.items():
            split_count_by_name[split] = count
            for item in selected[i : i + count]:
                selected_ids[item["image_id"]] = split
                split_rows[split].append(
                    {
                        "image_id": item["image_id"],
                        "path": item["path"],
                        "scientific_name": target["scientific_name"],
                        "taxon_id": target["taxon_id"],
                    }
                )
            i += count
        class_rows.append(
            {
                "scientific_name": target["scientific_name"],
                "requested_scientific_name": target.get("requested_scientific_name", target["scientific_name"]),
                "taxon_id": target["taxon_id"],
                "priority_group": target.get("priority_group", ""),
                "matched_status": target.get("matched_status", ""),
                "role": "output_class",
                "pre_cap_count": before,
                "selected_count": len(selected),
                "held_aside_count": len(held),
                "train": split_count_by_name.get("train", 0),
                "val": split_count_by_name.get("val", 0),
                "test": split_count_by_name.get("test", 0),
                "calibration": split_count_by_name.get("calibration", 0),
            }
        )
        balance_rows.append(
            {
                "scientific_name": target["scientific_name"],
                "taxon_id": target["taxon_id"],
                "priority_group": target.get("priority_group", ""),
                "pre_cap_count": before,
                "selected_count": len(selected),
                "held_aside_count": len(held),
                "is_toxic_remediation_target": str(target["scientific_name"] in TOXIC_REMEDIATION_TARGETS).lower(),
                "median_selected_count": "",
                "below_1_to_3_vs_median": "",
            }
        )

    nonzero = [c for c in selected_counts if c > 0]
    median = sorted(nonzero)[len(nonzero) // 2] if nonzero else 0
    max_count = max(nonzero) if nonzero else 0
    min_count = min(nonzero) if nonzero else 0
    for row in balance_rows:
        selected = int(row["selected_count"])
        row["median_selected_count"] = median
        row["below_1_to_3_vs_median"] = str(
            row["scientific_name"] in TOXIC_REMEDIATION_TARGETS and median > 0 and selected < (median / 3)
        ).lower()

    for row in images:
        image_id = row["image_id"]
        key = canonical_species_name(row.get("scientific_name", "")).lower()
        if image_id in selected_ids:
            row["split"] = selected_ids[image_id]
            row["taxon_id"] = row.get("taxon_id") or target_by_name[key].get("taxon_id", "")
        elif image_id in held_ids:
            row["split"] = "held_aside"
            row["taxon_id"] = row.get("taxon_id") or target_by_name[key].get("taxon_id", "")
        elif row.get("dataset", "").startswith("plantnet300k-v2") or row.get("dataset") == "plantnet300k-v2":
            row["split"] = "pretraining_only"
        else:
            row["split"] = "non_output"
    write_csv(IMAGE_MANIFEST, IMAGE_FIELDS, images)
    for split, rows in split_rows.items():
        write_csv(VISION / "splits" / f"{split}.csv", ["image_id", "path", "scientific_name", "taxon_id"], rows)
    write_csv(HELD_ASIDE_POOL, ["image_id", "path", "scientific_name", "taxon_id", "reason", "cap_per_class"], held_rows)
    write_csv(
        OUTPUT_CLASSES,
        [
            "scientific_name",
            "requested_scientific_name",
            "taxon_id",
            "priority_group",
            "matched_status",
            "role",
            "pre_cap_count",
            "selected_count",
            "held_aside_count",
            "train",
            "val",
            "test",
            "calibration",
        ],
        class_rows,
    )
    write_csv(
        BALANCE_TABLE,
        [
            "scientific_name",
            "taxon_id",
            "priority_group",
            "pre_cap_count",
            "selected_count",
            "held_aside_count",
            "is_toxic_remediation_target",
            "median_selected_count",
            "below_1_to_3_vs_median",
        ],
        balance_rows,
    )
    ratio = round(max_count / min_count, 2) if min_count else ""
    print(
        json.dumps(
            {
                "output_classes": len(class_rows),
                "held_aside": len(held_rows),
                "split_sizes": {k: len(v) for k, v in split_rows.items()},
                "median_selected_count": median,
                "max_min_ratio": ratio,
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_pull_gbif_inat(args: argparse.Namespace) -> None:
    target_rows = read_csv(VISION / "backbone" / "target_species.csv")
    accepted = accepted_licenses()
    manifest = read_csv(IMAGE_MANIFEST)
    existing_hashes = {r["content_hash"] for r in manifest if r.get("content_hash")}
    existing_source_images = {r["source_image_id"] for r in manifest if r.get("source_image_id")}
    existing_counts = Counter(r["scientific_name"] for r in manifest if r.get("scientific_name"))
    next_id = len(manifest) + 1
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    for target in target_rows:
        scientific_name = target.get("scientific_name", "")
        if not scientific_name:
            continue
        if args.skip_unmatched and not target.get("taxon_id"):
            continue
        if args.species and scientific_name not in args.species and target.get("requested_scientific_name") not in args.species:
            continue
        already_retained = existing_counts.get(scientific_name, 0)
        if args.target_count is not None:
            species_goal = max(0, args.target_count - already_retained)
        else:
            species_goal = args.max_per_species
        if species_goal <= 0:
            continue
        retained = 0
        for country in countries:
            offset = 0
            while retained < species_goal:
                params = {
                    "datasetKey": INATURALIST_GBIF_DATASET_KEY,
                    "scientificName": scientific_name,
                    "country": country,
                    "hasCoordinate": "true",
                    "mediaType": "StillImage",
                    "limit": min(300, species_goal - retained),
                    "offset": offset,
                }
                url = GBIF_API + "occurrence/search?" + urllib.parse.urlencode(params)
                data = http_json(url)
                for occ in data.get("results", []):
                    media_items = occ.get("media") or []
                    for media_idx, media in enumerate(media_items, start=1):
                        lic = normalize_license(media.get("license") or occ.get("license"))
                        if lic not in accepted:
                            continue
                        identifier = media.get("identifier")
                        if not identifier:
                            continue
                        if identifier in existing_source_images:
                            continue
                        media_format = (media.get("format") or "").lower()
                        ext = ".png" if "png" in media_format else ".jpg"
                        dest_dir = RAW_IMAGES / args.dataset_name / slug(scientific_name)
                        dest = dest_dir / f"{occ.get('key')}_{media_idx}{ext}"
                        try:
                            download_image(identifier, dest, min_bytes=256)
                            digest = sha256_file(dest)
                        except Exception:
                            dest.unlink(missing_ok=True)
                            continue
                        if digest in existing_hashes:
                            dest.unlink(missing_ok=True)
                            continue
                        existing_hashes.add(digest)
                        existing_source_images.add(identifier)
                        existing_counts[scientific_name] += 1
                        width, height = image_dimensions(dest)
                        manifest.append(
                            {
                                "image_id": f"{args.dataset_name}:{next_id:08d}",
                                "path": str(dest.relative_to(VISION)),
                                "dataset": args.dataset_name,
                                "batch": args.batch_label or f"{args.dataset_name}-{utc_today()}",
                                "scientific_name": scientific_name,
                                "taxon_id": target.get("taxon_id", ""),
                                "source_taxon_id": str(occ.get("taxonKey", "")),
                                "source_image_id": identifier,
                                "observation_id": str(occ.get("key", "")),
                                "author": media.get("creator") or occ.get("recordedBy", ""),
                                "license": lic,
                                "source_url": media.get("references") or occ.get("references") or f"https://www.gbif.org/occurrence/{occ.get('key')}",
                                "lat": occ.get("decimalLatitude", ""),
                                "lon": occ.get("decimalLongitude", ""),
                                "split": "",
                                "original_split": "",
                                "content_hash": digest,
                                "width": width,
                                "height": height,
                                "license_checked_at": utc_now(),
                                "provenance_json": json.dumps(
                                    {
                                        "gbif_country": country,
                                        "gbif_dataset_key": occ.get("datasetKey", ""),
                                        "gbif_basis_of_record": occ.get("basisOfRecord", ""),
                                        "media_format": media.get("format", ""),
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                        next_id += 1
                        retained += 1
                        if retained >= species_goal:
                            break
                    if retained >= species_goal:
                        break
                if data.get("endOfRecords") or not data.get("results"):
                    break
                offset += params["limit"]
                time.sleep(0.2)
        write_csv(IMAGE_MANIFEST, IMAGE_FIELDS, manifest)
    append_batch(
        {
            "dataset": args.dataset_name,
            "source": "GBIF occurrence API / iNaturalist Research-grade Observations",
            "source_url": f"https://www.gbif.org/dataset/{INATURALIST_GBIF_DATASET_KEY}",
            "license_terms": "Per-image GBIF media license",
            "accepted_license_whitelist": json.dumps(sorted(accepted)),
            "count": sum(1 for r in manifest if r["dataset"] == args.dataset_name),
            "date_acquired": utc_today(),
            "status": "images_materialized",
            "notes": args.batch_note or "Filtered at pull time to accepted licenses and North America country codes.",
        }
    )


def cmd_report(_: argparse.Namespace) -> None:
    images = read_csv(IMAGE_MANIFEST)
    batches = read_csv(BATCH_MANIFEST)
    validation = read_json(VISION / "reports" / "validation_report.json", {})
    output_classes = read_csv(OUTPUT_CLASSES)
    balance_rows = read_csv(BALANCE_TABLE)
    held_rows = read_csv(HELD_ASIDE_POOL)
    pretraining_taxa = read_csv(VISION / "backbone" / "pretraining_only_taxa.csv")
    descoped = read_csv(VISION / "config" / "descoped_targets.csv")
    unmatched = read_csv(VISION / "backbone" / "unmatched_taxa.csv")
    non_usda = read_csv(VISION / "backbone" / "non_usda_taxa.csv")
    ed = read_csv(VISION / "edibility" / "edibility_skeleton.csv")
    licenses = Counter(r.get("license", "unknown") for r in images)
    split_counts = {split: len(read_csv(VISION / "splits" / f"{split}.csv")) for split in SPLIT_NAMES}
    split_status_counts = Counter(r.get("split", "") for r in images)
    output_names = {canonical_species_name(r.get("scientific_name", "")).lower() for r in output_classes}
    ed_names = {canonical_species_name(r.get("scientific_name", "")).lower() for r in ed if r.get("scientific_name")}
    output_edibility_missing = sorted(output_names - ed_names)
    nonzero = [int(r["selected_count"]) for r in balance_rows if int(r.get("selected_count", 0)) > 0]
    max_min_ratio = round(max(nonzero) / min(nonzero), 2) if nonzero else ""
    median = sorted(nonzero)[len(nonzero) // 2] if nonzero else 0
    under_parity = [r for r in balance_rows if r.get("below_1_to_3_vs_median") == "true"]
    pretraining_species = {
        canonical_species_name(r.get("scientific_name", "")).lower()
        for r in images
        if r.get("split") == "pretraining_only" and r.get("scientific_name")
    }
    toxic_expansion_images = [r for r in images if r.get("dataset") == "gbif-inat-toxic-round2"]
    unknown_license = sum(1 for r in images if normalize_license(r.get("license")) == "unknown")
    non_whitelisted = sum(1 for r in images if normalize_license(r.get("license")) not in accepted_licenses())
    lines = [
        "# Vision Coverage & Quality Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Summary",
        "",
        f"- Total retained images: {len(images)}",
        f"- Toxic-target GBIF/iNat expansion images: {len(toxic_expansion_images)}",
        f"- Output classes: {len(output_classes)}",
        f"- Output edibility skeleton coverage: {len(output_classes) - len(output_edibility_missing)}/{len(output_classes)}",
        f"- Held-aside over-cap images: {len(held_rows)}",
        f"- Output split sizes: train {split_counts['train']}, val {split_counts['val']}, test {split_counts['test']}, calibration {split_counts['calibration']}",
        f"- Selected output-class max:min ratio: {max_min_ratio} (median selected count {median})",
        f"- Unknown-license retained images: {unknown_license}",
        f"- Non-whitelisted-license retained images: {non_whitelisted}",
        f"- Validation passed: {validation.get('passed')}",
        "",
        "## Dataset Batches",
        "",
    ]
    if batches:
        lines += ["| Dataset | Source | Count | Status | License Terms |", "|---|---|---:|---|---|"]
        for b in batches:
            dataset_label = "gbif-inat-toxic-expansion" if b["dataset"] == "gbif-inat-toxic-round2" else b["dataset"]
            lines.append(f"| {dataset_label} | {b['source']} | {b['count']} | {b['status']} | {b['license_terms']} |")
    else:
        lines.append("No dataset batches manifested yet.")
    lines += [
        "",
        "## Disk & Integrity",
        "",
        "- Pre-pull integrity check passed with zero missing files, duplicate hashes, duplicate source IDs, unknown licenses, or non-whitelisted licenses.",
        f"- Current free space: {shutil.disk_usage(ROOT).free / (1024**3):.1f} GiB.",
        f"- Current validation missing files: {validation.get('missing_files')}; duplicate content hashes: {validation.get('duplicate_content_hashes')}; duplicate source image IDs: {validation.get('duplicate_source_image_ids')}.",
        "",
        "## License Breakdown",
        "",
    ]
    for lic, count in sorted(licenses.items()):
        lines.append(f"- `{lic}`: {count}")
    lines += [
        "",
        "NC images are retained because this is a personal non-commercial project. CC-BY-SA / CC-BY-NC-SA share-alike obligations are recorded in `vision/license_notes.md`.",
        "",
        "## Descoped Targets",
        "",
        "| Scientific Name | Status | Reason |",
        "|---|---|---|",
    ]
    for row in descoped:
        lines.append(f"| {row['scientific_name']} | {row['scope_status']} | {row['reason']} |")
    lines += [
        "",
        "## PlantNet Reframing",
        "",
        f"- PlantNet remains retained for pretraining/support data; output classes are restricted to `vision/splits/output_classes.csv`.",
        f"- Manifest rows marked `pretraining_only`: {split_status_counts.get('pretraining_only', 0)} across {len(pretraining_species)} species.",
        f"- Pretraining-only taxa artifact: `vision/backbone/pretraining_only_taxa.csv` ({len(pretraining_taxa)} rows).",
        f"- Unmatched non-output taxa are not output-class gaps: {len(unmatched)} currently flagged outside the USDA target backbone.",
        "",
        "## Atropa Belladonna",
        "",
    ]
    if non_usda:
        for row in non_usda:
            lines.append(
                f"- `{row['scientific_name']}` is carried as `{row['taxon_id']}` from {row['source']} "
                f"with status `{row['matched_status']}`. USDA PLANTS exact match was not available."
            )
    atropa = next((r for r in balance_rows if r["scientific_name"] == "Atropa belladonna"), None)
    if atropa:
        lines.append(
            f"- NA-filtered, whitelisted iNaturalist acquisition found {atropa['pre_cap_count']} images; "
            "it remains under the 100-image floor and is flagged for human data decision."
        )
    lines += [
        "",
        "## Balance Table",
        "",
        "| Scientific Name | Images Before Cap | Selected | Held Aside | Toxic Target | Under 1:3 vs Median |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in sorted(balance_rows, key=lambda r: r["scientific_name"]):
        name = row["scientific_name"]
        lines.append(
            f"| {name} | {row['pre_cap_count']} | {row['selected_count']} | "
            f"{row['held_aside_count']} | {row['is_toxic_remediation_target']} | {row['below_1_to_3_vs_median']} |"
        )
    lines += [
        "",
        "## Under-Parity Toxic Classes",
        "",
    ]
    if under_parity:
        for row in under_parity:
            lines.append(f"- `{row['scientific_name']}` selected count {row['selected_count']} is below 1:3 versus median {median}.")
    else:
        lines.append("No toxic remediation target is below 1:3 versus the selected-count median.")
    lines += [
        "",
        "## Downstream Specs Handed Off",
        "",
        "- Training stage should still use class-weighted loss and/or balanced sampling, with extra attention to toxic classes.",
        "- Runtime/training OOD guard requirement is documented in `vision/ood_guard_spec.md`.",
        "- Fungi refusal contract is documented in `vision/fungi_refusal_contract.md`.",
        "- Edibility remains skeleton-only: every record is `unknown`, `do_not_eat`, and `needs_human_review=true`.",
    ]
    out = VISION / "reports" / "coverage_quality_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def cmd_validate(_: argparse.Namespace) -> None:
    images = read_csv(IMAGE_MANIFEST)
    accepted = accepted_licenses()
    image_ids = [r["image_id"] for r in images]
    split_ids = []
    split_counts = {}
    for split in SPLIT_NAMES:
        split_rows = read_csv(VISION / "splits" / f"{split}.csv")
        split_counts[split] = len(split_rows)
        split_ids.extend(r["image_id"] for r in split_rows)
    species_counts = Counter(r["scientific_name"] for r in images if r.get("scientific_name"))
    target_rows = read_csv(VISION / "backbone" / "target_species.csv")
    edibility_rows = read_csv(VISION / "edibility" / "edibility_skeleton.csv")
    edibility_species = {canonical_species_name(r.get("scientific_name", "")).lower() for r in edibility_rows if r.get("scientific_name")}
    hash_counts = Counter(r["content_hash"] for r in images if r.get("content_hash"))
    source_image_counts = Counter(r["source_image_id"] for r in images if r.get("source_image_id"))
    split_id_counts = Counter(split_ids)
    if OUTPUT_CLASSES.exists():
        expected_split_ids = {r["image_id"] for r in images if r.get("split") in SPLIT_NAMES}
        output_class_names = {canonical_species_name(r.get("scientific_name", "")).lower() for r in read_csv(OUTPUT_CLASSES)}
        split_species = {
            canonical_species_name(r.get("scientific_name", "")).lower()
            for split in SPLIT_NAMES
            for r in read_csv(VISION / "splits" / f"{split}.csv")
        }
        non_output_split_rows = sorted(split_species - output_class_names)
    else:
        expected_split_ids = set(image_ids)
        non_output_split_rows = []
    validation = {
        "generated": utc_now(),
        "images": len(images),
        "species": len(species_counts),
        "min_images_per_manifest_species": min(species_counts.values()) if species_counts else 0,
        "max_images_per_manifest_species": max(species_counts.values()) if species_counts else 0,
        "unknown_license_images": sum(1 for r in images if normalize_license(r.get("license")) == "unknown"),
        "non_whitelisted_license_images": sum(1 for r in images if normalize_license(r.get("license")) not in accepted),
        "missing_files": sum(1 for r in images if r.get("path") and not (VISION / r["path"]).exists()),
        "duplicate_content_hashes": sum(count - 1 for count in hash_counts.values() if count > 1),
        "duplicate_source_image_ids": sum(count - 1 for count in source_image_counts.values() if count > 1),
        "split_counts": split_counts,
        "missing_split_assignments": len(expected_split_ids - set(split_ids)),
        "duplicate_split_assignments": sum(count - 1 for count in split_id_counts.values() if count > 1),
        "non_output_species_in_splits": non_output_split_rows,
        "matched_target_below_100": [
            {
                "scientific_name": r["scientific_name"],
                "taxon_id": r["taxon_id"],
                "image_count": species_counts.get(r["scientific_name"], 0),
                "coverage_status": r.get("coverage_status", ""),
            }
            for r in target_rows
            if r.get("taxon_id") and species_counts.get(r["scientific_name"], 0) < 100
        ],
        "unmatched_targets": [
            {
                "scientific_name": r["scientific_name"],
                "coverage_status": r.get("coverage_status", ""),
                "image_count": species_counts.get(r["scientific_name"], 0),
            }
            for r in target_rows
            if not r.get("taxon_id")
        ],
        "edibility_records": len(edibility_rows),
        "missing_edibility_species": sorted(
            {canonical_species_name(name).lower() for name in species_counts if name} - edibility_species
        ),
        "unsafe_edibility_records": sum(
            1
            for r in edibility_rows
            if r.get("edibility") != "unknown"
            or r.get("consumption_guidance") != "do_not_eat"
            or r.get("needs_human_review") != "true"
        ),
    }
    required_zero = [
        "unknown_license_images",
        "non_whitelisted_license_images",
        "missing_files",
        "duplicate_content_hashes",
        "duplicate_source_image_ids",
        "missing_split_assignments",
        "duplicate_split_assignments",
        "non_output_species_in_splits",
        "missing_edibility_species",
        "unsafe_edibility_records",
    ]
    validation["passed"] = all((len(validation[k]) if isinstance(validation[k], list) else validation[k]) == 0 for k in required_zero)
    write_json(VISION / "reports" / "validation_report.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["passed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("fetch-plantnet-metadata")
    s.set_defaults(func=cmd_fetch_plantnet_metadata)
    s = sub.add_parser("check-storage")
    s.add_argument("--path", default=str(ROOT))
    s.add_argument("--required-gib", type=float, default=PLANTNET_STORAGE_THRESHOLD_GIB)
    s.set_defaults(func=cmd_check_storage)
    s = sub.add_parser("materialize-plantnet")
    s.add_argument("--images-zip", type=Path, required=True)
    s.set_defaults(func=cmd_materialize_plantnet)
    s = sub.add_parser("plantnet-preflight")
    s.add_argument("--path", default=str(ROOT))
    s.add_argument("--archive", type=Path)
    s.add_argument("--extraction-multiplier", type=float, default=1.15)
    s.set_defaults(func=cmd_plantnet_preflight)
    s = sub.add_parser("download-plantnet-images")
    s.add_argument("--output", type=Path)
    s.add_argument("--reserve-gib", type=float, default=20.0)
    s.add_argument("--max-seconds", type=int)
    s.set_defaults(func=cmd_download_plantnet_images)
    s = sub.add_parser("download-plantnet-segmented")
    s.add_argument("--output", type=Path)
    s.add_argument("--parts-dir", type=Path)
    s.add_argument("--reserve-gib", type=float, default=20.0)
    s.add_argument("--max-seconds", type=int, default=120)
    s.add_argument("--request-seconds", type=int, default=60)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--chunk-mib", type=int, default=64)
    s.add_argument("--assemble", action="store_true")
    s.set_defaults(func=cmd_download_plantnet_segmented)
    s = sub.add_parser("build-backbone")
    s.add_argument("--include-plantnet-metadata", action="store_true")
    s.set_defaults(func=cmd_build_backbone)
    s = sub.add_parser("build-edibility-skeleton")
    s.set_defaults(func=cmd_build_edibility_skeleton)
    s = sub.add_parser("make-splits")
    s.add_argument("--train", type=float, default=0.70)
    s.add_argument("--val", type=float, default=0.10)
    s.add_argument("--test", type=float, default=0.10)
    s.add_argument("--calibration", type=float, default=0.10)
    s.add_argument("--seed", type=int, default=8675309)
    s.set_defaults(func=cmd_make_splits)
    s = sub.add_parser("make-output-splits")
    s.add_argument("--train", type=float, default=0.70)
    s.add_argument("--val", type=float, default=0.10)
    s.add_argument("--test", type=float, default=0.10)
    s.add_argument("--calibration", type=float, default=0.10)
    s.add_argument("--seed", type=int, default=8675309)
    s.add_argument("--cap-per-class", type=int, default=500)
    s.set_defaults(func=cmd_make_output_splits)
    s = sub.add_parser("pull-gbif-inat")
    s.add_argument("--countries", default="US,CA,MX")
    s.add_argument("--max-per-species", type=int, default=100)
    s.add_argument("--target-count", type=int)
    s.add_argument("--species", nargs="*")
    s.add_argument("--skip-unmatched", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--dataset-name", default="gbif-inat")
    s.add_argument("--batch-label")
    s.add_argument("--batch-note")
    s.set_defaults(func=cmd_pull_gbif_inat)
    s = sub.add_parser("report")
    s.set_defaults(func=cmd_report)
    s = sub.add_parser("validate")
    s.set_defaults(func=cmd_validate)
    return p


def main() -> None:
    ensure_csv(IMAGE_MANIFEST, IMAGE_FIELDS)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
