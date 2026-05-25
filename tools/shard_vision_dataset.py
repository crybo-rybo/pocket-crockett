#!/usr/bin/env python3
"""
Create tar shards for the Pocket Crockett vision dataset.

By default this packages the final output-class training splits plus the
held-aside over-cap pool. The pretraining-only PlantNet pool can be included
with --include-pretraining when there is enough local space or when streaming
directly to external storage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
IMAGE_MANIFEST = VISION / "images" / "manifest.csv"
SHARDS_DIR = VISION / "shards"
CHECKSUMS = VISION / "checksums.sha256"
SHARD_MANIFEST = VISION / "shards_manifest.csv"
SPLIT_NAMES = ["train", "val", "test", "calibration"]


@dataclass
class ShardItem:
    image_id: str
    path: str
    split: str
    scientific_name: str
    taxon_id: str
    license: str
    dataset: str
    source_url: str
    bytes: int


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tar_add_json(tar: tarfile.TarFile, arcname: str, data: object) -> int:
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
    return len(payload)


def load_items(include_pretraining: bool) -> dict[str, list[ShardItem]]:
    by_id = {r["image_id"]: r for r in read_csv(IMAGE_MANIFEST)}
    groups: dict[str, list[ShardItem]] = {name: [] for name in SPLIT_NAMES}

    for split in SPLIT_NAMES:
        for row in read_csv(VISION / "splits" / f"{split}.csv"):
            manifest_row = by_id[row["image_id"]]
            path = manifest_row["path"]
            file_path = VISION / path
            groups[split].append(
                ShardItem(
                    image_id=row["image_id"],
                    path=path,
                    split=split,
                    scientific_name=row.get("scientific_name") or manifest_row.get("scientific_name", ""),
                    taxon_id=row.get("taxon_id") or manifest_row.get("taxon_id", ""),
                    license=manifest_row.get("license", ""),
                    dataset=manifest_row.get("dataset", ""),
                    source_url=manifest_row.get("source_url", ""),
                    bytes=file_path.stat().st_size,
                )
            )

    held = []
    for row in read_csv(VISION / "splits" / "held_aside_pool.csv"):
        manifest_row = by_id[row["image_id"]]
        file_path = VISION / manifest_row["path"]
        held.append(
            ShardItem(
                image_id=row["image_id"],
                path=manifest_row["path"],
                split="heldaside",
                scientific_name=row.get("scientific_name") or manifest_row.get("scientific_name", ""),
                taxon_id=row.get("taxon_id") or manifest_row.get("taxon_id", ""),
                license=manifest_row.get("license", ""),
                dataset=manifest_row.get("dataset", ""),
                source_url=manifest_row.get("source_url", ""),
                bytes=file_path.stat().st_size,
            )
        )
    groups["heldaside"] = held

    if include_pretraining:
        pretraining = []
        for manifest_row in by_id.values():
            if manifest_row.get("split") != "pretraining_only":
                continue
            file_path = VISION / manifest_row["path"]
            pretraining.append(
                ShardItem(
                    image_id=manifest_row["image_id"],
                    path=manifest_row["path"],
                    split="pretraining",
                    scientific_name=manifest_row.get("scientific_name", ""),
                    taxon_id=manifest_row.get("taxon_id", ""),
                    license=manifest_row.get("license", ""),
                    dataset=manifest_row.get("dataset", ""),
                    source_url=manifest_row.get("source_url", ""),
                    bytes=file_path.stat().st_size,
                )
            )
        groups["pretraining"] = pretraining
    return groups


def write_shard(split: str, index: int, items: list[ShardItem], out_dir: Path) -> dict[str, object]:
    shard_name = f"{split}-{index:03d}.tar"
    shard_path = out_dir / shard_name
    tmp_path = shard_path.with_suffix(".tar.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w", format=tarfile.PAX_FORMAT) as tar:
        manifest_rows = []
        for item in items:
            source = VISION / item.path
            arcname = item.path
            tar.add(source, arcname=arcname, recursive=False)
            manifest_rows.append(
                {
                    "image_id": item.image_id,
                    "path": item.path,
                    "split": item.split,
                    "scientific_name": item.scientific_name,
                    "taxon_id": item.taxon_id,
                    "license": item.license,
                    "dataset": item.dataset,
                    "source_url": item.source_url,
                    "bytes": item.bytes,
                }
            )
        tar_add_json(tar, "manifest.json", manifest_rows)
    tmp_path.replace(shard_path)
    digest = sha256_file(shard_path)
    return {
        "shard": shard_name,
        "split": split,
        "index": index,
        "image_count": len(items),
        "bytes": shard_path.stat().st_size,
        "sha256": digest,
    }


def shard_items(items: list[ShardItem], max_bytes: int) -> list[list[ShardItem]]:
    shards: list[list[ShardItem]] = []
    current: list[ShardItem] = []
    current_bytes = 0
    for item in items:
        if current and current_bytes + item.bytes > max_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item.bytes
    if current:
        shards.append(current)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument("--include-pretraining", action="store_true")
    parser.add_argument("--splits", nargs="*", help="Optional subset of groups to shard.")
    args = parser.parse_args()

    max_bytes = int(args.max_shard_gib * (1024**3))
    groups = load_items(args.include_pretraining)
    selected_groups = args.splits or list(groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_rows: list[dict[str, object]] = []
    for split in selected_groups:
        items = groups.get(split)
        if items is None:
            raise SystemExit(f"Unknown split/group: {split}")
        for index, chunk in enumerate(shard_items(items, max_bytes)):
            shard_rows.append(write_shard(split, index, chunk, args.output_dir))

    write_csv(
        SHARD_MANIFEST,
        ["shard", "split", "index", "image_count", "bytes", "sha256"],
        shard_rows,
    )
    CHECKSUMS.write_text(
        "".join(f"{row['sha256']}  shards/{row['shard']}\n" for row in shard_rows),
        encoding="utf-8",
    )
    print(json.dumps({"shards": len(shard_rows), "images": sum(int(r["image_count"]) for r in shard_rows)}, indent=2))


if __name__ == "__main__":
    main()
