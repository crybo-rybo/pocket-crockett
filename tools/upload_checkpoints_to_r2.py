#!/usr/bin/env python3
"""Upload training run artifacts to Cloudflare R2.

Uploads checkpoints, calibration, and evaluation reports from a run directory
to ``s3://$R2_BUCKET/$R2_CHECKPOINT_PREFIX/<run-name>/...``.

Credentials match ``tools/upload_to_r2.py`` (``R2_*`` env vars or repo ``.env``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = ROOT / ".env"
DEFAULT_PREFIX = "model-checkpoints/"
ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
UPLOAD_NAMES = (
    "checkpoint-best.pt",
    "checkpoint-last.pt",
    "calibration.json",
    "eval_report.json",
    "eval_report.md",
    "label_map.json",
    "train_history.json",
    "config.resolved.json",
    "model.onnx",
    "export_onnx.json",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_client(account_id: str, access_key: str, secret_key: str):
    import boto3
    from botocore.config import Config

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def content_type(name: str) -> str:
    if name.endswith(".pt"):
        return "application/octet-stream"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".md"):
        return "text/markdown"
    if name.endswith(".onnx"):
        return "application/octet-stream"
    return "application/octet-stream"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    load_dotenv(DOTENV_PATH)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=os.environ.get("R2_CHECKPOINT_PREFIX", DEFAULT_PREFIX))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [v for v in ENV_VARS if not os.environ.get(v)]
    if missing:
        print(f"error: missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    if not args.run_dir.is_dir():
        print(f"error: run dir not found: {args.run_dir}", file=sys.stderr)
        return 2

    bucket = os.environ["R2_BUCKET"]
    client = make_client(
        os.environ["R2_ACCOUNT_ID"],
        os.environ["R2_ACCESS_KEY_ID"],
        os.environ["R2_SECRET_ACCESS_KEY"],
    )
    run_name = args.run_dir.name
    results: list[dict] = []
    uploaded = 0

    for name in UPLOAD_NAMES:
        path = args.run_dir / name
        if not path.exists():
            continue
        key = f"{args.prefix}{run_name}/{name}"
        record = {
            "file": name,
            "key": key,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if args.dry_run:
            print(f"plan   {path} -> s3://{bucket}/{key}")
            record["status"] = "would_upload"
        else:
            with path.open("rb") as f:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=f,
                    ContentLength=record["bytes"],
                    ContentType=content_type(name),
                )
            print(f"upload {name} -> s3://{bucket}/{key}")
            record["status"] = "uploaded"
            uploaded += 1
        results.append(record)

    manifest = {
        "bucket": bucket,
        "prefix": args.prefix,
        "run_name": run_name,
        "results": results,
    }
    manifest_path = args.run_dir / "checkpoint_upload_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {display_path(manifest_path)}")
    if uploaded == 0 and not args.dry_run:
        print("warning: no artifacts uploaded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
