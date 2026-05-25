#!/usr/bin/env python3
"""Upload Pocket Crockett vision shards to Cloudflare R2.

Reads ``vision/shards_manifest.csv`` (source of truth for shard SHA-256s),
uploads each shard to R2 with a pre-computed SHA-256 at the storage layer,
and verifies the server-side checksum after upload. Idempotent: shards
that already exist in R2 with the expected SHA-256 are skipped before the
local tar file is required, so older uploaded shards can be deleted locally
while their rows remain in the committed manifest.

Credentials are read from environment variables — either exported in
your shell or written to a ``.env`` file at the repo root (gitignored):

    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET

See ``.env.example`` for the file format. Already-exported shell vars take
precedence over ``.env`` entries, so you can override per-shell.

Typical use:

    # Inspect what would happen without touching R2.
    python3 tools/upload_to_r2.py --dry-run

    # Upload one small shard end-to-end first.
    python3 tools/upload_to_r2.py --only val-000.tar

    # Full set.
    python3 tools/upload_to_r2.py

Outputs:

    vision/upload_manifest.json   generated run record (gitignored)
    vision/upload_manifest.sha256 generated per-key SHA-256 list (gitignored)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
SHARDS_DIR = VISION / "shards"
SHARD_MANIFEST = VISION / "shards_manifest.csv"
UPLOAD_MANIFEST_JSON = VISION / "upload_manifest.json"
UPLOAD_MANIFEST_SHA = VISION / "upload_manifest.sha256"
DOTENV_PATH = ROOT / ".env"

DEFAULT_PREFIX = "vision-shards/"
ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def load_dotenv(path: Path) -> None:
    """Load ``KEY=value`` lines from ``path`` into ``os.environ`` without
    overriding already-set variables. Silently no-ops if the file is absent."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Shard:
    name: str
    split: str
    index: int
    image_count: int
    bytes: int
    sha256_hex: str

    @property
    def sha256_b64(self) -> str:
        # R2/S3 expect the SHA-256 header as base64 over the raw 32 bytes.
        return base64.b64encode(binascii.unhexlify(self.sha256_hex)).decode("ascii")

    @property
    def local_path(self) -> Path:
        return SHARDS_DIR / self.name


def read_shard_manifest(path: Path) -> list[Shard]:
    rows: list[Shard] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                Shard(
                    name=row["shard"],
                    split=row["split"],
                    index=int(row["index"]),
                    image_count=int(row["image_count"]),
                    bytes=int(row["bytes"]),
                    sha256_hex=row["sha256"].lower(),
                )
            )
    return rows


def b64_to_hex(s: str) -> str:
    return binascii.hexlify(base64.b64decode(s)).decode("ascii")


def make_client(account_id: str, access_key: str, secret_key: str):
    # Imported lazily so --help and arg validation work without boto3 installed.
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


def remote_sha256(client, bucket: str, key: str) -> Optional[str]:
    """Hex SHA-256 R2 has for ``key``, ``""`` if the object exists without a
    recorded checksum, or ``None`` if the object is absent."""
    from botocore.exceptions import ClientError

    try:
        resp = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return None
        raise
    b64 = resp.get("ChecksumSHA256")
    if not b64:
        return ""
    return b64_to_hex(b64)


def upload_shard(client, bucket: str, key: str, shard: Shard) -> None:
    with shard.local_path.open("rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ChecksumSHA256=shard.sha256_b64,
            ContentLength=shard.bytes,
            ContentType="application/x-tar",
        )


def emit_upload_manifest(results: list[dict], prefix: str, bucket: str) -> None:
    UPLOAD_MANIFEST_JSON.write_text(
        json.dumps({"bucket": bucket, "prefix": prefix, "results": results}, indent=2) + "\n"
    )
    lines = [
        f"{r['sha256']}  {r['key']}\n"
        for r in results
        if r["status"] in ("uploaded", "verified", "already_present")
    ]
    UPLOAD_MANIFEST_SHA.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload vision shards to Cloudflare R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Key prefix in the bucket (default: {DEFAULT_PREFIX!r})")
    parser.add_argument("--only", action="append", default=[], metavar="SHARD",
                        help="Restrict to specific shard name(s); may be passed multiple times")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report planned actions; make no R2 writes")
    args = parser.parse_args()

    load_dotenv(DOTENV_PATH)
    missing_env = [v for v in ENV_VARS if not os.environ.get(v)]
    if missing_env:
        print(f"error: missing env vars: {', '.join(missing_env)}", file=sys.stderr)
        return 2

    if not SHARD_MANIFEST.exists():
        print(f"error: shard manifest not found: {SHARD_MANIFEST}", file=sys.stderr)
        return 2

    shards = read_shard_manifest(SHARD_MANIFEST)
    if args.only:
        wanted = set(args.only)
        shards = [s for s in shards if s.name in wanted]
        missing = wanted - {s.name for s in shards}
        if missing:
            print(f"error: --only matched nothing for: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    bucket = os.environ["R2_BUCKET"]
    client = make_client(
        os.environ["R2_ACCOUNT_ID"],
        os.environ["R2_ACCESS_KEY_ID"],
        os.environ["R2_SECRET_ACCESS_KEY"],
    )

    results: list[dict] = []
    failures = 0
    for shard in shards:
        key = f"{args.prefix}{shard.name}"
        record = {"shard": shard.name, "key": key, "sha256": shard.sha256_hex, "bytes": shard.bytes}

        existing = remote_sha256(client, bucket, key)
        if existing == shard.sha256_hex:
            print(f"present {shard.name} -> s3://{bucket}/{key}")
            results.append({**record, "status": "already_present"})
            continue
        if existing not in (None, ""):
            print(f"mismatch {shard.name}: remote={existing} expected={shard.sha256_hex} — will re-upload")

        if not shard.local_path.exists():
            if existing == "":
                print(f"skip   {shard.name}: remote object exists without checksum and local file is absent at {shard.local_path}")
            else:
                print(f"skip   {shard.name}: not present locally at {shard.local_path}")
            results.append({**record, "status": "missing_local"})
            failures += 1
            continue

        if args.dry_run:
            print(f"plan   {shard.name} -> s3://{bucket}/{key} ({shard.bytes / 1e9:.2f} GB)")
            results.append({**record, "status": "would_upload"})
            continue

        print(f"upload {shard.name} -> s3://{bucket}/{key} ({shard.bytes / 1e9:.2f} GB)")
        upload_shard(client, bucket, key, shard)

        verified = remote_sha256(client, bucket, key)
        if verified == shard.sha256_hex:
            print(f"verify {shard.name}: ok")
            results.append({**record, "status": "verified"})
        else:
            print(f"verify {shard.name}: FAILED remote={verified}", file=sys.stderr)
            results.append({**record, "status": "verify_failed", "remote_sha256": verified})
            failures += 1

    emit_upload_manifest(results, args.prefix, bucket)
    print(f"wrote {UPLOAD_MANIFEST_JSON.relative_to(ROOT)}")
    print(f"wrote {UPLOAD_MANIFEST_SHA.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
