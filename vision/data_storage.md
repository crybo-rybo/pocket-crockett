# Vision Data Storage

The image bytes are stored as tar shards outside git. Git tracks the dataset description: manifests, splits, class lists, provenance, safety notes, and checksums.

## Local Shard Layout

Generated shards live under `vision/shards/` and are ignored by git.

Default shard groups:

- `train-*.tar`
- `val-*.tar`
- `test-*.tar`
- `calibration-*.tar`
- `heldaside-*.tar`

The `pretraining_only` PlantNet pool is intentionally not part of the closed-set output splits. It can be sharded separately with `--include-pretraining` when there is enough local space or when writing directly to external storage.

Current local shard set:

- `train-000.tar` through `train-003.tar`
- `val-000.tar`
- `test-000.tar`
- `calibration-000.tar`
- `heldaside-000.tar`

These 8 shards cover the final output-class training/evaluation data plus the held-aside over-cap pool. The PlantNet `pretraining_only` pool is described by `vision/backbone/pretraining_only_taxa.csv` and remains unsharded locally in this pass because it requires another ~38 GiB of tar output space.

## Generate Shards

```bash
python3 tools/shard_vision_dataset.py --max-shard-gib 4
```

This writes:

- `vision/shards_manifest.csv`
- `vision/checksums.sha256`
- tar files under `vision/shards/`

The per-image manifest is committed as `vision/images/manifest.csv.gz` because the uncompressed CSV exceeds GitHub's 100 MB file limit. The pipeline tools read the gzipped manifest when the local uncompressed CSV is absent. To restore the working CSV explicitly:

```bash
gzip -dc vision/images/manifest.csv.gz > vision/images/manifest.csv
```

To include the pretraining-only PlantNet pool as additional shards:

```bash
python3 tools/shard_vision_dataset.py --max-shard-gib 4 --include-pretraining
```

## Verify Shards

From the `vision/` directory:

```bash
shasum -a 256 -c checksums.sha256
```

The bucket/storage location is intentionally not recorded with any secret credentials. Keep any image bucket private because the dataset includes CC-BY-SA and CC-BY-NC-family images.
