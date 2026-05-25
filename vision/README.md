# Pocket Crockett Vision Dataset

This tree is the manifest-driven Pipeline B workspace.

- `images/raw/`: retained original images only. Unknown-license images must never be stored here.
- `images/manifest.csv`: one row per retained image with license and provenance.
- `splits/`: stratified `train`, `val`, `test`, and separate `calibration` files.
- `backbone/`: USDA PLANTS taxonomic backbone and target species list.
- `edibility/`: fail-safe skeleton only. Records default to `unknown` and `do_not_eat`, with `needs_human_review=true`.
- `sources/`: small source metadata and downloaded manifests. Large image archives are intentionally not committed.
- `shards/`: local tar shards for transfer/training. Ignored by git; hashes live in `checksums.sha256`.
- `shards_manifest.csv`: shard inventory with split, image count, byte count, and SHA-256.

Run `python3 tools/vision_pipeline.py --help` for the repeatable pipeline.
Run `python3 tools/shard_vision_dataset.py --help` for the tar-sharding workflow.
