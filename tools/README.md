# Project Tooling

This directory is for reusable project tooling. Avoid adding one-off remediation
scripts here unless they are safe to re-run and documented as a stable command.

Current entry points:

- `text_pipeline.py` - Pipeline A text corpus build/remediation.
  - `python3 tools/text_pipeline.py`
- `vision_pipeline.py` - Pipeline B dataset/backbone/report generation.
  - `python3 tools/vision_pipeline.py report`
- `shard_vision_dataset.py` - Build split-based tar shards from the vision manifests.
- `upload_to_r2.py` - Upload and verify committed shard inventory objects in Cloudflare R2.
- `upload_checkpoints_to_r2.py` - Upload training run artifacts to R2 under `model-checkpoints/`.

Generated artifacts belong under `text/`, `vision/`, or run output dirs — not beside the tools.
See `training/README.md` and `runpod/` for the vision training workflow.
