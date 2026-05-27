# Vision classifier training

Training code for the Pocket Crockett closed-set plant classifier. Data bytes live in Cloudflare R2 as tar shards; this package reads unpacked images on a RunPod Network Volume.

## Layout

```text
training/
  train.py              Fine-tune (or BioCLIP pretrain stage)
  evaluate.py           Val/test metrics + top-N report
  calibrate.py          Temperature scaling on calibration split
  export_onnx.py        Optional ONNX export for Jetson work later
  configs/              YAML configs
  artifacts/label_map.json
runpod/
  bootstrap.sh          Pull shards from R2, verify, unpack
  setup_venv.sh         Create CUDA-safe training venv
  train.sh              Stage wrapper (smoke/train/calibrate/eval/upload)
```

## RunPod first campaign

1. Attach a Network Volume (~120 GB) and start a CUDA PyTorch pod.
2. Clone this repo to `/workspace/pocket-crockett`.
3. Configure R2 credentials (`cp runpod/env.example .env` and fill in values) and rclone (`runpod/rclone.conf.example`).
4. Create the training venv. This reuses the pod image's CUDA PyTorch instead of reinstalling `torch` from pip:

```bash
cd /workspace/pocket-crockett
./runpod/setup_venv.sh
```

5. Bootstrap output-class shards:

```bash
cd /workspace/pocket-crockett
SHARDS="train-000.tar val-000.tar test-000.tar calibration-000.tar" ./runpod/bootstrap.sh   # smoke workflow
# or full output set:
./runpod/bootstrap.sh
```

6. Run stages:

```bash
./runpod/train.sh smoke
RUN_NAME=<dir-from-smoke> ./runpod/train.sh calibrate
RUN_NAME=<dir-from-smoke> ./runpod/train.sh eval
RUN_NAME=<dir-from-smoke> ./runpod/train.sh upload
```

For the full baseline:

```bash
./runpod/bootstrap.sh
RUN_NAME=baseline-v1 ./runpod/train.sh train
RUN_NAME=baseline-v1 ./runpod/train.sh calibrate
RUN_NAME=baseline-v1 ./runpod/train.sh eval
RUN_NAME=baseline-v1 ./runpod/train.sh upload
```

## Local dev (Mac)

Install PyTorch/torchvision for your local platform first, then the support deps for helper tests and label-map generation:

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision
.venv/bin/pip install -r training/requirements.txt
.venv/bin/python training/scripts/generate_label_map.py
.venv/bin/python -m unittest training.tests.test_label_map
.venv/bin/python training/scripts/smoke_dry_run.py
```

`smoke_dry_run.py` validates config, label map, split CSV parsing, and a model forward pass without unpacked images. On RunPod, use `./runpod/train.sh smoke` after `bootstrap.sh`.

If you still have unpacked images under `vision/images/raw/`, you can point `--data-root` at the repo `vision/` directory:

```bash
.venv/bin/python training/train.py \
  --config training/configs/smoke_test.yaml \
  --data-root vision \
  --output-dir /tmp/pc-smoke
```

## BioCLIP upgrade track

BioCLIP uses CLIP-style image normalization (`normalization: openai_clip`) and normalized image embeddings. Before spending another full pretraining run, first run a direct 47-class BioCLIP fine-tune from the base BioCLIP weights:

```bash
INCLUDE_BIOCLIP=1 ./runpod/setup_venv.sh
RUN_NAME=bioclip-direct-v2 ./runpod/train.sh bioclip-finetune
RUN_NAME=bioclip-direct-v2 ./runpod/train.sh calibrate
RUN_NAME=bioclip-direct-v2 ./runpod/train.sh eval
RUN_NAME=bioclip-direct-v2 ./runpod/train.sh upload
```

If direct BioCLIP is competitive, rerun the two-stage path:

```bash
INCLUDE_PRETRAINING=1 ./runpod/bootstrap.sh
RUN_NAME=bioclip-pretrain-v2 ./runpod/train.sh bioclip-pretrain
CHECKPOINT=/workspace/runs/bioclip-pretrain-v2/checkpoint-best.pt RUN_NAME=bioclip-v2 ./runpod/train.sh bioclip-finetune
RUN_NAME=bioclip-v2 ./runpod/train.sh calibrate
RUN_NAME=bioclip-v2 ./runpod/train.sh eval
RUN_NAME=bioclip-v2 ./runpod/train.sh upload
```

## Artifacts

Each run directory contains:

- `checkpoint-best.pt`, `checkpoint-last.pt`
- `calibration.json` (temperature + recommended confidence floor)
- `eval_report.json`, `eval_report.md`
- `label_map.json`

Upload to R2 with `tools/upload_checkpoints_to_r2.py` or `./runpod/train.sh upload`.
