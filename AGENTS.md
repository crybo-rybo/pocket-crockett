# Pocket Crockett Agent Guide

## Documentation Source

Project documentation lives outside this repository at:

`/Users/conorrybacki/Management/Vault/Pocket-Crockett`

Consult that vault before making architecture, data, safety, or roadmap decisions. Current source documents include:

- `PocketCrockettOverview.md`
- `DataGatheringGuide.md`

## Project Purpose

Pocket Crockett is an offline-first frontier assistant designed to run on an Nvidia Jetson Orin Nano. It has two core capabilities:

- Chat: answer survival, farming, food preservation, and first-aid questions from a curated, cited knowledge base.
- Vision: identify plants or trees from a USB camera and look up whether the candidate species is edible, useful, dangerous, or unknown.

The design ethos is: offline-first, fail-safe, cite-your-sources.

## Non-Negotiable Safety Rules

- Fail safe. If the system is uncertain, the answer must default to `unknown`, `do_not_eat`, or equivalent conservative guidance.
- Never present uncertain plant, mushroom, medical, or edibility information as definitive.
- Show top-N vision candidates instead of a single confident answer when identification is uncertain.
- Gate any edibility verdict on both model identification confidence and confidence in the edibility record.
- Surface deadly look-alikes prominently regardless of model score.
- Cite sources for chat answers and distinguish high-trust modern sources from low-trust historical or herbal sources.

## Architecture Direction

- Use RAG for survival and medical knowledge. Do not fine-tune factual survival knowledge into model weights.
- Keep fine-tuning limited to persona and answer style: calm, practical, risk-aware, and source-grounded.
- Target small on-device LLMs in the 3B-8B range with 4-bit quantization.
- Start with llama.cpp or Ollama for local LLM runtime unless the project later needs TensorRT-LLM or MLC for performance.
- Use FAISS, Chroma, or sqlite-vec for local vector retrieval.
- Use a fine-tuned BioCLIP, EfficientNet, ConvNeXt, MobileNetV3, or small ViT-style classifier for species identification.
- Do not use YOLO as the plant identifier. YOLO may be used only as an optional crop or detection stage before classification.
- Calibrate model confidence before using scores for safety-sensitive decisions.

## Data Principles

- Track provenance and license for every text source, image, dataset, and edibility record from day one.
- Prefer smaller, cleaner, vetted datasets over large noisy collections.
- Avoid copyrighted modern survival, bushcraft, foraging, or medical books as training data unless explicit rights are available.
- Keep raw source data untouched and write cleaned, chunked, and derived artifacts separately.
- Preserve per-item metadata needed for attribution, trust scoring, and reproducibility.

## Knowledge Pipeline

Text corpus work should produce:

- Raw source downloads.
- Clean Markdown or plain text.
- RAG chunks of roughly 300-800 tokens with semantic boundaries and 10-15% overlap.
- Metadata on every chunk, including source title, source id, license, topic, page when available, and trust tier.
- A small JSONL fine-tuning set for answer style only, with a held-out evaluation slice.

Prioritize modern public-domain or openly licensed sources such as US Army survival manuals, USDA/NCHFP food preservation guidance, cooperative extension publications, FEMA/Ready.gov material, and carefully checked Hesperian material.

## Vision Pipeline

Vision work should produce:

- Licensed image manifests with scientific name, taxon id, source, license, split, and any useful location metadata.
- Stratified train, validation, test, and calibration splits.
- A calibrated classifier that can report top-N candidates.
- A reviewed species-to-edibility lookup table keyed to a taxonomic backbone such as USDA PLANTS.

Start small with PlantNet-300K or another manageable licensed dataset before expanding to North-America-filtered iNaturalist data.

## Edibility Data Model

The edibility lookup must be a deliberate reviewed artifact, not a casual scrape. It should support at least:

- `taxon_id`
- `scientific_name`
- `common_names`
- `edibility`
- `preparation_required`
- `toxic_lookalikes`
- `hazard_notes`
- `confidence_of_record`
- `sources`

The `edibility` enum must include conservative states such as `deadly`, `toxic`, `unknown`, and `do_not_eat`. New or unreviewed species must default to `unknown` or `do_not_eat`.

## Suggested Repository Layout

When code and data tooling are added, keep the two pipelines separate:

```text
project-data/
  text/
    raw/
    clean/
    rag_chunks/
    finetune/
    MANIFEST.csv
  vision/
    images/raw/
    images/manifest.csv
    splits/
    edibility/
    MANIFEST.csv
```

If production code, training code, or app code is added later, preserve a clear boundary between:

- Data acquisition and cleaning.
- RAG indexing and retrieval.
- LLM serving and prompting.
- Vision training, calibration, and export.
- Jetson runtime integration.
- Safety-critical edibility lookup and presentation.

## Development Guidance

- Keep offline operation as a primary constraint.
- Prefer simple, inspectable data formats such as CSV, JSON, JSONL, Markdown, and plain text for source and intermediate artifacts.
- Make source citation and license metadata part of every pipeline interface instead of bolting it on later.
- Add tests around safety gates, default edibility behavior, citation behavior, and confidence-threshold logic as soon as those modules exist.
- Treat old public-domain medical or herbal material as low-trust unless verified against modern sources.
- Document any model, dataset, threshold, or source-quality decision near the code or manifest that depends on it.
