---
title: "Pl@ntNet-300K-v2: A Plant Image Dataset for Set-Valued Classification"
version: "v2"
authors:
  - name: "C. Garcin"
  - name: "A. Joly"
  - name: "P. Bonnet"
  - name: "A. Affouard"
  - name: "J.-C. Lombardo"
  - name: "M. Chouet"
  - name: "M. Servajean"
  - name: "T. Lorieul"
  - name: "H. Goëau"
  - name: "J. Salmon"
date: "2026-02-10"
keywords: ["plant identification", "set-valued classification", "long-tailed distribution", "label ambiguity", "computer vision", "biodiversity"]
---

# Pl@ntNet-300K-v2: A Plant Image Dataset for Set-Valued Classification

**Pl@ntNet-300K-v2** is an image dataset designed for evaluating set-valued classification methods.
It is derived from the [Pl@ntNet](https://plantnet.org/) citizen observatory database and comprises **306,087 images** spanning **1,000 plant species**. The new version provides improved images resolution and better naming for the species.

---

## Key Features

The dataset reflects two notable characteristics, intrinsic to both the image acquisition process and the morphological diversity of plants:

- **Strong class imbalance**: A few species account for the majority of images.
- **Visual similarity**: Many species are visually indistinguishable, posing challenges even for expert identification.

These attributes make **Pl@ntNet-300K-v2** particularly suited for benchmarking set-valued classification approaches.

---

## Dataset Structure

### Image Organization
Images are partitioned into `train`, `test`, and `val` sets, stored in directories labeled `0000` to `0999`.

### Metadata Files

#### 1. `plantnet300K_metadata.csv`
Contains **306,087 image entries**, each with the following fields:

| Field               | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `species_id`        | Numerical species index (0–999)                                             |
| `PN_observation_id` | Unique Pl@ntNet observation identifier                                     |
| `organ`             | Plant organ in the image (`leaf`, `flower`, `other`, `habit`, `fruit`, etc.)|
| `author`            | Photographer’s identity                                                    |
| `license`           | Image license type (`cc-by-sa`, `cc-by-nc`, `cc-by-nc-sa`)                  |
| `split`             | Dataset partition (`train`, `test`, `val`)                                 |
| `PN_hash`           | Image hash name                                                            |

#### 2. `species_metadata.csv`
Provides taxonomic and conservation details for each species (aligned with the *World Checklist of Vascular Plants, v13*):

| Field               | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `species_id`        | Numerical species index (matches image directories)                        |
| `full_species`     | Full species name (with author)                                             |
| `species`           | Species name (without author)                                               |
| `genus`             | Genus name                                                                  |
| `family`            | Family name                                                                 |
| `epithet`           | Species epithet                                                             |
| `author`            | Species name author(s)                                                      |
| `unmatched_terms`  | Unresolved terms (e.g., `spp.`, `f.`)                                       |
| `iucn_status`       | IUCN conservation status (e.g., **EX**, **CR**, **EN**, **VU**, **LC**, **DD**, **NE**) |

---

## Resources

### Scientific Publication
The dataset and baseline results are described in:
[Garcin *et al.* (2021), *NeurIPS Datasets and Benchmarks*](https://openreview.net/forum?id=eLYinD0TtIt)

### Utilities
- **PyTorch tools** for data loading and model training:
  [GitHub Repository](https://github.com/plantnet/PlantNet-300K/)

---

## Citation

If you use this dataset, please cite the following publication:

```bibtex
@inproceedings{Garcin_Joly_Bonnet_Affouard_Lombardo_Chouet_Servajean_Lorieul_Salmon2021,
  author    = {Garcin, C. and Joly, A. and Bonnet, P. and Affouard, A. and Lombardo, J.-C. and Chouet, M. and Servajean, M. and Lorieul, T. and Salmon, J.},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  pdf       = {https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf},
  title     = {Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution},
  year      = {2021},
  comment   = {[<a href="https://github.com/plantnet/PlantNet-300K">Code</a>]}
}
