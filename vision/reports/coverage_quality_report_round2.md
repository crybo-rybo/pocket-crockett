# Vision Coverage & Quality Report - Round 2 Remediation

Generated: 2026-05-25T14:37:02+00:00

## Summary

- Total retained images: 316678 (baseline 310676; delta +6002)
- Round-2 toxic-target GBIF/iNat additions: 6002
- Output classes: 47
- Output edibility skeleton coverage: 47/47
- Held-aside over-cap images: 10836
- Output split sizes: train 8000, val 1143, test 1143, calibration 1143
- Selected output-class max:min ratio: 250.0 (median selected count 100)
- Unknown-license retained images: 0
- Non-whitelisted-license retained images: 0
- Validation passed: True

## Disk & Integrity

- Pre-pull integrity check passed with zero missing files, duplicate hashes, duplicate source IDs, unknown licenses, or non-whitelisted licenses.
- Current free space: 23.4 GiB.
- Current validation missing files: 0; duplicate content hashes: 0; duplicate source image IDs: 0.

## License Breakdown

- `cc-by-4.0`: 1015
- `cc-by-nc`: 376
- `cc-by-nc-4.0`: 9129
- `cc-by-nc-sa`: 97
- `cc-by-nc-sa-4.0`: 80
- `cc-by-sa`: 305603
- `cc-by-sa-4.0`: 10
- `cc0`: 368

NC images are retained because this is a personal non-commercial project. CC-BY-SA / CC-BY-NC-SA share-alike obligations are recorded in `vision/license_notes.md`.

## Descoped Targets

| Scientific Name | Status | Reason |
|---|---|---|
| Amanita bisporigera | descoped_v1_fungi | Fungi are out of scope for Pocket Crockett v1; runtime must refuse fungus identification. |
| Amanita phalloides | descoped_v1_fungi | Fungi are out of scope for Pocket Crockett v1; runtime must refuse fungus identification. |
| Amanita ocreata | descoped_v1_fungi | Fungi are out of scope for Pocket Crockett v1; runtime must refuse fungus identification. |

## PlantNet Reframing

- PlantNet remains retained for pretraining/support data; output classes are restricted to `vision/splits/output_classes.csv`.
- Manifest rows marked `pretraining_only`: 294413 across 987 species.
- Pretraining-only taxa artifact: `vision/backbone/pretraining_only_taxa.csv` (991 rows).
- Unmatched non-output taxa are not output-class gaps: 475 currently flagged outside the USDA target backbone.

## Atropa Belladonna

- `Atropa belladonna` is carried as `gbif:3802655` from GBIF species match API with status `gbif_exact_species_match`. USDA PLANTS exact match was not available.
- NA-filtered, whitelisted iNaturalist acquisition found 2 images; it remains under the 100-image floor and is flagged for human data decision.

## Balance Table

| Scientific Name | Baseline | Post Pull | Selected | Held Aside | Toxic Target | Under 1:3 vs Median |
|---|---:|---:|---:|---:|---|---|
| Acer rubrum | 100 | 100 | 100 | 0 | false | false |
| Acer saccharum | 100 | 100 | 100 | 0 | false | false |
| Achillea millefolium | 100 | 100 | 100 | 0 | false | false |
| Actaea pachypoda | 100 | 500 | 500 | 0 | true | false |
| Actaea rubra | 100 | 500 | 500 | 0 | true | false |
| Allium canadense | 100 | 100 | 100 | 0 | false | false |
| Allium tricoccum | 100 | 100 | 100 | 0 | false | false |
| Amelanchier alnifolia | 100 | 100 | 100 | 0 | false | false |
| Asimina triloba | 100 | 100 | 100 | 0 | false | false |
| Atropa belladonna | 0 | 2 | 2 | 0 | true | true |
| Betula papyrifera | 100 | 100 | 100 | 0 | false | false |
| Carya ovata | 100 | 100 | 100 | 0 | false | false |
| Cicuta douglasii | 100 | 500 | 500 | 0 | true | false |
| Cicuta maculata | 100 | 500 | 500 | 0 | true | false |
| Conium maculatum | 100 | 500 | 500 | 0 | true | false |
| Daucus carota | 9110 | 9110 | 500 | 8610 | false | false |
| Digitalis purpurea | 100 | 500 | 500 | 0 | true | false |
| Echinacea purpurea | 100 | 100 | 100 | 0 | false | false |
| Fagus grandifolia | 100 | 100 | 100 | 0 | false | false |
| Fragaria virginiana | 127 | 127 | 127 | 0 | false | false |
| Juglans nigra | 100 | 100 | 100 | 0 | false | false |
| Juniperus virginiana | 100 | 100 | 100 | 0 | false | false |
| Liriodendron tulipifera | 2726 | 2726 | 500 | 2226 | false | false |
| Nerium oleander | 100 | 500 | 500 | 0 | true | false |
| Opuntia humifusa | 100 | 100 | 100 | 0 | false | false |
| Phytolacca americana | 100 | 500 | 500 | 0 | true | false |
| Pinus strobus | 100 | 100 | 100 | 0 | false | false |
| Plantago major | 100 | 100 | 100 | 0 | false | false |
| Prunus serotina | 100 | 100 | 100 | 0 | false | false |
| Quercus alba | 100 | 100 | 100 | 0 | false | false |
| Quercus rubra | 100 | 100 | 100 | 0 | false | false |
| Rubus allegheniensis | 100 | 100 | 100 | 0 | false | false |
| Rubus idaeus | 100 | 100 | 100 | 0 | false | false |
| Sambucus nigra | 100 | 100 | 100 | 0 | false | false |
| Solanum dulcamara | 100 | 500 | 500 | 0 | true | false |
| Toxicodendron diversilobum | 100 | 500 | 500 | 0 | true | false |
| Toxicodendron radicans | 100 | 500 | 500 | 0 | true | false |
| Toxicodendron vernix | 100 | 500 | 500 | 0 | true | false |
| Tsuga canadensis | 100 | 100 | 100 | 0 | false | false |
| Typha angustifolia | 100 | 100 | 100 | 0 | false | false |
| Typha latifolia | 100 | 100 | 100 | 0 | false | false |
| Urtica dioica | 100 | 100 | 100 | 0 | false | false |
| Vaccinium angustifolium | 100 | 100 | 100 | 0 | false | false |
| Vaccinium corymbosum | 100 | 100 | 100 | 0 | false | false |
| Zigadenus glaberrimus | 100 | 500 | 500 | 0 | true | false |
| Zigadenus paniculatus | 100 | 500 | 500 | 0 | true | false |
| Zigadenus venenosus | 100 | 500 | 500 | 0 | true | false |

## Under-Parity Toxic Classes

- `Atropa belladonna` selected count 2 is below 1:3 versus median 100.

## Downstream Specs Handed Off

- Training stage should still use class-weighted loss and/or balanced sampling, with extra attention to toxic classes.
- Runtime/training OOD guard requirement is documented in `vision/ood_guard_spec.md`.
- Fungi refusal contract is documented in `vision/fungi_refusal_contract.md`.
- Edibility remains skeleton-only: every record is `unknown`, `do_not_eat`, and `needs_human_review=true`.
