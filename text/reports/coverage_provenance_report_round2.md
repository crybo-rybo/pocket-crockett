# Pocket Crockett Pipeline A Coverage & Provenance Report - Round 2 Remediation

Generated: 2026-05-25

## Baseline Diff

- Baseline report date: 2026-05-25
- Baseline acquired sources: 13
- Round 2 sources added: 4
- Total acquired sources after round 2: 17
- Total RAG chunks after round 2: 2239
- Foraging claim records flagged `needs_human_review`: 863

## Sources Added

- `us_military_fm_4_25_11_first_aid` - FM 4-25.11 (FM 21-11) First Aid (Change 1) (public-domain-usgov, high, clean, 160 chunks)
- `usda_food_plants_north_american_indians` - Food Plants of the North American Indians (public-domain-usgov, low, clean, 74 chunks)
- `appropedia_rainwater_harvesting_wayback` - Rainwater Harvesting (CC-BY-SA-3.0, low, clean, 7 chunks)
- `epa_emergency_disinfection_drinking_water` - Emergency Disinfection of Drinking Water (public-domain-usgov, high, clean, 4 chunks)

## Trust-Tier Audit

- `wikibooks_gardening`: high -> low - Community-authored Wikibooks page should not share the same trust tier as USDA/Army/FEMA/EPA sources. Reclassification already applied.

## FM 21-76 OCR Spot-Check

- `us_army_fm_21_76`: clean -> clean; Spot-check found readable headings and plant/first-aid content with no replacement characters; retained clean flag.

## Topic Coverage After Round 2

- shelter: 3 high, 0 low - covered
- water: 4 high, 1 low - covered
- fire: 2 high, 0 low - covered
- navigation: 2 high, 0 low - covered
- first_aid: 4 high, 0 low - covered
- low_resource_medicine: 1 high, 0 low - covered
- wound_care: 1 high, 0 low - covered
- foraging: 2 high, 1 low - covered
- edible_plants: 0 high, 1 low - gap: zero high-trust coverage
- gardening: 1 high, 4 low - covered
- farming: 2 high, 2 low - covered
- animal_husbandry: 1 high, 1 low - covered
- food_preservation: 1 high, 0 low - covered
- pre_industrial_trades: 0 high, 2 low - gap: zero high-trust coverage

## Chunk Volume By Trust Tier

- high: 910 chunks (40.64%)
- low: 1329 chunks (59.36%)

Recommendation: keep low-trust historical/community material available for recall, but down-weight it during retrieval and cap per-source contribution for oversized low-trust books. In particular, cap or dampen `gutenberg_leather_manufacture`, `gutenberg_manual_gardening`, and the historical USDA foraging source so they cannot dominate answers over modern government and medical material.

## Foraging Safety Coordination

- New foraging source: `usda_food_plants_north_american_indians`.
- Every chunk from this source carries `edibility_claim_review: needs_human_review`.
- Extracted claim-level review records are in `text/reports/foraging_claims_needing_human_review_round2.jsonl`.
- These claims must not be treated as final edibility authority. They require reconciliation with the vision pipeline's species edibility table.

## License Evidence For New Sources

- `us_military_fm_4_25_11_first_aid`: public-domain-usgov evidence: https://commons.wikimedia.org/wiki/File:FM_4-25.11_(FM_21-11)_First_Aid_(Change_1).pdf
- `usda_food_plants_north_american_indians`: public-domain-usgov evidence: https://commons.wikimedia.org/wiki/File:Food_plants_of_the_North_American_Indians_(IA_foodplantsofnort237yano).pdf
- `appropedia_rainwater_harvesting_wayback`: CC-BY-SA-3.0 evidence: https://web.archive.org/web/20240225191326/https://www.appropedia.org/Rainwater_harvesting
- `epa_emergency_disinfection_drinking_water`: public-domain-usgov evidence: https://www.epa.gov/ground-water-and-drinking-water/emergency-disinfection-drinking-water

## Recommended Human Follow-Ups

- Pursue written digital-use permission from Hesperian if the project still wants `Where There Is No Doctor` or related Hesperian texts.
- Review all round-2 foraging claims before any species-level edibility table import.
- Add retrieval-time weighting/capping for low-trust and oversized sources.
- Later pass: address the deferred high-trust gap for `pre_industrial_trades`.
