# Pocket Crockett Pipeline A Coverage & Provenance Report

Generated: 2026-05-25

## Summary

- Acquired sources: 13
- Skipped source categories/items: 5
- Total RAG chunks: 1994
- Trust tiers: {'high': 7, 'low': 6}
- OCR/text quality: {'clean': 13}
- Topics with zero high-trust coverage: 1

## Acquired Sources

- `us_army_atp_3_50_21` - ATP 3-50.21 Survival (public-domain-usgov, high, clean, 309 chunks)
- `us_army_fm_21_76` - FM 21-76 US Army Survival Manual (public-domain-usgov, high, clean, 291 chunks)
- `usda_complete_guide_home_canning` - Complete Guide to Home Canning (public-domain-usgov, high, clean, 20 chunks)
- `fema_are_you_ready` - Are You Ready? An In-depth Guide to Citizen Preparedness (public-domain-usgov, high, clean, 38 chunks)
- `usda_nrcs_community_garden_guide` - Community Garden Guide (public-domain-usgov, high, clean, 5 chunks)
- `usda_nrcs_range_pasture_handbook` - National Range and Pasture Handbook (public-domain-usgov, high, clean, 83 chunks)
- `usda_subsistence_farm_gardens` - Subsistence Farm Gardens (public-domain-usgov, low, clean, 104 chunks)
- `usda_poultry_keeping_back_yards` - Poultry Keeping in Back Yards (public-domain-usgov, low, clean, 24 chunks)
- `gutenberg_first_book_farming` - The First Book of Farming (public-domain, low, clean, 137 chunks)
- `gutenberg_manual_gardening` - Manual of Gardening (public-domain, low, clean, 362 chunks)
- `gutenberg_forge_work` - Forge Work (public-domain, low, clean, 110 chunks)
- `gutenberg_leather_manufacture` - The Principles of Leather Manufacture (public-domain, low, clean, 507 chunks)
- `wikibooks_gardening` - Adventist Youth Honors Answer Book - Gardening (CC-BY-SA-3.0, high, clean, 4 chunks)

## Skipped Sources

- Cooperative Extension Service publications / Modern land-grant extension publications: skipped: represented-by-open-usda-sources - Many state/university extension pages are copyrighted or have per-site terms; no single publication was imported without item-level open/public-domain verification.
- Hesperian / Where There Is No Doctor / Hesperian PDFs: skipped: permission-required - Hesperian's open copyright policy says digital use requires written permission; not imported under the no-unverified-license rule.
- Survivor Library / Public-domain survival and trade library: skipped: represented-by-primary-public-domain-sources - Survivor Library is a useful index, but this corpus used primary hosts with explicit item-level public-domain evidence for the initial loop.
- FAO / generic FAO agriculture publications: skipped: license-unverified - FAO's general permissions page says publications are protected by copyright; no specific CC-BY publication was selected in this initial pass.
- Appropedia / Rainwater Harvesting: skipped: download-blocked - The page license was visible in search results as CC-BY-SA-3.0, but Appropedia blocked direct retrieval with a Cloudflare challenge during acquisition.

## Topic Coverage

- shelter: 3 high, 0 low - covered
- water: 3 high, 0 low - covered
- fire: 2 high, 0 low - covered
- navigation: 2 high, 0 low - covered
- first_aid: 3 high, 0 low - covered
- foraging: 2 high, 0 low - covered
- gardening: 2 high, 3 low - covered
- farming: 2 high, 2 low - covered
- animal_husbandry: 1 high, 1 low - covered
- food_preservation: 1 high, 0 low - covered
- pre_industrial_trades: 0 high, 2 low - gap: zero high-trust coverage

## Zero High-Trust Coverage Gaps

- pre_industrial_trades

## License Evidence

- `us_army_atp_3_50_21`: public-domain-usgov evidence: https://armypubs.army.mil/epubs/DR_pubs/DR_a/pdf/web/ARN12086_ATP%203-50x21%20FINAL%20WEB%202.pdf
- `us_army_fm_21_76`: public-domain-usgov evidence: https://archive.org/details/Fm21-76SurvivalManual
- `usda_complete_guide_home_canning`: public-domain-usgov evidence: https://commons.wikimedia.org/wiki/File:Complete_guide_to_home_canning_(IA_CAT30998639).pdf
- `fema_are_you_ready`: public-domain-usgov evidence: https://www.ready.gov/collection/are-you-ready
- `usda_nrcs_community_garden_guide`: public-domain-usgov evidence: https://www.nrcs.usda.gov/plantmaterials/mipmctn11120.pdf
- `usda_nrcs_range_pasture_handbook`: public-domain-usgov evidence: https://directives.nrcs.usda.gov/sites/default/files2/1712930386/33922.pdf
- `usda_subsistence_farm_gardens`: public-domain-usgov evidence: https://commons.wikimedia.org/wiki/File:Subsistence_farm_gardens_(IA_CAT87204756).pdf
- `usda_poultry_keeping_back_yards`: public-domain-usgov evidence: https://commons.wikimedia.org/wiki/File:Poultry_keeping_in_back_yards_(IA_CAT87207022).pdf
- `gutenberg_first_book_farming`: public-domain evidence: https://www.gutenberg.org/ebooks/16900
- `gutenberg_manual_gardening`: public-domain evidence: https://www.gutenberg.org/ebooks/9550
- `gutenberg_forge_work`: public-domain evidence: https://www.gutenberg.org/ebooks/53854
- `gutenberg_leather_manufacture`: public-domain evidence: https://www.gutenberg.org/ebooks/57548
- `wikibooks_gardening`: CC-BY-SA-3.0 evidence: https://en.wikibooks.org/wiki/Adventist_Youth_Honors_Answer_Book/Skill_Level_1_Outdoor_Industries

## Notes

- Hesperian was not imported because its policy requires written permission for digital use.
- FAO was not imported in this pass because no item-specific approved license was selected.
- Survivor Library was treated as an index; initial corpus items were acquired from primary or item-level public-domain hosts instead.
- No source with unknown or unverifiable license was included in the manifest as acquired.
- Nothing flagged `unusable` was included in RAG chunks.
