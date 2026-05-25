# Fungi Refusal Contract

Status: downstream runtime requirement for v1.

Pocket Crockett v1 does not identify mushrooms or fungi. Fungal targets and images are out of scope for the v1 vision classifier.

Runtime behavior required:

- If the input is suspected to be a mushroom or fungus, return an explicit refusal.
- If the classifier cannot determine whether the subject is an in-scope plant/tree, return `unknown` and include the refusal text.
- Never provide an edibility or safety verdict for a wild mushroom or fungus.

Required refusal text:

> I cannot identify mushrooms or fungi. Never eat a wild mushroom based on this tool.

This is a safety contract handed to the app/runtime stage; this repository pass does not implement runtime code.
