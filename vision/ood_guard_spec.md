# Open-Set / OOD Guard Requirement

Status: downstream training/runtime requirement for v1.

The v1 output classifier is a closed-set classifier over `vision/splits/output_classes.csv`. It must not force-label out-of-distribution inputs as one of those species.

Required behavior:

- Calibrate confidence using the separate calibration split.
- Show top-N candidates rather than a single forced label.
- Apply a confidence floor before presenting a species ID as actionable.
- Add an OOD mechanism before runtime use: either an explicit background/unknown class, an embedding-distance/OOD detector, or both.
- If the input is below confidence threshold or fails the OOD check, return `unknown`.
- Edibility lookup remains gated separately by the fail-safe skeleton; `unknown` and `do_not_eat` are the defaults.

This is a specification handed to the training/runtime stage; this repository pass does not implement the model or runtime guard.
