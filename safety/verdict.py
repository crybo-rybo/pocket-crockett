"""Pure safety verdict policy for calibrated vision candidates.

The Jetson runtime should import and call ``decide()`` from this package rather
than reimplementing consumption guidance. Keeping the policy here lets training
reports, tests, and future runtime adapters share the same fail-safe contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safety.edibility import EdibilityTable


FUNGUS_REFUSAL_TEXT = "I cannot identify mushrooms or fungi. Never eat a wild mushroom based on this tool."

STATUS_IDENTIFIED = "identified"
STATUS_REFUSED = "refused"
STATUS_UNKNOWN = "unknown"
GUIDANCE_DO_NOT_EAT = "do_not_eat"
GUIDANCE_EAT = "eat"


@dataclass(frozen=True)
class Candidate:
    scientific_name: str
    calibrated_confidence: float


@dataclass(frozen=True)
class VisionSignal:
    top_n: list[Candidate]
    ood_score: float
    subject_status: str = "plant_candidate"


@dataclass(frozen=True)
class Policy:
    confidence_floor: float
    ood_threshold: float


@dataclass(frozen=True)
class Verdict:
    status: str
    consumption_guidance: str
    candidates: list[Candidate]
    surfaced_lookalikes: list[str] = field(default_factory=list)
    refusal_text: str = ""
    reasons: list[str] = field(default_factory=list)


def _unknown(signal: VisionSignal, reasons: list[str], *, refusal_text: str = "") -> Verdict:
    return Verdict(
        status=STATUS_UNKNOWN,
        consumption_guidance=GUIDANCE_DO_NOT_EAT,
        candidates=signal.top_n,
        refusal_text=refusal_text,
        reasons=reasons,
    )


def _is_eat_eligible(edibility: str, confidence_of_record: str, needs_human_review: bool) -> bool:
    return (
        edibility == "edible"
        and confidence_of_record == "high"
        and not needs_human_review
    )


def decide(signal: VisionSignal, table: EdibilityTable, policy: Policy) -> Verdict:
    if signal.subject_status == "fungus_suspected":
        return Verdict(
            status=STATUS_REFUSED,
            consumption_guidance=GUIDANCE_DO_NOT_EAT,
            candidates=signal.top_n,
            refusal_text=FUNGUS_REFUSAL_TEXT,
            reasons=["fungus_suspected"],
        )

    if signal.subject_status == "in_scope_unknown":
        return _unknown(
            signal,
            ["in_scope_unknown"],
            refusal_text=FUNGUS_REFUSAL_TEXT,
        )

    if not signal.top_n:
        return _unknown(signal, ["no_candidates"])

    top_candidate = signal.top_n[0]
    if signal.ood_score > policy.ood_threshold:
        return _unknown(signal, ["ood_score_above_threshold"])

    if top_candidate.calibrated_confidence < policy.confidence_floor:
        return _unknown(signal, ["confidence_below_floor"])

    record = table.record_for(top_candidate.scientific_name)
    if record is None:
        return Verdict(
            status=STATUS_UNKNOWN,
            consumption_guidance=GUIDANCE_DO_NOT_EAT,
            candidates=signal.top_n,
            reasons=["missing_edibility_record"],
        )

    if _is_eat_eligible(record.edibility, record.confidence_of_record, record.needs_human_review):
        guidance = record.consumption_guidance or GUIDANCE_EAT
        reasons = ["identified", "edibility_record_high_confidence"]
    else:
        guidance = GUIDANCE_DO_NOT_EAT
        reasons = ["identified", "edibility_record_not_eat_eligible"]

    return Verdict(
        status=STATUS_IDENTIFIED,
        consumption_guidance=guidance,
        candidates=signal.top_n,
        surfaced_lookalikes=record.toxic_lookalikes,
        reasons=reasons,
    )
