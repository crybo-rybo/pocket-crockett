"""Stdlib-only edibility table loader."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EdibilityRecord:
    taxon_id: str
    scientific_name: str
    common_names: list[str]
    edibility: str
    consumption_guidance: str
    preparation_required: str
    toxic_lookalikes: list[str]
    hazard_notes: str
    confidence_of_record: str
    sources: list[dict[str, Any]]
    needs_human_review: bool


def _parse_json_list(value: str, *, field: str, scientific_name: str) -> list[Any]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{field} for {scientific_name!r} must be a JSON list")
    return parsed


def _parse_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _name_key(scientific_name: str) -> str:
    return " ".join(scientific_name.split()).casefold()


class EdibilityTable:
    def __init__(self, records: list[EdibilityRecord]) -> None:
        self.records = records
        self._by_name = {_name_key(record.scientific_name): record for record in records}

    @classmethod
    def load(cls, csv_path: str | Path) -> "EdibilityTable":
        path = Path(csv_path)
        records: list[EdibilityRecord] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scientific_name = row.get("scientific_name", "")
                common_names = _parse_json_list(
                    row.get("common_names", ""),
                    field="common_names",
                    scientific_name=scientific_name,
                )
                toxic_lookalikes = _parse_json_list(
                    row.get("toxic_lookalikes", ""),
                    field="toxic_lookalikes",
                    scientific_name=scientific_name,
                )
                sources = _parse_json_list(
                    row.get("sources", ""),
                    field="sources",
                    scientific_name=scientific_name,
                )
                records.append(
                    EdibilityRecord(
                        taxon_id=row.get("taxon_id", ""),
                        scientific_name=scientific_name,
                        common_names=[str(item) for item in common_names],
                        edibility=row.get("edibility", ""),
                        consumption_guidance=row.get("consumption_guidance", ""),
                        preparation_required=row.get("preparation_required", ""),
                        toxic_lookalikes=[str(item) for item in toxic_lookalikes],
                        hazard_notes=row.get("hazard_notes", ""),
                        confidence_of_record=row.get("confidence_of_record", ""),
                        sources=[item for item in sources if isinstance(item, dict)],
                        needs_human_review=_parse_bool(row.get("needs_human_review", "true")),
                    )
                )
        return cls(records)

    def record_for(self, scientific_name: str) -> EdibilityRecord | None:
        return self._by_name.get(_name_key(scientific_name))

    def lookalike_graph(self, restrict_to: set[str] | None = None) -> dict[str, list[str]]:
        restrict_keys = {_name_key(name): name for name in restrict_to} if restrict_to is not None else None
        graph: dict[str, list[str]] = {}
        for record in self.records:
            source_key = _name_key(record.scientific_name)
            if restrict_keys is not None and source_key not in restrict_keys:
                continue
            targets: list[str] = []
            for lookalike in record.toxic_lookalikes:
                if restrict_keys is not None and _name_key(lookalike) not in restrict_keys:
                    continue
                targets.append(lookalike)
            if targets:
                source_name = restrict_keys[source_key] if restrict_keys is not None else record.scientific_name
                graph[source_name] = targets
        return graph
