"""Session bookkeeping for the dashboard — plain Python so it is testable without Streamlit."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AnalysisRecord:
    when: str
    filename: str
    predicted_class: str
    confidence: float
    risk: str
    healthy: bool


@dataclass
class SessionHistory:
    records: list[AnalysisRecord] = field(default_factory=list)

    def add(self, filename: str, result: dict[str, Any]) -> AnalysisRecord:
        record = AnalysisRecord(
            when=datetime.now().strftime("%H:%M:%S"),
            filename=filename,
            predicted_class=str(result["predicted_class"]),
            confidence=float(result["confidence"]),
            risk=str(result["risk"]),
            healthy=bool(result.get("healthy", False)),
        )
        self.records.append(record)
        return record

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def latest(self) -> AnalysisRecord | None:
        return self.records[-1] if self.records else None

    @property
    def average_confidence(self) -> float | None:
        if not self.records:
            return None
        return sum(r.confidence for r in self.records) / len(self.records)

    def distribution(self, class_names: list[str]) -> dict[str, int]:
        counts = Counter(r.predicted_class for r in self.records)
        return {name: counts.get(name, 0) for name in class_names}

    def risk_counts(self) -> dict[str, int]:
        counts = Counter("HEALTHY" if r.healthy else r.risk for r in self.records)
        return {k: counts.get(k, 0) for k in ("HEALTHY", "HIGH", "MODERATE", "LOW")}

    def recent(self, n: int = 10) -> list[AnalysisRecord]:
        return list(reversed(self.records[-n:]))
