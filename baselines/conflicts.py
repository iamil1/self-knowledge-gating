"""Fixed-trust and majority conflict baselines."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from self_knowledge_gating.schemas import (
    Answer,
    ConflictAssessment,
    Evidence,
    claims_by_key,
)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


class FixedTrustAdjudicator:
    def __init__(self, winner: str):
        if winner not in {"parameter", "retrieval"}:
            raise ValueError("winner must be parameter or retrieval")
        self.winner = winner

    def adjudicate(self, parametric_answer: Answer, grounded_answer: Answer,
                   evidence: Sequence[Evidence]) -> ConflictAssessment:
        parameter = claims_by_key(parametric_answer.claims)
        grounded = claims_by_key(grounded_answer.claims)
        conflicts = []
        for key in set(parameter) & set(grounded):
            if {_norm(c.value) for c in parameter[key]}.isdisjoint(
                {_norm(c.value) for c in grounded[key]}
            ):
                conflicts.append(key)
        if not conflicts:
            return ConflictAssessment(reason="no conflicting claims detected")
        return ConflictAssessment(
            status="resolved",
            score=1.0,
            conflicting_claims=tuple(sorted(conflicts)),
            winner=self.winner,
            reason=f"fixed trust baseline selected {self.winner}",
        )


class MajorityRuleAdjudicator:
    def adjudicate(self, parametric_answer: Answer, grounded_answer: Answer,
                   evidence: Sequence[Evidence]) -> ConflictAssessment:
        parameter = claims_by_key(parametric_answer.claims)
        grounded = claims_by_key(grounded_answer.claims)
        evidence_values: dict[str, list[str]] = {}
        for item in evidence:
            if item.claim_key and item.claim_value is not None:
                evidence_values.setdefault(item.claim_key, []).append(_norm(item.claim_value))
        conflicts = []
        for key in set(parameter) & set(grounded):
            if {_norm(c.value) for c in parameter[key]}.isdisjoint(
                {_norm(c.value) for c in grounded[key]}
            ):
                conflicts.append(key)
        if not conflicts:
            return ConflictAssessment(reason="no conflicting claims detected")
        winners = []
        for key in conflicts:
            counts = Counter(evidence_values.get(key, []))
            if not counts:
                return ConflictAssessment(
                    status="unresolved",
                    score=1.0,
                    conflicting_claims=tuple(sorted(conflicts)),
                    reason="majority baseline had no evidence vote",
                )
            top = counts.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                return ConflictAssessment(
                    status="unresolved",
                    score=1.0,
                    conflicting_claims=tuple(sorted(conflicts)),
                    reason="majority baseline encountered a tie",
                )
            winners.append(top[0][0])
        return ConflictAssessment(
            status="resolved",
            score=0.5,
            conflicting_claims=tuple(sorted(conflicts)),
            winner="retrieval",
            reason="majority baseline selected the most frequent evidence value",
        )


def build_conflict_adjudicator(name: str):
    normalized = name.casefold().replace("_", "-")
    if normalized in {"claim-level", "claim-adjudication", "production"}:
        from self_knowledge_gating.conflict import RuleBasedConflictAdjudicator

        return RuleBasedConflictAdjudicator()
    if normalized == "trust-parameter":
        return FixedTrustAdjudicator("parameter")
    if normalized == "trust-retrieval":
        return FixedTrustAdjudicator("retrieval")
    if normalized in {"majority-rule", "majority"}:
        return MajorityRuleAdjudicator()
    raise ValueError(f"unknown conflict baseline: {name}")
