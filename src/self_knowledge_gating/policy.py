"""Gate routing and post-retrieval finalization rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schemas import (
    Action,
    BoundarySignals,
    BoundaryState,
    ConflictAssessment,
    Evidence,
    ReasonCode,
    VerificationResult,
)


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    reason_code: ReasonCode | None = None


@dataclass(frozen=True)
class GateThresholds:
    self_knowledge: float = 0.80
    answerability: float = 0.75
    evidence_sufficiency: float = 0.70
    conflict_score: float = 0.50
    retrieval_utility: float = 0.00
    ambiguity: float = 0.50
    no_access_probability: float = 0.50
    stale_probability: float = 0.50


class GatePolicy:
    """Small auditable policy that can later be replaced by a learned router."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.allow_resolved_conflict = False

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        del question
        if signals.ambiguity >= self.thresholds.ambiguity:
            return PolicyDecision(Action.CLARIFY, ReasonCode.AMBIGUOUS_QUESTION)
        if (
            not signals.access_allowed
            or signals.boundary_probs[BoundaryState.NO_ACCESS] >= self.thresholds.no_access_probability
        ):
            return PolicyDecision(Action.ABSTAIN, ReasonCode.NO_AUTHORIZED_SOURCE)
        if signals.answerability < 0.40:
            return PolicyDecision(Action.ABSTAIN, ReasonCode.HIGH_RISK_UNSUPPORTED)
        if signals.boundary_probs[BoundaryState.STALE] >= self.thresholds.stale_probability:
            return PolicyDecision(Action.RETRIEVE, ReasonCode.STALE_PARAMETRIC_MEMORY)
        if (
            signals.retrieval_utility > self.thresholds.retrieval_utility
            or signals.self_knowledge < self.thresholds.self_knowledge
            or signals.boundary_probs[BoundaryState.UNCERTAIN]
            >= 1.0 - self.thresholds.self_knowledge
        ):
            return PolicyDecision(Action.RETRIEVE, ReasonCode.PARAMETRIC_UNCERTAINTY)
        return PolicyDecision(Action.ANSWER)

    def finalize_retrieval(
        self,
        signals: BoundarySignals,
        evidence: Sequence[Evidence],
        verification: VerificationResult,
        conflict: ConflictAssessment,
    ) -> PolicyDecision:
        if not evidence:
            return PolicyDecision(Action.ABSTAIN, ReasonCode.INSUFFICIENT_EVIDENCE)
        if conflict.status == "unresolved" or (
            conflict.status == "none" and conflict.score >= self.thresholds.conflict_score
        ):
            return PolicyDecision(Action.ABSTAIN, ReasonCode.CONFLICT_UNRESOLVED)
        if conflict.status == "resolved" and self.allow_resolved_conflict:
            return PolicyDecision(Action.ANSWER)
        if verification.evidence_sufficiency < self.thresholds.evidence_sufficiency:
            return PolicyDecision(Action.ABSTAIN, ReasonCode.INSUFFICIENT_EVIDENCE)
        return PolicyDecision(Action.ANSWER)
