"""Main orchestration entry point for self-knowledge gating."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Mapping

from .conflict import ConflictAdjudicator
from .generation import Answerer
from .policy import GatePolicy
from .retrieval import Retriever
from .schemas import (
    Action,
    Answer,
    BoundarySignals,
    ConflictAssessment,
    Claim,
    Evidence,
    PipelineResult,
    Question,
    ReasonCode,
    ResourceUsage,
    VerificationResult,
)
from .signals import BoundarySignalEstimator
from .verification import EvidenceVerifier


@dataclass
class SelfKnowledgeGatingPipeline:
    """Coordinate estimation, routing, retrieval, verification and adjudication.

    The pipeline keeps the route and the user-visible outcome separate. A
    retrieval route can therefore end in an answer when evidence passes checks,
    or in an abstention when evidence is missing or conflict remains unresolved.
    """

    signal_estimator: BoundarySignalEstimator
    retriever: Retriever
    answerer: Answerer
    verifier: EvidenceVerifier
    adjudicator: ConflictAdjudicator
    policy: GatePolicy
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def run(self, question: Question) -> PipelineResult:
        started = perf_counter()
        signals = self.signal_estimator.estimate(question)
        try:
            route_plan = self.policy.route(signals, question)
        except TypeError:
            # Preserve compatibility with older custom policies.
            route_plan = self.policy.route(signals)
        route = route_plan.action
        base_metadata = {
            "route_reason_code": route_plan.reason_code.value if route_plan.reason_code else None,
            "dominant_boundary_state": signals.dominant_state.value,
            "retrieval_used": False,
        }

        if route == Action.CLARIFY:
            clarification = str(
                question.metadata.get(
                    "clarification_prompt",
                    "请补充会影响答案的关键条件。",
                )
            )
            return self._result(
                question=question,
                route=route,
                outcome=Action.CLARIFY,
                reason_code=route_plan.reason_code,
                signals=signals,
                clarification=clarification,
                started=started,
                metadata=base_metadata,
            )

        if route == Action.ABSTAIN:
            return self._result(
                question=question,
                route=route,
                outcome=Action.ABSTAIN,
                reason_code=route_plan.reason_code,
                signals=signals,
                started=started,
                metadata=base_metadata,
            )

        if route == Action.ANSWER:
            answer = self.answerer.generate(question, ())
            return self._result(
                question=question,
                route=route,
                outcome=Action.ANSWER,
                answer=answer,
                signals=signals,
                started=started,
                metadata=base_metadata,
            )

        parametric_answer = self.answerer.generate(question, ())
        evidence = tuple(self.retriever.search(question, top_k=self.top_k))
        grounded_answer = self.answerer.generate(question, evidence)
        verification = self.verifier.verify(grounded_answer, evidence)
        grounded_answer, verification, revised = self._revise_if_needed(
            question, grounded_answer, evidence, verification
        )
        conflict = self.adjudicator.adjudicate(parametric_answer, grounded_answer, evidence)
        selected_answer = (
            parametric_answer
            if conflict.status == "resolved" and conflict.winner == "parameter"
            else grounded_answer
        )
        selected_verification = (
            self.verifier.verify(selected_answer, evidence)
            if selected_answer is not grounded_answer
            else verification
        )
        final_plan = self.policy.finalize_retrieval(
            signals, evidence, selected_verification, conflict
        )
        metadata = dict(base_metadata)
        metadata["retrieval_used"] = True
        metadata["parametric_answer"] = parametric_answer.text
        metadata["grounded_answer"] = grounded_answer.text
        metadata["grounded_answer_revised"] = revised

        original_verification = selected_verification
        answer = selected_answer if final_plan.action == Action.ANSWER else None
        outcome = final_plan.action
        reason_code = final_plan.reason_code
        if (
            outcome == Action.ABSTAIN
            and reason_code == ReasonCode.INSUFFICIENT_EVIDENCE
            and conflict.status == "none"
            and signals.answerability >= 0.40
        ):
            projected = self._project_supported_claims(
                parametric_answer, selected_answer, evidence, signals
            )
            if projected is not None:
                answer, verification = projected
                outcome = Action.ANSWER
                reason_code = None
                metadata["partial_answer"] = True
                metadata["original_verification"] = {
                    "evidence_sufficiency": original_verification.evidence_sufficiency,
                    "supported_claims": original_verification.supported_claims,
                    "total_claims": original_verification.total_claims,
                    "unsupported_claims": list(original_verification.unsupported_claims),
                }

        return self._result(
            question=question,
            route=route,
            outcome=outcome,
            reason_code=reason_code,
            answer=answer,
            signals=signals,
            evidence=evidence,
            verification=verification,
            conflict=conflict,
            started=started,
            metadata=metadata,
        )

    def _project_supported_claims(self, parametric_answer, answer, evidence, signals):
        """Return a transparent answer containing only verified claim fragments."""
        details = getattr(self.verifier, "last_details", None) or []
        supported_details = [item for item in details if item.get("supported")]
        if not supported_details:
            max_relevance = max((item.relevance for item in evidence), default=0.0)
            if max_relevance >= 0.60 and answer.text.strip():
                return Answer(
                    text=(
                        answer.text
                        + "\n\n检索校验：当前文献相关度较高，但声明级验证未能逐条确认全部内容；请将其作为文献线索核对。"
                    ),
                    claims=answer.claims,
                ), VerificationResult(
                    evidence_sufficiency=0.0,
                    supported_claims=0,
                    total_claims=max(len(answer.claims), 1),
                    unsupported_claims=tuple(
                        claim.key for claim in answer.claims
                    ) or ("unverified_grounded_answer",),
                )
            if (
                signals.answerability >= 0.70
                and parametric_answer.text.strip()
                and not self._contains_quantitative_claim(parametric_answer.text)
            ):
                return Answer(
                    text=(
                        parametric_answer.text
                        + "\n\n检索校验：当前文献未能逐条确认这段回答，以上内容属于模型参数记忆。"
                    ),
                    claims=parametric_answer.claims,
                ), VerificationResult(
                    evidence_sufficiency=0.0,
                    supported_claims=0,
                    total_claims=max(len(parametric_answer.claims), 1),
                    unsupported_claims=tuple(
                        claim.key for claim in parametric_answer.claims
                    ) or ("unverified_parametric_answer",),
                )
            return None
        claims_by_key = {claim.key: claim for claim in answer.claims}
        projected_claims = []
        fragments = []
        for detail in supported_details:
            claim = claims_by_key.get(detail.get("claim_key"))
            if claim is None:
                continue
            evidence_id = detail.get("best_evidence_id")
            projected_claims.append(
                Claim(
                    key=claim.key,
                    value=claim.value,
                    source_refs=(evidence_id,) if evidence_id else claim.source_refs,
                )
            )
            fragments.append(f"{claim.key}: {claim.value}")
        if not projected_claims:
            return None
        evidence_note = (
            "检索校验：当前证据直接支持的是："
            + "；".join(fragments)
            + "。其余参数记忆内容未在当前证据中逐条确认。"
        )
        if signals.answerability >= 0.70 and parametric_answer.text.strip():
            return Answer(
                text=parametric_answer.text + "\n\n" + evidence_note,
                claims=parametric_answer.claims,
            ), VerificationResult(
                evidence_sufficiency=1.0,
                supported_claims=len(projected_claims),
                total_claims=len(parametric_answer.claims) or len(projected_claims),
                unsupported_claims=tuple(
                    claim.key
                    for claim in parametric_answer.claims
                    if claim.key not in {item.key for item in projected_claims}
                ),
            )
        text = (
            "根据当前检索证据，能够直接支持的是："
            + "；".join(fragments)
            + "。其余相关内容未在当前证据中得到直接确认。"
        )
        return Answer(text=text, claims=tuple(projected_claims)), VerificationResult(
            evidence_sufficiency=1.0,
            supported_claims=len(projected_claims),
            total_claims=len(projected_claims),
            unsupported_claims=(),
        )

    @staticmethod
    def _contains_quantitative_claim(text: str) -> bool:
        """Keep unverified numeric/unit claims out of the parametric fallback."""
        # This deliberately catches LaTeX forms such as ``$10^3$`` as well as
        # ordinary values. A false positive is preferable to presenting an
        # unverified scientific number as a grounded answer.
        return bool(re.search(r"\d", text))

    def _revise_if_needed(self, question, answer, evidence, verification):
        """Give answerers with a revision hook one evidence-constrained retry."""
        if verification.evidence_sufficiency >= self.policy.thresholds.evidence_sufficiency:
            return answer, verification, False
        revise = getattr(self.answerer, "revise_grounded", None)
        if revise is None or not evidence:
            return answer, verification, False
        revised_answer = revise(question, answer, evidence, verification)
        revised_verification = self.verifier.verify(revised_answer, evidence)
        if revised_verification.evidence_sufficiency > verification.evidence_sufficiency:
            return revised_answer, revised_verification, True
        return answer, verification, False

    @staticmethod
    def _result(
        *,
        question: Question,
        route: Action,
        outcome: Action,
        started: float,
        reason_code: ReasonCode | None = None,
        answer: Answer | None = None,
        clarification: str | None = None,
        signals: BoundarySignals | None = None,
        evidence: tuple[Evidence, ...] = (),
        verification: VerificationResult | None = None,
        conflict: ConflictAssessment | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        context_tokens = sum(item.context_tokens for item in evidence)
        output_tokens = len(answer.text.split()) if answer is not None else 0
        usage = ResourceUsage(
            retrieval_calls=1 if evidence else 0,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            latency_ms=(perf_counter() - started) * 1000.0,
        )
        return PipelineResult(
            question=question,
            route=route,
            outcome=outcome,
            reason_code=reason_code,
            answer=answer,
            clarification=clarification,
            signals=signals,
            evidence=tuple(evidence),
            verification=verification,
            conflict=conflict,
            usage=usage,
            metadata=metadata or {},
        )
