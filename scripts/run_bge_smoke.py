"""Run a small end-to-end BGE retrieval and gating smoke check."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from self_knowledge_gating.bge_retrieval import ComposableRetriever
from self_knowledge_gating.conflict import RuleBasedConflictAdjudicator
from self_knowledge_gating.generation import MetadataAnswerer
from self_knowledge_gating.pipeline import SelfKnowledgeGatingPipeline
from self_knowledge_gating.policy import GatePolicy
from self_knowledge_gating.schemas import Evidence, Question
from self_knowledge_gating.signals import MetadataSignalEstimator
from self_knowledge_gating.verification import RuleBasedEvidenceVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--encoder-type", default="flag_embedding")
    parser.add_argument("--reranker-type", default="flag_embedding")
    parser.add_argument("--output", type=Path, default=Path("results/runs/bge-smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = (
        Evidence(
            evidence_id="mp-149-band-gap",
            text="Materials Project mp-149 reports the silicon band gap as 1.42 eV at 300 K.",
            source_type="materials_project",
            claim_key="band_gap",
            claim_value="1.42 eV",
            reliability=0.95,
            freshness=0.95,
            relevance=0.0,
            source_version="demo-v1",
        ),
        Evidence(
            evidence_id="recipe-001",
            text="A silicon synthesis recipe uses annealing at 300 C for 60 minutes.",
            source_type="mined_recipe",
            claim_key="annealing_condition",
            claim_value="300 C for 60 minutes",
            reliability=0.80,
            freshness=0.90,
            relevance=0.0,
            source_version="demo-v1",
        ),
        Evidence(
            evidence_id="nomad-mp149",
            text="A NOMAD record links mp-149 to crystalline silicon and a band gap near 1.42 eV.",
            source_type="nomad",
            claim_key="band_gap",
            claim_value="1.42 eV",
            reliability=0.90,
            freshness=0.90,
            relevance=0.0,
            source_version="demo-v1",
        ),
    )
    encoder_options = {"name": args.encoder_path}
    reranker_options = {"name": args.reranker_path}
    if args.encoder_type.casefold().replace("-", "_") in {"flag_embedding", "flag", "bge_m3"}:
        encoder_options["use_fp16"] = True
    else:
        encoder_options.update({"device": "auto", "precision": "auto", "pooling": "cls"})
    if args.reranker_type.casefold().replace("-", "_") in {"flag_embedding", "flag", "bge_reranker"}:
        reranker_options["use_fp16"] = True
    else:
        reranker_options.update({"device": "auto", "precision": "auto"})
    retriever = ComposableRetriever.from_model_paths(
        encoder_path=args.encoder_path,
        reranker_path=args.reranker_path,
        encoder_type=args.encoder_type,
        reranker_type=args.reranker_type,
        encoder_options=encoder_options,
        reranker_options=reranker_options,
        corpus=corpus,
        candidate_top_k=3,
        final_top_k=3,
        batch_size=4,
        max_length=512,
    )
    pipeline = SelfKnowledgeGatingPipeline(
        signal_estimator=MetadataSignalEstimator(),
        retriever=retriever,
        answerer=MetadataAnswerer(),
        verifier=RuleBasedEvidenceVerifier(),
        adjudicator=RuleBasedConflictAdjudicator(),
        policy=GatePolicy(),
        top_k=3,
    )
    result = pipeline.run(
        Question(
            text="What is the Materials Project band gap of mp-149 at 300 K?",
            question_id="bge-smoke",
            metadata={
                "boundary_state": "uncertain",
                "parametric_answer": "The stored value may be 1.10 eV.",
                "parametric_claims": [{"key": "band_gap", "value": "1.10 eV"}],
            },
        )
    )
    payload = {
        "encoder_path": str(args.encoder_path),
        "reranker_path": str(args.reranker_path),
        "route": result.route.value,
        "outcome": result.outcome.value,
        "reason_code": result.reason_code.value if result.reason_code else None,
        "answer": result.answer.text if result.answer else None,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "relevance": item.relevance,
                "metadata": dict(item.metadata),
            }
            for item in result.evidence
        ],
        "verification": asdict(result.verification) if result.verification else None,
        "conflict": asdict(result.conflict) if result.conflict else None,
        "usage": asdict(result.usage),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
