"""Probe bus for interpretability research.

Every run of the gating system must persist the raw signals behind each
decision. A probe point is a named, versioned observation slot; the registry
below is the single source of truth for which probes exist, who produces them,
and what shape consumers can rely on. New probes are added by extending
PROBE_POINTS and calling ProbeRecorder.record at the producing site.

Storage layout per run (under results/runs/<run_id>/):
  probes.jsonl   one JSON object per (question_id, point) record
  registry.json  machine-readable copy of PROBE_POINTS
  PROBES.md      human-readable probe contract (auto-generated)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

# point -> spec: producer, when, shape contract, version
PROBE_POINTS: dict[str, dict[str, Any]] = {
    "question_profile": {
        "producer": "run_e2e",
        "when": "before pipeline.run",
        "version": 1,
        "shape": {"question_id": "str", "question_text": "str",
                  "source_work_id": "str|null", "venue": "str|null",
                  "year": "int|null", "expected_boundary_state": "str|null",
                  "access_condition": "str|null", "freshness_condition": "str|null",
                  "matched_group_id": "str|null"},
    },
    "p_ik": {
        "producer": "llm.QwenAnswerer.probe_self_knowledge",
        "when": "pre-routing signal pass",
        "version": 1,
        "shape": {"p_yes": "float", "p_no": "float", "p_ik": "float=p_yes/(p_yes+p_no)",
                  "prompt": "str"},
    },
    "p_true": {
        "producer": "llm.QwenAnswerer.probe_p_true",
        "when": "pre-routing signal pass",
        "version": 1,
        "shape": {"p_true": "float", "greedy_answer": "str", "prompt": "str"},
    },
    "semantic_entropy": {
        "producer": "llm.QwenAnswerer.probe_semantic_entropy",
        "when": "pre-routing signal pass",
        "version": 1,
        "shape": {"entropy": "float", "n_samples": "int",
                  "clusters": "int", "samples": ["str"]},
    },
    "signals": {
        "producer": "signals.ProbeSignalEstimator",
        "when": "inside pipeline.run, before routing",
        "version": 1,
        "shape": {"boundary_probs": "dict[str,float]",
                  "self_knowledge": "float", "answerability": "float",
                  "retrieval_utility": "float", "ambiguity": "float",
                  "access_allowed": "bool"},
    },
    "route": {
        "producer": "policy.GatePolicy.route",
        "when": "inside pipeline.run",
        "version": 1,
        "shape": {"action": "str", "reason_code": "str|null"},
    },
    "retrieval": {
        "producer": "run_e2e (from BGERetriever evidence metadata)",
        "when": "after retrieval inside pipeline.run",
        "version": 1,
        "shape": {"top_k": "int",
                  "evidence": [{"evidence_id": "str", "relevance": "float",
                                "dense_score": "float", "reranker_score": "float",
                                "reliability": "float", "freshness": "float"}]},
    },
    "generation": {
        "producer": "llm.QwenAnswerer.generate",
        "when": "parametric/grounded generation",
        "version": 1,
        "shape": {"mode": "parametric|grounded", "text": "str",
                  "mean_logprob": "float", "min_logprob": "float",
                  "n_tokens": "int", "latency_ms": "float"},
    },
    "grounded_revision": {
        "producer": "llm.QwenAnswerer.revise_grounded",
        "when": "after evidence verification fails and before final gate",
        "version": 1,
        "shape": {"text": "str", "mean_logprob": "float",
                  "min_logprob": "float", "n_tokens": "int",
                  "latency_ms": "float", "removed_claims": ["str"]},
    },
    "claim_extraction": {
        "producer": "llm.QwenAnswerer._extract_claims",
        "when": "after each generation",
        "version": 1,
        "shape": {"mode": "parametric|grounded", "claims": [{"key": "str",
                  "value": "str", "source_refs": ["str"]}], "parse_ok": "bool"},
    },
    "verification": {
        "producer": "pipeline verifier (RuleBasedEvidenceVerifier | EmbeddingEvidenceVerifier)",
        "when": "finalize stage",
        "version": 1,
        "shape": {"evidence_sufficiency": "float", "supported_claims": "int",
                  "total_claims": "int", "unsupported_claims": ["str"]},
    },
    "claim_support_detail": {
        "producer": "verification.EmbeddingEvidenceVerifier",
        "when": "finalize stage, before verification verdict",
        "version": 1,
        "shape": {"threshold": "float",
                  "claims": [{"claim_key": "str", "claim_value": "str",
                              "best_evidence_id": "str",
                              "best_sentence": "str",
                              "similarity": "float", "supported": "bool"}]},
        "note": "per-claim best-evidence cosine; the interpretability core",
    },
    "conflict": {
        "producer": "pipeline (RuleBasedConflictAdjudicator)",
        "when": "finalize stage",
        "version": 1,
        "shape": {"status": "str", "score": "float",
                  "conflicting_claims": ["str"], "winner": "str|null",
                  "reason": "str"},
    },
    "outcome": {
        "producer": "run_e2e",
        "when": "after pipeline.run",
        "version": 1,
        "shape": {"route": "str", "outcome": "str", "reason_code": "str|null",
                  "latency_ms": "float", "context_tokens": "int",
                  "output_tokens": "int", "retrieval_calls": "int"},
    },
    "hidden_states": {
        "producer": "llm.QwenAnswerer (optional, --hidden-states)",
        "when": "generation, forward hook",
        "version": 1,
        "shape": {"mode": "str", "layers": ["int"],
                  "mean_pooled": {"layer": "[float]"}},
        "note": "off by default; stores mean-pooled hidden states per layer",
    },
}


@dataclass
class ProbeRecorder:
    """Append-only JSONL recorder plus registry dump for one run."""

    run_dir: str
    records: int = 0
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self._fh = open(os.path.join(self.run_dir, "probes.jsonl"), "a",
                        encoding="utf-8")
        with open(os.path.join(self.run_dir, "registry.json"), "w",
                  encoding="utf-8") as f:
            json.dump(PROBE_POINTS, f, ensure_ascii=False, indent=2)
        self._write_readme()

    def record(self, question_id: str, point: str,
               payload: Mapping[str, Any]) -> None:
        if point not in PROBE_POINTS:
            raise KeyError(f"unknown probe point: {point}")
        obj = {
            "ts": time.time(),
            "question_id": question_id,
            "point": point,
            "point_version": PROBE_POINTS[point]["version"],
            "payload": payload,
        }
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.records += 1

    def _write_readme(self) -> None:
        lines = ["# Probe contract\n",
                 "Auto-generated from PROBE_POINTS. One JSONL record per "
                 "(question_id, point).\n"]
        for name, spec in PROBE_POINTS.items():
            lines.append(f"## {name} (v{spec['version']})\n")
            lines.append(f"- producer: `{spec['producer']}`\n")
            lines.append(f"- when: {spec['when']}\n")
            lines.append(f"- shape: `{json.dumps(spec['shape'], ensure_ascii=False)}`\n")
            if spec.get("note"):
                lines.append(f"- note: {spec['note']}\n")
            lines.append("\n")
        with open(os.path.join(self.run_dir, "PROBES.md"), "w",
                  encoding="utf-8") as f:
            f.writelines(lines)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
