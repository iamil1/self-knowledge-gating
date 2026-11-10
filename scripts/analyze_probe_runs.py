"""Aggregate probe records by controlled boundary state.

The analyzer accepts a run directory containing ``probes.jsonl`` and a
boundary benchmark JSONL. It reports signal means, action/outcome counts and
simple calibration summaries. It never treats the expected state as a model
input; it is used only after the run for analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATES = ("parametric_known", "uncertain", "no_access", "stale")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def brier(probabilities: list[float], labels: list[int]) -> float | None:
    if not probabilities:
        return None
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def ece(probabilities: list[float], labels: list[int], bins: int = 10) -> float | None:
    if not probabilities:
        return None
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if low <= p < high or (index == bins - 1 and p == high)
        ]
        if selected:
            confidence = sum(p for p, _ in selected) / len(selected)
            accuracy = sum(y for _, y in selected) / len(selected)
            error += len(selected) / total * abs(confidence - accuracy)
    return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    benchmark_path = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark
    output_path = args.output or run_dir / "probe_analysis.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    benchmark = {row["sample_id"]: row for row in load_jsonl(benchmark_path)}
    records = load_jsonl(run_dir / "probes.jsonl")
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[record["question_id"]][record["point"]] = record["payload"]

    state_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for question_id, points in by_question.items():
        item = benchmark.get(question_id)
        if item is None:
            missing.append(question_id)
            continue
        state_rows[item["boundary_state"]].append({"id": question_id, "points": points})

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "benchmark": str(benchmark_path),
        "questions_seen": len(by_question),
        "questions_matched": sum(len(rows) for rows in state_rows.values()),
        "missing_benchmark_rows": missing,
        "by_state": {},
    }
    for state in STATES:
        rows = state_rows.get(state, [])
        p_ik = [float(r["points"]["p_ik"].get("p_ik", 0.5)) for r in rows if "p_ik" in r["points"]]
        p_true = [float(r["points"]["p_true"].get("p_true", 0.5)) for r in rows if "p_true" in r["points"]]
        entropy = [float(r["points"]["semantic_entropy"].get("entropy", 0.5)) for r in rows if "semantic_entropy" in r["points"]]
        outcomes = Counter(r["points"].get("outcome", {}).get("outcome") for r in rows)
        routes = Counter(r["points"].get("route", {}).get("action") for r in rows)
        summary["by_state"][state] = {
            "n": len(rows),
            "mean_p_ik": mean(p_ik),
            "mean_p_true": mean(p_true),
            "mean_semantic_entropy": mean(entropy),
            "routes": dict(routes),
            "outcomes": dict(outcomes),
        }

    known_rows = [r for r in state_rows.get("parametric_known", []) if "p_ik" in r["points"]]
    all_state_rows = [r for state in STATES for r in state_rows.get(state, []) if "p_ik" in r["points"]]
    known_probs = [float(r["points"]["p_ik"].get("p_ik", 0.5)) for r in all_state_rows]
    known_labels = [int(benchmark[r["id"]]["boundary_state"] == "parametric_known") for r in all_state_rows]
    summary["calibration"] = {
        "p_ik_known_brier": brier(known_probs, known_labels),
        "p_ik_known_ece": ece(known_probs, known_labels),
        "known_rows": len(known_rows),
        "note": "These are descriptive summaries; threshold fitting must use a separate validation split.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
