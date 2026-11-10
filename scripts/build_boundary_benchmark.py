"""Build a controlled four-state boundary benchmark from held-out pairs.

This script creates matched rows: the same question is repeated under four
explicit experimental conditions. Expected labels are stored as metadata and
must never be passed to the model prompt. The benchmark is a probe/evaluation
fixture, not a human-labeled answer-quality dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATES = ("parametric_known", "uncertain", "no_access", "stale")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/interpretability_probe.yaml"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["run"]["seed"])
    source_path = resolve(config["data"]["source_pairs"])
    output_path = resolve(config["run"]["output_benchmark"])
    rows = [json.loads(line) for line in source_path.open(encoding="utf-8")]
    rng = random.Random(seed)
    n_base = config["run"].get("n_base_questions")
    if n_base is not None and len(rows) > int(n_base):
        rows = rng.sample(rows, int(n_base))
    rows.sort(key=lambda row: row.get("positive_id", ""))

    output = []
    for base_index, row in enumerate(rows):
        question = row["query"]
        group_id = f"boundary-{base_index:05d}-{row['positive_id']}"
        for state in STATES:
            intervention = config["interventions"][state]
            output.append(
                {
                    "sample_id": f"{group_id}-{state}",
                    "matched_group_id": group_id,
                    "base_question_index": base_index,
                    "question": question,
                    "source_work_id": row["positive_id"],
                    "source_title": question,
                    "source_version": "openalex-test-split",
                    "boundary_state": state,
                    "access_condition": intervention["access_condition"],
                    "freshness_condition": intervention["freshness_condition"],
                    "expected_actions": intervention["expected_actions"],
                    "prompt_label_excluded": True,
                    "seed": seed,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "seed": seed,
        "base_questions": len(rows),
        "rows": len(output),
        "states": list(STATES),
        "matched_groups": len(rows),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
