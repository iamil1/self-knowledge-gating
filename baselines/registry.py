"""Single registry for reproducible baseline names and citations."""

BASELINES = {
    "always-answer": {
        "kind": "routing",
        "description": "closed-book answer on every question",
    },
    "always-rag": {
        "kind": "routing",
        "description": "retrieve on every question, then use the common verifier",
    },
    "uncalibrated-confidence": {
        "kind": "routing",
        "description": "route only on raw self-knowledge confidence",
    },
    "complexity-router": {
        "kind": "routing",
        "status": "alias-to-exploratory",
        "description": "legacy alias; use complexity-heuristic and do not call it Adaptive-RAG reproduced",
    },
    "complexity-heuristic": {
        "kind": "routing",
        "status": "exploratory-only",
        "description": "token-length/metadata heuristic, not the Adaptive-RAG classifier",
    },
    "complexity-classifier-router": {
        "kind": "routing",
        "status": "requires-training",
        "description": "reserved for a trained complexity classifier with a frozen validation protocol",
    },
    "conformal-abstention": {
        "kind": "routing",
        "status": "requires-calibration",
        "description": "abstain from a pre-calibrated risk score",
    },
    "trust-parameter": {
        "kind": "conflict",
        "description": "always prefer parametric memory when claims conflict",
    },
    "trust-retrieval": {
        "kind": "conflict",
        "description": "always prefer retrieved evidence when claims conflict",
    },
    "majority-rule": {
        "kind": "conflict",
        "description": "prefer the most frequent evidence value",
    },
    "claim-level": {
        "kind": "conflict",
        "description": "source-aware claim-level adjudication used by the proposed system",
    },
}
