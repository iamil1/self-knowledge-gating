"""Persistent RAG chat server for the gating system (stdlib HTTP only).

Runs on the AutoDL instance at 127.0.0.1:18231; reached from the local
machine through an SSH tunnel. Loads the full pipeline once (retriever index
+ Qwen3-8B) and serves POST /chat {"question": ...}.

Every request is recorded through ProbeRecorder, so casual chat usage also
accumulates interpretability probe data under results/runs/rag-chat/.

Usage: nohup .venv/bin/python scripts/rag_server.py > rag-server.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from self_knowledge_gating.bge_retrieval import ComposableRetriever
from self_knowledge_gating.conflict import RuleBasedConflictAdjudicator
from self_knowledge_gating.llm import QwenAnswerer
from self_knowledge_gating.pipeline import SelfKnowledgeGatingPipeline
from self_knowledge_gating.policy import GatePolicy
from self_knowledge_gating.probes import ProbeRecorder
from self_knowledge_gating.schemas import Evidence, Question
from self_knowledge_gating.signals import ProbeSignalEstimator
from self_knowledge_gating.verification import EmbeddingEvidenceVerifier

ROOT = "/root/autodl-tmp/self-knowledge-gating"
MODELS = f"{ROOT}/models/modelscope/models"
ENCODER = os.environ.get(
    "RAG_ENCODER_PATH",
    f"{ROOT}/models/finetuned/bge-m3-matsci-v1-seed42",
)
RERANKER = os.environ.get(
    "RAG_RERANKER_PATH",
    f"{ROOT}/models/finetuned/bge-reranker-v2-m3-matsci-v1-seed42",
)
ENCODER_TYPE = os.environ.get("RAG_ENCODER_TYPE", "flag_embedding")
RERANKER_TYPE = os.environ.get("RAG_RERANKER_TYPE", "flag_embedding")
ENCODER_PRECISION = os.environ.get("RAG_ENCODER_PRECISION", "auto")
RERANKER_PRECISION = os.environ.get("RAG_RERANKER_PRECISION", "auto")
ENCODER_DEVICE = os.environ.get("RAG_ENCODER_DEVICE", "auto")
RERANKER_DEVICE = os.environ.get("RAG_RERANKER_DEVICE", "auto")
ENCODER_POOLING = os.environ.get("RAG_ENCODER_POOLING", "cls")
ENCODER_QUERY_PREFIX = os.environ.get("RAG_ENCODER_QUERY_PREFIX", "")
USE_FP16 = os.environ.get("RAG_USE_FP16", "true").casefold() in {"1", "true", "yes"}
LLM = f"{MODELS}/Qwen--Qwen3-8B/snapshots/master"
INTERIM = f"{ROOT}/data/interim/openalex_clean.jsonl"
CORPUS_SIZE = 8000
PORT = 18231

_lock = threading.Lock()
_state: dict = {}


def build_pipeline() -> None:
    corpus = []
    with open(INTERIM, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= CORPUS_SIZE:
                break
            r = json.loads(line)
            age = max(0, 2025 - int(r["year"] or 2025))
            corpus.append(Evidence(
                evidence_id=r["work_id"], text=r["abstract"],
                source_type="openalex",
                reliability=min(0.98, 0.55 + (r["cited_by_count"] or 0) / 3000.0),
                freshness=round(max(0.3, 1.0 - 0.07 * age), 3),
                metadata={"venue": r["venue"], "year": r["year"],
                          "title": r["title"]},
            ))
    encoder_options = {"use_fp16": USE_FP16, "name": ENCODER, "query_prefix": ENCODER_QUERY_PREFIX}
    reranker_options = {"use_fp16": USE_FP16, "name": RERANKER}
    if ENCODER_TYPE.casefold().replace("-", "_") in {"transformers", "huggingface", "hf"}:
        encoder_options.update(
            {
                "device": ENCODER_DEVICE,
                "precision": ENCODER_PRECISION,
                "pooling": ENCODER_POOLING,
                "query_prefix": ENCODER_QUERY_PREFIX,
            }
        )
    if RERANKER_TYPE.casefold().replace("-", "_") in {"transformers", "huggingface", "hf", "sequence_classifier"}:
        reranker_options.update({"device": RERANKER_DEVICE, "precision": RERANKER_PRECISION})

    retriever = ComposableRetriever.from_model_paths(
        encoder_path=ENCODER,
        reranker_path=RERANKER,
        encoder_type=ENCODER_TYPE,
        reranker_type=RERANKER_TYPE,
        encoder_options=encoder_options,
        reranker_options=reranker_options,
        corpus=corpus,
        candidate_top_k=int(os.environ.get("RAG_CANDIDATE_TOP_K", "40")),
        final_top_k=int(os.environ.get("RAG_FINAL_TOP_K", "5")),
        encoder_batch_size=int(os.environ.get("RAG_ENCODER_BATCH_SIZE", "32")),
        reranker_batch_size=int(os.environ.get("RAG_RERANKER_BATCH_SIZE", "8")),
        encoder_max_length=int(os.environ.get("RAG_ENCODER_MAX_LENGTH", "512")),
        reranker_max_length=int(os.environ.get("RAG_RERANKER_MAX_LENGTH", "512")),
    )
    answerer = QwenAnswerer(LLM, recorder=None)
    verifier = EmbeddingEvidenceVerifier(ENCODER, threshold=0.55)
    recorder = ProbeRecorder(f"{ROOT}/results/runs/rag-chat")
    probe_cache: dict[str, dict] = {}

    def probe_fn(question: Question) -> dict:
        qid = question.question_id or question.text[:64]
        if qid not in probe_cache:
            p_ik = answerer.probe_self_knowledge(question)
            ent = answerer.probe_semantic_entropy(question)
            p_true = answerer.probe_p_true(question)
            probe_cache[qid] = {"p_ik": p_ik["p_ik"],
                                "semantic_entropy": ent,
                                "p_true": p_true["p_true"]}
            recorder.record(qid, "p_ik", p_ik)
            recorder.record(qid, "semantic_entropy", ent)
            recorder.record(qid, "p_true", p_true)
        return probe_cache[qid]

    pipeline = SelfKnowledgeGatingPipeline(
        retriever=retriever, signal_estimator=ProbeSignalEstimator(probe_fn),
        answerer=answerer, verifier=verifier,
        adjudicator=RuleBasedConflictAdjudicator(), policy=GatePolicy(),
        top_k=5)
    _state.update(pipeline=pipeline, recorder=recorder, verifier=verifier,
                  ready=True)
    # warm the dense index so the first chat request is fast
    retriever._ensure_index()
    print("index warmed", flush=True)


def handle_chat(question_text: str) -> dict:
    if not question_text.strip():
        return {"error": "empty question"}
    qid = f"chat-{int(time.time() * 1000)}"
    question = Question(text=question_text.strip(), question_id=qid)
    pipeline, recorder, verifier = (_state["pipeline"], _state["recorder"],
                                    _state["verifier"])
    with _lock:
        verifier._current_question_id = qid
        result = pipeline.run(question)
    recorder.record(qid, "question_profile", {
        "question_id": qid, "question_text": question.text,
        "source_work_id": None, "venue": None, "year": None})
    recorder.record(qid, "signals", {
        "boundary_probs": {k.value: round(v, 6) for k, v in
                           result.signals.boundary_probs.items()},
        "self_knowledge": result.signals.self_knowledge,
        "answerability": result.signals.answerability,
        "retrieval_utility": result.signals.retrieval_utility,
        "ambiguity": result.signals.ambiguity,
        "access_allowed": result.signals.access_allowed})
    recorder.record(qid, "outcome", {
        "route": result.route.value, "outcome": result.outcome.value,
        "reason_code": result.reason_code.value
        if result.reason_code else None,
        "latency_ms": result.usage.latency_ms,
        "context_tokens": result.usage.context_tokens,
        "output_tokens": result.usage.output_tokens,
        "retrieval_calls": result.usage.retrieval_calls})
    return {
        "question": question.text,
        "route": result.route.value,
        "outcome": result.outcome.value,
        "reason_code": result.reason_code.value if result.reason_code else None,
        "answer": result.answer.text if result.answer else None,
        "boundary_probs": {k.value: round(v, 4) for k, v in
                           result.signals.boundary_probs.items()},
        "self_knowledge": result.signals.self_knowledge,
        "evidence": [{
            "evidence_id": e.evidence_id,
            "title": e.metadata.get("title", ""),
            "venue": e.metadata.get("venue", ""),
            "year": e.metadata.get("year"),
            "reranker_score": e.metadata.get("reranker_score"),
            "snippet": e.text[:280],
        } for e in result.evidence],
        "verification": {
            "evidence_sufficiency": result.verification.evidence_sufficiency,
            "supported_claims": result.verification.supported_claims,
            "total_claims": result.verification.total_claims,
        } if result.verification else None,
        "conflict": {
            "status": result.conflict.status,
            "score": result.conflict.score,
            "winner": result.conflict.winner,
        } if result.conflict else None,
        "latency_ms": round(result.usage.latency_ms, 1),
        "parametric_answer": result.metadata.get("parametric_answer"),
        "grounded_answer": result.metadata.get("grounded_answer"),
        "partial_answer": bool(result.metadata.get("partial_answer", False)),
        "original_verification": result.metadata.get("original_verification"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ready": _state.get("ready", False)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            out = handle_chat(str(req.get("question", "")))
            self._json(200, out)
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": repr(e)})


if __name__ == "__main__":
    print("building pipeline (one-off, ~1 min)...", flush=True)
    build_pipeline()
    print(f"ready on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
