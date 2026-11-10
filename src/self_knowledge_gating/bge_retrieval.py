"""Composable dense retrieval and reranking pipeline.

``ComposableRetriever`` accepts any objects implementing the encoder and
reranker protocols. ``BGERetriever`` remains as a backwards-compatible
convenience wrapper for the default FlagEmbedding BGE pair.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .retrieval_components import (
    DenseEncoder,
    PairReranker,
    build_encoder,
    build_reranker,
)
from .schemas import Evidence, Question


@dataclass
class ComposableRetriever:
    """Retrieve with one encoder and rerank with an independent component."""

    encoder: DenseEncoder
    reranker: PairReranker
    corpus: Sequence[Evidence]
    candidate_top_k: int = 100
    final_top_k: int = 20
    encoder_batch_size: int = 16
    reranker_batch_size: int = 8
    encoder_max_length: int = 8192
    reranker_max_length: int = 512
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.candidate_top_k <= 0 or self.final_top_k <= 0:
            raise ValueError("candidate_top_k and final_top_k must be positive")
        if self.final_top_k > self.candidate_top_k:
            raise ValueError("final_top_k cannot exceed candidate_top_k")
        if self.encoder_batch_size <= 0 or self.reranker_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        self._doc_embeddings: np.ndarray | None = None

    @classmethod
    def from_model_paths(
        cls,
        *,
        encoder_path: str,
        reranker_path: str,
        corpus: Sequence[Evidence],
        encoder_type: str = "flag_embedding",
        reranker_type: str = "flag_embedding",
        encoder_options: dict | None = None,
        reranker_options: dict | None = None,
        **kwargs,
    ) -> "ComposableRetriever":
        encoder = build_encoder(
            encoder_type,
            encoder_path,
            **(encoder_options or {}),
        )
        reranker = build_reranker(
            reranker_type,
            reranker_path,
            **(reranker_options or {}),
        )
        return cls(encoder=encoder, reranker=reranker, corpus=corpus, **kwargs)

    def search(self, question: Question, top_k: int = 5) -> Sequence[Evidence]:
        if not self.corpus:
            return ()
        requested_k = min(top_k, self.final_top_k, len(self.corpus))
        if requested_k <= 0:
            return ()
        self._ensure_index()
        query_embedding = self._encode([question.text], is_query=True)[0]
        dense_scores = self._doc_embeddings @ query_embedding  # type: ignore[operator]
        candidate_k = min(self.candidate_top_k, len(self.corpus))
        candidate_indices = np.argsort(-dense_scores)[:candidate_k]
        candidate_texts = [self.corpus[index].text for index in candidate_indices]
        rerank_scores = self.reranker.score(
            question.text,
            candidate_texts,
            batch_size=self.reranker_batch_size,
            max_length=self.reranker_max_length,
        )
        if len(rerank_scores) != len(candidate_indices):
            raise ValueError(
                "reranker returned a different number of scores than candidate passages"
            )

        ranked = sorted(
            zip(candidate_indices.tolist(), rerank_scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[Evidence] = []
        for index, rerank_score in ranked[:requested_k]:
            evidence = self.corpus[index]
            metadata = dict(evidence.metadata)
            metadata.update(
                {
                    "dense_score": float(dense_scores[index]),
                    "reranker_score": float(rerank_score),
                    "retrieval_encoder": getattr(self.encoder, "name", type(self.encoder).__name__),
                    "retrieval_reranker": getattr(self.reranker, "name", type(self.reranker).__name__),
                }
            )
            results.append(
                replace(
                    evidence,
                    relevance=float(max(0.0, min(1.0, rerank_score))),
                    metadata=metadata,
                )
            )
        return tuple(results)

    def _ensure_index(self) -> None:
        if self._doc_embeddings is None:
            self._doc_embeddings = self._encode([item.text for item in self.corpus])

    def _encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        vectors = self.encoder.encode(
            texts,
            batch_size=self.encoder_batch_size,
            max_length=self.encoder_max_length,
            is_query=is_query,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors


class BGERetriever(ComposableRetriever):
    """Backwards-compatible default BGE-M3 plus BGE reranker adapter."""

    def __init__(
        self,
        encoder_path: str,
        reranker_path: str,
        corpus: Sequence[Evidence],
        candidate_top_k: int = 100,
        final_top_k: int = 20,
        batch_size: int = 16,
        max_length: int = 8192,
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
        reranker_batch_size: int | None = None,
        reranker_max_length: int = 512,
    ) -> None:
        encoder = build_encoder(
            "flag_embedding",
            encoder_path,
            use_fp16=use_fp16,
            normalize_embeddings=normalize_embeddings,
            name=encoder_path,
        )
        reranker = build_reranker(
            "flag_embedding",
            reranker_path,
            use_fp16=use_fp16,
            name=reranker_path,
        )
        super().__init__(
            encoder=encoder,
            reranker=reranker,
            corpus=corpus,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            encoder_batch_size=batch_size,
            reranker_batch_size=reranker_batch_size or batch_size,
            encoder_max_length=max_length,
            reranker_max_length=reranker_max_length,
            normalize_embeddings=normalize_embeddings,
        )
