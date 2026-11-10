"""Composable encoder and reranker adapters for retrieval.

The pipeline depends on two small protocols instead of a concrete model
library. Built-in adapters cover FlagEmbedding BGE models and generic
HuggingFace Transformers checkpoints; custom adapters can implement the same
methods and be passed directly to ``ComposableRetriever``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class DenseEncoder(Protocol):
    name: str

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
        is_query: bool = False,
    ) -> np.ndarray:
        """Return one vector per input text."""


class PairReranker(Protocol):
    name: str

    def score(
        self,
        query: str,
        passages: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        """Return one relevance score per query/passage pair."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _device_and_dtype(torch, device: str, precision: str):
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if precision == "auto":
        resolved_precision = "bf16" if resolved_device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    else:
        resolved_precision = precision
    if resolved_precision == "bf16":
        dtype = torch.bfloat16
    elif resolved_precision == "fp16":
        dtype = torch.float16
    elif resolved_precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported precision: {precision}")
    if resolved_device == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    return resolved_device, dtype


def _load_transformers_model(loader, path: str, dtype):
    try:
        return loader.from_pretrained(path, dtype=dtype)
    except TypeError:
        return loader.from_pretrained(path, torch_dtype=dtype)


@dataclass
class FlagEmbeddingEncoderAdapter:
    model_path: str
    use_fp16: bool = True
    normalize_embeddings: bool = True
    query_prefix: str = ""
    name: str | None = None

    def __post_init__(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbeddingEncoderAdapter requires the retrieval extra"
            ) from exc
        self.name = self.name or self.model_path
        self._model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)

    def encode(self, texts: Sequence[str], *, batch_size: int, max_length: int, is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + text for text in texts]
        encoded = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
        )
        vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        return _normalize(vectors) if self.normalize_embeddings else vectors


@dataclass
class TransformersEncoderAdapter:
    model_path: str
    device: str = "auto"
    precision: str = "auto"
    pooling: str = "cls"
    normalize_embeddings: bool = True
    query_prefix: str = ""
    name: str | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = self.name or self.model_path
        self._device, self._dtype = _device_and_dtype(torch, self.device, self.precision)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = _load_transformers_model(AutoModel, self.model_path, self._dtype)
        self._model.to(self._device).eval()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

    def encode(self, texts: Sequence[str], *, batch_size: int, max_length: int, is_query: bool = False) -> np.ndarray:
        import torch

        if is_query and self.query_prefix:
            texts = [self.query_prefix + text for text in texts]
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = pooled.float().cpu().numpy()
            outputs.append(_normalize(vectors) if self.normalize_embeddings else vectors)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


@dataclass
class FlagEmbeddingRerankerAdapter:
    model_path: str
    use_fp16: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbeddingRerankerAdapter requires the retrieval extra"
            ) from exc
        self.name = self.name or self.model_path
        self._model = FlagReranker(self.model_path, use_fp16=self.use_fp16)

    def score(self, query: str, passages: Sequence[str], *, batch_size: int, max_length: int) -> np.ndarray:
        del batch_size, max_length
        pairs = [[query, passage] for passage in passages]
        scores = self._model.compute_score(pairs, normalize=True)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


@dataclass
class TransformersRerankerAdapter:
    model_path: str
    device: str = "auto"
    precision: str = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = self.name or self.model_path
        self._device, self._dtype = _device_and_dtype(torch, self.device, self.precision)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = _load_transformers_model(
            AutoModelForSequenceClassification,
            self.model_path,
            self._dtype,
        )
        self._model.to(self._device).eval()

    def score(self, query: str, passages: Sequence[str], *, batch_size: int, max_length: int) -> np.ndarray:
        import torch

        scores: list[np.ndarray] = []
        for start in range(0, len(passages), batch_size):
            batch = list(passages[start : start + batch_size])
            encoded = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            if logits.shape[-1] == 1:
                values = logits[:, 0]
            else:
                values = logits[:, 1]
            scores.append(values.float().cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)


def build_encoder(kind: str, model_path: str, **kwargs) -> DenseEncoder:
    normalized = kind.casefold().replace("-", "_")
    if normalized in {"flag_embedding", "flag", "bge_m3"}:
        return FlagEmbeddingEncoderAdapter(model_path=model_path, **kwargs)
    if normalized in {"transformers", "huggingface", "hf"}:
        return TransformersEncoderAdapter(model_path=model_path, **kwargs)
    raise ValueError(f"unsupported encoder adapter: {kind}")


def build_reranker(kind: str, model_path: str, **kwargs) -> PairReranker:
    normalized = kind.casefold().replace("-", "_")
    if normalized in {"flag_embedding", "flag", "bge_reranker"}:
        return FlagEmbeddingRerankerAdapter(model_path=model_path, **kwargs)
    if normalized in {"transformers", "huggingface", "hf", "sequence_classifier"}:
        return TransformersRerankerAdapter(model_path=model_path, **kwargs)
    raise ValueError(f"unsupported reranker adapter: {kind}")
