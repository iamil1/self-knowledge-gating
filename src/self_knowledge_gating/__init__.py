"""Calibrated self-knowledge gating pipeline."""

from .pipeline import SelfKnowledgeGatingPipeline
from .bge_retrieval import BGERetriever, ComposableRetriever
from .retrieval_components import (
    FlagEmbeddingEncoderAdapter,
    FlagEmbeddingRerankerAdapter,
    TransformersEncoderAdapter,
    TransformersRerankerAdapter,
)
from .schemas import (
    Action,
    Answer,
    BoundarySignals,
    BoundaryState,
    Claim,
    Evidence,
    PipelineResult,
    Question,
    ReasonCode,
)

__all__ = [
    "Action",
    "BGERetriever",
    "Answer",
    "BoundarySignals",
    "BoundaryState",
    "Claim",
    "Evidence",
    "PipelineResult",
    "ComposableRetriever",
    "Question",
    "ReasonCode",
    "SelfKnowledgeGatingPipeline",
    "FlagEmbeddingEncoderAdapter",
    "FlagEmbeddingRerankerAdapter",
    "TransformersEncoderAdapter",
    "TransformersRerankerAdapter",
]

__version__ = "0.1.0"
