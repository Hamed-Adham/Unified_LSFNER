"""
hybrid_linkner Architecture Package
Foundational SpanNER neural span classification + Single-pass LLM arbitration + LOF novelty scoring.
"""

from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
from src.architectures.hybrid_linkner.novelty_detector import (
    LOFNoveltyDetector,
    build_lof_novelty_detector,
    get_default_embedder,
    LocalTransformersEmbedder
)
from src.architectures.hybrid_linkner.prompt_templates import (
    build_lsf_linkner_prompt,
    build_lsf_linkner_batch_prompt,
    LSF_CATEGORY_DEFINITIONS,
    get_category_definitions
)
from src.architectures.hybrid_linkner.pipeline import (
    LinkNERPipeline,
    BaseLLMProvider,
    DummyLLMProvider
)

__all__ = [
    "SpanNERModel",
    "LOFNoveltyDetector",
    "build_lof_novelty_detector",
    "get_default_embedder",
    "LocalTransformersEmbedder",
    "build_lsf_linkner_prompt",
    "build_lsf_linkner_batch_prompt",
    "LSF_CATEGORY_DEFINITIONS",
    "get_category_definitions",
    "LinkNERPipeline",
    "BaseLLMProvider",
    "DummyLLMProvider",
]
