"""
2_dinasor_dingen Architecture Package
Dynamic Generator-Reflector Multi-Agent Pipeline with Anchored Context Masking and Targeted Disambiguation.
"""

from src.architectures.dinasor_dingen.dinasor_agent import Dinasor
from src.architectures.dinasor_dingen.dingen_agent import DinGenerator
from src.architectures.dinasor_dingen.pipeline import (
    SpanNER_DinGenPipeline,
    run_nms_entities,
    build_anchored_sentence,
    build_anchored_abstract,
    classify_challenge_type
)

__all__ = [
    "Dinasor",
    "DinGenerator",
    "SpanNER_DinGenPipeline",
    "run_nms_entities",
    "build_anchored_sentence",
    "build_anchored_abstract",
    "classify_challenge_type"
]
