"""
JeBert: Dual-Model Hybrid Named Entity Recognition Architecture.
Combines Dense Neural Span Proposal (SpanNER), Fast Semantic Judgment (TypeSafe Jev),
Dual-Model Gating, and Escalated Heavy LLM Arbitration.
"""

from src.architectures.JeBert.client import TypeSafeClient
from src.architectures.JeBert.gating import JeBertGating
from src.architectures.JeBert.jev_evaluator import JeBertJevEvaluator
from src.architectures.JeBert.arbitrator import JeBertArbitrator
from src.architectures.JeBert.pipeline import JeBertPipeline
from src.architectures.JeBert.jev_generator import JevGenerator

__all__ = [
    "JeBertPipeline",
    "JeBertGating",
    "JeBertJevEvaluator",
    "JeBertArbitrator",
    "TypeSafeClient",
    "JevGenerator",
]
