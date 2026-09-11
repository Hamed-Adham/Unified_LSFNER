"""
Architectures Package for Unified Hybrid ALinkNER.
Contains 5 dedicated subfolders per architecture type:
- hybrid_linkner: SpanNER + Single-Pass LLM Arbitration + LOF Novelty
- dinasor_dingen: Dynamic Generator-Reflector Multi-Agent Pipeline
- gating_nn: Gating Neural Network Offline Dataset & Feature Preparation
- ablation_engine: Systematic Multi-Stage Parameter Sweeps & Pareto Frontier
- agentic_guidelines: Pure LLM Self-Reflective Annotation & Dynamic Guideline Curation
"""

from src.architectures.hybrid_linkner import SpanNERModel, LinkNERPipeline, LOFNoveltyDetector
from src.architectures.dinasor_dingen import Dinasor, DinGenerator, SpanNER_DinGenPipeline
from src.architectures.gating_nn import GatingDatasetBuilder, GatingNNPipeline
from src.architectures.ablation_engine import AblationEngine, PersistentLLMDecisionCache
from src.architectures.agentic_guidelines import (
    Generator,
    Reflector,
    Curator,
    ReflectorCuratorCombined,
    DynamicGuidebookManager,
    DynamicGuidelinesManager,
    GuidebookStore,
    SchemaStore,
    AgenticNERPipeline
)

__all__ = [
    "SpanNERModel",
    "LinkNERPipeline",
    "LOFNoveltyDetector",
    "Dinasor",
    "DinGenerator",
    "SpanNER_DinGenPipeline",
    "GatingDatasetBuilder",
    "GatingNNPipeline",
    "AblationEngine",
    "PersistentLLMDecisionCache",
    "Generator",
    "Reflector",
    "Curator",
    "ReflectorCuratorCombined",
    "DynamicGuidebookManager",
    "DynamicGuidelinesManager",
    "GuidebookStore",
    "SchemaStore",
    "AgenticNERPipeline"
]

