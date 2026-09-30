"""
Architectures Package for Unified Hybrid ALinkNER.
Contains 5 dedicated subfolders per architecture type:
- hybrid_linkner: SpanNER + Single-Pass LLM Arbitration + LOF Novelty
- dinasor_dingen: Dynamic Generator-Reflector Multi-Agent Pipeline
- gating_nn: Gating Neural Network Offline Dataset & Feature Preparation
- ablation_engine: Systematic Multi-Stage Parameter Sweeps & Pareto Frontier
- agentic_guidelines: Pure LLM Self-Reflective Annotation & Dynamic Guideline Curation
"""

try:
    from src.architectures.hybrid_linkner import SpanNERModel, LinkNERPipeline, LOFNoveltyDetector
except ImportError:
    SpanNERModel = LinkNERPipeline = LOFNoveltyDetector = None

try:
    from src.architectures.dinasor_dingen import Dinasor, DinGenerator, SpanNER_DinGenPipeline
except ImportError:
    Dinasor = DinGenerator = SpanNER_DinGenPipeline = None

try:
    from src.architectures.gating_nn import GatingDatasetBuilder, GatingNNPipeline
except ImportError:
    GatingDatasetBuilder = GatingNNPipeline = None

try:
    from src.architectures.ablation_engine import AblationEngine, PersistentLLMDecisionCache
except ImportError:
    AblationEngine = PersistentLLMDecisionCache = None

try:
    from src.architectures.agentic_guidelines import (
        Generator,
        Reflector,
        Curator,
        ReflectorCuratorCombined,
        DynamicGuidebookManager,
        DynamicGuidelinesManager,
        GuidebookStore,
        SchemaStore,
        AgenticNERPipeline,
        BertizedACEPipeline,
        BertizedGenerator,
        BertizedReflector,
        BertizedCurator
    )
except ImportError:
    Generator = Reflector = Curator = ReflectorCuratorCombined = None
    DynamicGuidebookManager = DynamicGuidelinesManager = None
    GuidebookStore = SchemaStore = AgenticNERPipeline = None
    BertizedACEPipeline = BertizedGenerator = BertizedReflector = BertizedCurator = None

from src.architectures.JeBert import (
    JeBertPipeline,
    JeBertGating,
    JeBertJevEvaluator,
    JeBertArbitrator,
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
    "AgenticNERPipeline",
    "BertizedACEPipeline",
    "BertizedGenerator",
    "BertizedReflector",
    "BertizedCurator",
    "JeBertPipeline",
    "JeBertGating",
    "JeBertJevEvaluator",
    "JeBertArbitrator",
]

