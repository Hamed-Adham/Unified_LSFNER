"""
src/graphs/__init__.py
Unified LangGraph Framework for Hybrid ALinkNER.
"""

from .state import UnifiedNERState, CandidateSpan
from .nodes.neural_nodes import spanner_inference_node, novelty_scoring_node
from .nodes.gating_nodes import composite_gating_node, compute_unreliability_score
from .nodes.audit_nodes import audit_eval_node, compute_prf_metrics
from .subgraphs.hybrid_linkner import build_hybrid_linkner_subgraph
from .subgraphs.dinasor_dingen import build_dinasor_dingen_subgraph, build_dinasor_pipeline_graph
from .subgraphs.agentic_guidelines import build_agentic_guidelines_subgraph

__all__ = [
    "UnifiedNERState",
    "CandidateSpan",
    "spanner_inference_node",
    "novelty_scoring_node",
    "composite_gating_node",
    "compute_unreliability_score",
    "audit_eval_node",
    "compute_prf_metrics",
    "build_hybrid_linkner_subgraph",
    "build_dinasor_dingen_subgraph",
    "build_dinasor_pipeline_graph",
    "build_agentic_guidelines_subgraph",
]
