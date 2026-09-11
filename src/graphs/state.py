"""
src/graphs/state.py
Unified state schema for LangGraph-based ALinkNER workflows.
Supports neural inference, vector novelty scoring, composite gating,
multi-architecture arbitration, and 8-outcome diagnostic audit logging.
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Union
from typing_extensions import Annotated
import operator


class CandidateSpan(TypedDict, total=False):
    """
    Structured representation of a biomedical candidate entity span across all stages.
    """
    start_char: int
    end_char: int
    span_text: str
    predicted_tag: str
    spanner_label: str
    confidence: float
    prob: float
    top1_label: str
    top2_label: str
    top2_prob: float
    margin: float
    p_o: float
    u_score: float                  # Uncertainty score in [0, 1] (MCD, PE, Margin, etc.)
    novelty: float                  # Novelty score in [0, 1] (LOF + kNN distance)
    unreliability: float            # Composite gating score: w_n * novelty + w_u * u_score
    challenge_type: str             # e.g., 'TOP2_COMPETITION', 'NOVEL_UNSEEN_TERM', 'CATEGORY_AMBIGUITY'
    allowed_categories: List[str]   # Restrained label set for high-competition ambiguity
    contrastive_exemplars: List[Dict[str, Any]] # Few-shot nearest neighbours from ChromaDB
    anchored_sentence: str          # Masked context string: [ANCHOR: ...] and [TARGET: ...]
    source: str                     # "spanner", "hybrid_linkner", "dinasor", "guidelines"
    arbitrated_label: Optional[str] # Prediction assigned by the Generative LLM
    final_label: str                # Final resolved label adopted after gating & safeguard
    gold_label: Optional[str]       # Ground truth label if available in dataset
    audit_category: Optional[str]   # One of the 10 canonical audit escalation outcomes
    diagnostic_reason: Optional[str]# Diagnostic human-readable explanation


class UnifiedNERState(TypedDict, total=False):
    """
    Central state container passed between nodes and subgraphs in LangGraph.
    """
    # -------------------------------------------------------------------------
    # Input Data Channel
    # -------------------------------------------------------------------------
    abstract_id: str
    text: str
    ground_truth: Optional[List[Dict[str, Any]]]
    
    # -------------------------------------------------------------------------
    # Stage 1: Neural Span Classification (SpanNER)
    # -------------------------------------------------------------------------
    candidate_spans: List[CandidateSpan]
    
    # -------------------------------------------------------------------------
    # Stage 2: Vector Novelty & Exemplars (LOF / kNN)
    # -------------------------------------------------------------------------
    novelty_scored_spans: List[CandidateSpan]
    
    # -------------------------------------------------------------------------
    # Stage 3: Unreliability Gating & Decision Routing
    # -------------------------------------------------------------------------
    confident_spans: List[CandidateSpan]       # Handled directly by local SpanNER (0 LLM cost)
    escalated_spans: List[CandidateSpan]       # Routed to active LLM arbitration architecture
    gating_summary: Dict[str, Any]             # Statistics on total, escalated, accepted counts
    
    # -------------------------------------------------------------------------
    # Stage 4: Dynamic Architecture Arbitration
    # -------------------------------------------------------------------------
    active_architecture: str                   # 'hybrid_linkner' | 'dinasor_dingen' | 'agentic_guidelines'
    arbitrated_spans: List[CandidateSpan]      # Outputs returned by the active escalation subgraph
    architecture_metadata: Dict[str, Any]      # Prompts used, LLM call counts, tokens, reflections
    
    # -------------------------------------------------------------------------
    # Stage 5: Final Aggregation, Audit & Telemetry
    # -------------------------------------------------------------------------
    final_predictions: List[CandidateSpan]     # Merged list (confident + arbitrated)
    audit_records: List[Dict[str, Any]]        # Per-span escalation audit breakdown
    audit_breakdown: Dict[str, Any]            # 10-outcome diagnostic summary table
    metrics: Dict[str, float]                  # Exact/Partial PRF and IAA metrics
