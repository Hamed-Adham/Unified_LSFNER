"""
3_gating_nn Architecture Package
Gating Neural Network Offline Dataset & Audit Preparation with Dual Parallel LLM Prompts (v1 & v2).
"""

from src.architectures.gating_nn.prompt_templates import (
    load_prompt_template,
    format_inlined_abstract,
    format_entities_evidence_v1,
    format_entities_evidence_v2,
    build_single_call_prompt,
    parse_single_call_llm_response,
    parse_llm_json_response
)
from src.architectures.gating_nn.span_features import (
    run_nms,
    match_gt_label
)
from src.architectures.gating_nn.dataset_builder import (
    GatingDatasetBuilder,
    classify_outcome
)
from src.architectures.gating_nn.pipeline import (
    GatingNNPipeline
)

__all__ = [
    "load_prompt_template",
    "format_inlined_abstract",
    "format_entities_evidence_v1",
    "format_entities_evidence_v2",
    "build_single_call_prompt",
    "parse_single_call_llm_response",
    "parse_llm_json_response",
    "run_nms",
    "match_gt_label",
    "GatingDatasetBuilder",
    "classify_outcome",
    "GatingNNPipeline"
]

