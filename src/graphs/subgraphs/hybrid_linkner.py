"""
src/graphs/subgraphs/hybrid_linkner.py
LangGraph Subgraph: Single-Pass LLM Escalation (Architecture 1 - Hybrid LinkNER).
Bundles escalated candidate spans into a single prompt with Definitions & Guidelines,
queries the LLM provider, parses structured responses, and applies the margin safeguard.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.graphs.state import UnifiedNERState, CandidateSpan
from src.common.label_mapping import canonicalize_label
from src.common.llm_client import generate_llm_response
from src.architectures.gating_nn.prompt_templates import (
    build_single_call_prompt,
    parse_single_call_llm_response
)


def format_linkner_prompt_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 1: Prepares consolidated prompt for all escalated spans in the abstract.
    """
    escalated = state.get("escalated_spans", [])
    text = state.get("text", "")
    if not escalated or not text:
        return {"architecture_metadata": {"skipped": True, "reason": "No escalated spans"}}

    cfg = (config or {}).get("configurable", {})
    prompt_version = cfg.get("prompt_version", "v1_with_label")
    ver_key = "v1" if "v1" in prompt_version else "v2"

    prompt = build_single_call_prompt(text, escalated, version=ver_key)
    meta = {
        "prompt": prompt,
        "prompt_version": ver_key,
        "escalated_count": len(escalated),
        "architecture": "hybrid_linkner"
    }
    return {"architecture_metadata": meta}


def call_linkner_llm_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 2: Invokes LLM provider (OpenAI, LMStudio, Ollama, or Dummy).
    """
    meta = dict(state.get("architecture_metadata", {}))
    if meta.get("skipped"):
        return {"architecture_metadata": meta}

    prompt = meta.get("prompt", "")
    cfg = (config or {}).get("configurable", {})
    llm_provider = cfg.get("llm_provider")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")

    raw_response = ""
    try:
        if llm_provider is not None and hasattr(llm_provider, "generate"):
            try:
                raw_response = llm_provider.generate(prompt)
            except TypeError:
                raw_response = llm_provider.generate(prompt, temperature=0.0)
        else:
            raw_response = generate_llm_response(
                prompt=prompt,
                model_name=model_name,
                backend=backend
            ) or "{}"
    except Exception as e:
        print(f"⚠️ [HybridLinkNER Subgraph] LLM call error ({e}). Returning empty response.")
        raw_response = "{}"

    meta["raw_response"] = raw_response
    return {"architecture_metadata": meta}


def parse_linkner_arbitration_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 3: Parses LLM responses, applies margin safeguard against false deletions,
    and constructs arbitrated CandidateSpan entities.
    """
    escalated = state.get("escalated_spans", [])
    if not escalated:
        return {"arbitrated_spans": []}

    meta = state.get("architecture_metadata", {})
    raw_response = meta.get("raw_response", "{}")

    cfg = (config or {}).get("configurable", {})
    use_new_labels = cfg.get("use_new_labels", True)
    use_margin_safeguard = cfg.get("use_margin_safeguard", True)
    margin_threshold = cfg.get("margin_threshold", 0.50)

    parsed_list = parse_single_call_llm_response(raw_response, escalated)
    arbitrated_spans: List[CandidateSpan] = []

    for cand, parsed in zip(escalated, parsed_list):
        cand_item = dict(cand)
        raw_llm_label = parsed.get("llm_label", "O")
        llm_label = canonicalize_label(raw_llm_label, use_new_labels=use_new_labels)
        cand_spanner_label = canonicalize_label(cand_item.get("spanner_label", "O"), use_new_labels=use_new_labels)
        cand_margin = float(cand_item.get("margin", cand_item.get("prob", 0.0)))
        rationale = parsed.get("llm_rationale", "")

        # Margin Safeguard Veto: High neural margin prevents LLM deletion into 'O'
        if use_margin_safeguard and llm_label == "O" and cand_margin >= margin_threshold and cand_spanner_label.lower() != "o":
            final_label = cand_spanner_label
            decision_source = "spanner_margin_veto"
        else:
            final_label = llm_label
            decision_source = "hybrid_linkner_llm"

        cand_item["arbitrated_label"] = llm_label
        cand_item["final_label"] = final_label
        cand_item["source"] = decision_source
        cand_item["diagnostic_reason"] = rationale or raw_llm_label
        arbitrated_spans.append(cand_item)

    return {"arbitrated_spans": arbitrated_spans}


def build_hybrid_linkner_subgraph() -> Any:
    """
    Builds and compiles the Hybrid LinkNER single-pass escalation subgraph.
    """
    builder = StateGraph(UnifiedNERState)
    builder.add_node("format_prompt", format_linkner_prompt_node)
    builder.add_node("call_llm", call_linkner_llm_node)
    builder.add_node("parse_arbitration", parse_linkner_arbitration_node)

    builder.add_edge(START, "format_prompt")
    builder.add_edge("format_prompt", "call_llm")
    builder.add_edge("call_llm", "parse_arbitration")
    builder.add_edge("parse_arbitration", END)

    return builder.compile()
