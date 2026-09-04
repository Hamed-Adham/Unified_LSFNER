"""
src/graphs/subgraphs/agentic_guidelines.py
LangGraph Subgraph: Self-Reflective Guideline Curation (Architecture 5 - Agentic Guidelines).
Implements the cyclic Generator -> Reflector -> Curator loop.
Extracts entities using dynamic guidelines, identifies errors against ground truth in train/curate mode,
and autonomously synthesizes and prunes domain guideline rules.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.graphs.state import UnifiedNERState, CandidateSpan
from src.common.label_mapping import canonicalize_label
from src.architectures.agentic_guidelines.generator import Generator
from src.architectures.agentic_guidelines.reflector import Reflector
from src.architectures.agentic_guidelines.curator import Curator
from src.architectures.agentic_guidelines.manager import DynamicGuidelinesManager
from src.common.config import FORMATS


def generator_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 1: Runs LLM Generator to predict entities using dynamic guidelines.
    """
    abstract = state.get("text", "")
    if not abstract:
        return {"architecture_metadata": {"skipped": True, "reason": "No abstract text"}}

    cfg = (config or {}).get("configurable", {})
    generator_agent = cfg.get("generator_agent")
    guidelines_manager = cfg.get("guidelines_manager")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")
    file_format = cfg.get("file_format", "BIO")

    if generator_agent is None:
        try:
            generator_agent = Generator(model_name=model_name, file_format=FORMATS.get(file_format, "BIO"), backend=backend)
        except Exception:
            generator_agent = None

    guidelines_md = ""
    if guidelines_manager is not None and hasattr(guidelines_manager, "get_dynamicGuidelines_for_generator_md"):
        guidelines_md = guidelines_manager.get_dynamicGuidelines_for_generator_md()

    predicted_result = {}
    if generator_agent is not None:
        try:
            predicted_result = generator_agent.run(abstract, dynamicGuidelines=guidelines_md)
        except Exception as e:
            print(f"⚠️ [Generator Node] Error ({e}). Using empty prediction.")
            predicted_result = {"labeled_entities": []}
    else:
        predicted_result = {"labeled_entities": []}

    meta = {
        "predicted_result": predicted_result,
        "guidelines_md": guidelines_md,
        "architecture": "agentic_guidelines"
    }
    return {"architecture_metadata": meta}


def reflector_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 2: Runs Reflector to identify discrepancies between predictions and ground truth.
    """
    meta = dict(state.get("architecture_metadata", {}))
    abstract = state.get("text", "")
    ground_truth = state.get("ground_truth", [])
    predicted = meta.get("predicted_result", {})

    cfg = (config or {}).get("configurable", {})
    reflector_agent = cfg.get("reflector_agent")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")
    file_format = cfg.get("file_format", "BIO")

    if reflector_agent is None:
        try:
            reflector_agent = Reflector(model_name=model_name, file_format=FORMATS.get(file_format, "BIO"), backend=backend)
        except Exception:
            reflector_agent = None

    reflections = []
    if reflector_agent is not None and ground_truth:
        try:
            reflections = reflector_agent.run(abstract, predicted, ground_truth)
        except Exception as e:
            print(f"⚠️ [Reflector Node] Error ({e}). Proceeding.")
            reflections = []

    meta["reflections"] = reflections
    return {"architecture_metadata": meta}


def curator_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 3: Synthesizes new guideline rules and prunes conflicting directives.
    """
    meta = dict(state.get("architecture_metadata", {}))
    abstract = state.get("text", "")
    reflections = meta.get("reflections", [])
    guidelines_md = meta.get("guidelines_md", "")

    cfg = (config or {}).get("configurable", {})
    curator_agent = cfg.get("curator_agent")
    guidelines_manager = cfg.get("guidelines_manager")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")
    file_format = cfg.get("file_format", "BIO")

    if curator_agent is None:
        try:
            curator_agent = Curator(model_name=model_name, file_format=FORMATS.get(file_format, "BIO"), backend=backend)
        except Exception:
            curator_agent = None

    curated_updates = {}
    if curator_agent is not None and reflections:
        try:
            curated_updates = curator_agent.run(abstract, reflections, guidelines_md)
            if guidelines_manager is not None and hasattr(guidelines_manager, "update_guidelines"):
                guidelines_manager.update_guidelines(curated_updates)
        except Exception as e:
            print(f"⚠️ [Curator Node] Error ({e}).")
            curated_updates = {}

    meta["curated_updates"] = curated_updates
    return {"architecture_metadata": meta}


def format_agentic_output_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 4: Formats predicted entities into CandidateSpan objects for arbitration.
    """
    meta = state.get("architecture_metadata", {})
    predicted = meta.get("predicted_result", {})
    abstract = state.get("text", "")
    escalated = state.get("escalated_spans", [])

    cfg = (config or {}).get("configurable", {})
    use_new_labels = cfg.get("use_new_labels", True)

    entities = predicted.get("labeled_entities", [])
    arbitrated_spans: List[CandidateSpan] = []

    # If escalated spans exist, align extracted entities to those spans
    if escalated:
        for c in escalated:
            cand = dict(c)
            st_clean = cand.get("span_text", "").strip().lower()
            matched_lbl = None
            matched_reason = "Dynamic guideline resolution"

            for ent in entities:
                ent_name = ent.get("entity", ent.get("text", "")).strip().lower()
                if ent_name == st_clean or st_clean in ent_name:
                    matched_lbl = ent.get("label", ent.get("predicted_tag", ""))
                    matched_reason = ent.get("reasoning", matched_reason)
                    break

            final_l = canonicalize_label(matched_lbl or cand.get("spanner_label", "O"), use_new_labels=use_new_labels)
            cand["arbitrated_label"] = final_l
            cand["final_label"] = final_l
            cand["source"] = "agentic_guidelines"
            cand["diagnostic_reason"] = matched_reason
            arbitrated_spans.append(cand)
    else:
        # Standalone mode: transform extracted entities into candidate spans
        for ent in entities:
            st = ent.get("entity", ent.get("text", ""))
            lbl = canonicalize_label(ent.get("label", ent.get("predicted_tag", "O")), use_new_labels=use_new_labels)
            start_c = abstract.find(st) if st and abstract else 0
            end_c = start_c + len(st) if start_c >= 0 else len(st)
            arbitrated_spans.append({
                "span_text": st,
                "start_char": max(0, start_c),
                "end_char": end_c,
                "final_label": lbl,
                "arbitrated_label": lbl,
                "spanner_label": "O",
                "source": "agentic_guidelines",
                "diagnostic_reason": ent.get("reasoning", "")
            })

    return {"arbitrated_spans": arbitrated_spans}


def should_reflect_and_curate(state: UnifiedNERState, config: Optional[RunnableConfig] = None) -> str:
    """
    Conditional edge: Routes to Reflector only if ground_truth is provided and mode is 'train'/'curate'.
    """
    cfg = (config or {}).get("configurable", {})
    mode = cfg.get("mode", "test").lower()
    gt = state.get("ground_truth")

    if mode in ("train", "curate", "learn") and gt:
        return "reflector"
    return "format_output"


def build_agentic_guidelines_subgraph() -> Any:
    """
    Builds and compiles the cyclic Generator-Reflector-Curator guideline learning subgraph.
    """
    builder = StateGraph(UnifiedNERState)
    builder.add_node("generator", generator_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("curator", curator_node)
    builder.add_node("format_output", format_agentic_output_node)

    builder.add_edge(START, "generator")
    builder.add_conditional_edges(
        "generator",
        should_reflect_and_curate,
        {
            "reflector": "reflector",
            "format_output": "format_output"
        }
    )
    builder.add_edge("reflector", "curator")
    builder.add_edge("curator", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()
