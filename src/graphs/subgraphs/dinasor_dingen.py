"""
src/graphs/subgraphs/dinasor_dingen.py
LangGraph Subgraph: Reflective Multi-Agent Pipeline (Architecture 2 - Dinasor & DinGenerator).
Orchestrates challenge categorization, anchored sentence masking, contrastive exemplar injection,
Dinasor hint synthesis, DinGenerator reflection, and margin safeguard arbitration.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.graphs.state import UnifiedNERState, CandidateSpan
from src.common.label_mapping import canonicalize_label
from src.architectures.dinasor_dingen.dinasor_agent import Dinasor
from src.architectures.dinasor_dingen.dingen_agent import DinGenerator
from src.architectures.dinasor_dingen.pipeline import (
    classify_challenge_type,
    build_anchored_sentence,
    build_anchored_abstract
)


def enrich_challenges_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 1: Categorizes uncertainty challenges and builds anchored context.
    """
    escalated = state.get("escalated_spans", [])
    confident = state.get("confident_spans", [])
    abstract = state.get("text", "")

    if not escalated or not abstract:
        return {"architecture_metadata": {"skipped": True, "reason": "No escalated spans"}}

    cfg = (config or {}).get("configurable", {})
    use_restricted_top2 = cfg.get("use_restricted_top2", True)
    use_anchored_context = cfg.get("use_anchored_context", True)

    enriched_payload: List[Dict[str, Any]] = []

    for c in escalated:
        c_item = dict(c)
        st = c_item.get("span_text", "")
        top1 = c_item.get("top1_label", c_item.get("spanner_label", "O"))
        top2 = c_item.get("top2_label", "O")
        prob = float(c_item.get("prob", c_item.get("confidence", 0.0)))
        top2_prob = float(c_item.get("top2_prob", 0.0))
        margin = float(c_item.get("margin", 0.0))
        p_o = float(c_item.get("p_o", 0.0))
        novelty = float(c_item.get("novelty", 0.0))
        u_score = float(c_item.get("u_score", 0.0))

        # 1. Classify Challenge Type
        ch_type = classify_challenge_type(
            top1_label=top1,
            top1_prob=prob,
            top2_label=top2,
            top2_prob=top2_prob,
            margin=margin,
            p_o=p_o,
            novelty=novelty,
            u_score=u_score
        )
        c_item["challenge_type"] = ch_type

        # 2. Build Anchored Sentence
        anchored_sent = build_anchored_sentence(abstract, c_item, confident) if use_anchored_context else ""
        c_item["anchored_sentence"] = anchored_sent

        # 3. Restricted candidate labels for tight competition
        if ch_type == "TOP2_COMPETITION" and use_restricted_top2:
            allowed = [top1]
            if top2 and top2.lower() != "o" and top2 != top1:
                allowed.append(top2)
            c_item["allowed_categories"] = allowed
        else:
            c_item["allowed_categories"] = []

        item_dict = {
            "entity": st,
            "challenge_type": ch_type,
            "allowed_categories": c_item["allowed_categories"],
            "spanner_evidence": {
                "predicted_tag": top1,
                "top1": f"{top1} ({prob:.4f})",
                "top2": f"{top2} ({top2_prob:.4f})",
                "margin": f"{margin:.4f}",
                "uncertainty_score": f"{u_score:.4f}",
                "novelty_score": f"{novelty:.4f}",
                "p_o": f"{p_o:.4f}"
            },
            "anchored_sentence": anchored_sent,
            "contrastive_exemplars": c_item.get("contrastive_exemplars", []),
            "raw_candidate": c_item
        }
        enriched_payload.append(item_dict)

    anchored_abstract = build_anchored_abstract(abstract, escalated, confident) if use_anchored_context else abstract

    meta = {
        "enriched_payload": enriched_payload,
        "escalated_count": len(escalated),
        "anchored_abstract": anchored_abstract,
        "architecture": "dinasor_dingen"
    }
    return {"architecture_metadata": meta}


def dinasor_hints_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 2: Runs Dinasor reflection agent to synthesize disambiguation hints.
    """
    meta = dict(state.get("architecture_metadata", {}))
    if meta.get("skipped"):
        return {"architecture_metadata": meta}

    abstract = meta.get("anchored_abstract", state.get("text", ""))
    enriched_payload = meta.get("enriched_payload", [])
    cfg = (config or {}).get("configurable", {})
    dinasor_agent = cfg.get("dinasor_agent")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")

    if dinasor_agent is None:
        try:
            dinasor_agent = Dinasor(model_name=model_name, backend=backend)
        except Exception:
            dinasor_agent = None

    hints = []
    dinasor_prompt = ""
    dinasor_raw = ""
    if dinasor_agent is not None:
        try:
            hints_out = dinasor_agent.run(
                abstract=abstract,
                uncertain_entities=enriched_payload,
                return_prompt=True,
                return_raw=True
            )
            if isinstance(hints_out, tuple):
                if len(hints_out) == 3:
                    hints, dinasor_prompt, dinasor_raw = hints_out
                elif len(hints_out) == 2:
                    hints, dinasor_prompt = hints_out
                else:
                    hints = hints_out[0]
            else:
                hints = hints_out
        except Exception as e:
            print(f"⚠️ [Dinasor Node] Reflection hint error ({e}). Proceeding without hints.")
            hints = []
    else:
        # Mock / Fallback hint generator
        for item in enriched_payload:
            hints.append({
                "target_entity": item["entity"],
                "competing_categories": item.get("allowed_categories", []),
                "resolution_hint": f"Consider context for {item['entity']}."
            })

    meta["hints"] = hints
    meta["dinasor_prompt"] = dinasor_prompt
    meta["dinasor_raw_response"] = dinasor_raw
    return {"architecture_metadata": meta}


def dingen_resolution_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 3: Runs DinGenerator agent to relabel entities based on synthesized hints.
    """
    meta = dict(state.get("architecture_metadata", {}))
    if meta.get("skipped"):
        return {"architecture_metadata": meta}

    abstract = meta.get("anchored_abstract", state.get("text", ""))
    hints = meta.get("hints", [])
    enriched_payload = meta.get("enriched_payload", [])
    cfg = (config or {}).get("configurable", {})
    dingen_agent = cfg.get("dingenerator_agent")
    model_name = cfg.get("model_name", "qwen2.5-7b-instruct")
    backend = cfg.get("backend", "lmstudio")

    if dingen_agent is None:
        try:
            dingen_agent = DinGenerator(model_name=model_name, backend=backend)
        except Exception:
            dingen_agent = None

    dingen_output = {}
    if dingen_agent is not None:
        try:
            dingen_output = dingen_agent.run(
                abstract=abstract,
                hints=hints,
                uncertain_entities=enriched_payload
            )
        except Exception as e:
            print(f"⚠️ [DinGenerator Node] Execution error ({e}). Returning fallback.")
            dingen_output = {"labeled_entities": []}
    else:
        # Mock fallback: adopt top1 labels
        mock_labeled = []
        for item in enriched_payload:
            raw = item["raw_candidate"]
            mock_labeled.append({
                "entity": item["entity"],
                "predicted_tag": raw.get("top1_label", raw.get("spanner_label", "O")),
                "reasoning": "Mock fallback decision"
            })
        dingen_output = {"labeled_entities": mock_labeled}

    meta["dingen_output"] = dingen_output
    meta["dingenerator_prompt"] = dingen_output.get("prompt", "")
    meta["dingenerator_raw_response"] = dingen_output.get("raw_response", "")
    return {"architecture_metadata": meta}


def safeguard_arbitration_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    SubGraph Node 4: Merges DinGenerator labels with margin veto safeguards.
    """
    escalated = state.get("escalated_spans", [])
    if not escalated:
        return {"arbitrated_spans": []}

    meta = state.get("architecture_metadata", {})
    enriched_payload = meta.get("enriched_payload", [])
    dingen_output = meta.get("dingen_output", {})

    cfg = (config or {}).get("configurable", {})
    use_new_labels = cfg.get("use_new_labels", True)
    use_margin_safeguard = cfg.get("use_margin_safeguard", True)
    margin_threshold = cfg.get("margin_threshold", 0.50)

    # Index DinGenerator labeled entities by mention text
    labeled_map = {}
    for ent in dingen_output.get("labeled_entities", []):
        name = ent.get("entity", ent.get("text", "")).strip().lower()
        if name:
            labeled_map[name] = ent

    arbitrated: List[CandidateSpan] = []
    dinasor_prompt = meta.get("dinasor_prompt", "")
    dinasor_raw = meta.get("dinasor_raw_response", "")
    dingen_prompt = meta.get("dingenerator_prompt", "")
    dingen_raw = meta.get("dingenerator_raw_response", "")

    for item in enriched_payload:
        cand = dict(item["raw_candidate"])
        st_norm = cand.get("span_text", "").strip().lower()
        spanner_lbl = canonicalize_label(cand.get("spanner_label", "O"), use_new_labels=use_new_labels)
        margin = float(cand.get("margin", cand.get("prob", 0.0)))
        cand_p_o = float(cand.get("p_o", cand.get("p_background_o", 0.0)))

        match = labeled_map.get(st_norm)
        raw_agent_label = match.get("predicted_tag", match.get("label", spanner_lbl)) if match else spanner_lbl
        
        # Strict Non_LSF -> O canonicalization matching pipeline.py
        if str(raw_agent_label).strip().lower() in ["non_lsf", "non-lsf", "none", "o", "non_lifestyle"]:
            agent_lbl = "O"
        else:
            agent_lbl = canonicalize_label(raw_agent_label, use_new_labels=use_new_labels)

        reason = match.get("reasoning", "") if match else "DinGen arbitration"

        # Stage 4c Margin Safeguard Veto (identical to pipeline.py)
        if use_margin_safeguard and agent_lbl == "O" and margin >= margin_threshold and spanner_lbl.lower() != "o" and cand_p_o < 0.20:
            final_lbl = spanner_lbl
            source = f"SpanNER Margin Veto (Decisive Margin: {margin:.4f} >= {margin_threshold:.2f} & P(O): {cand_p_o:.2f} protected from LLM drop)"
        else:
            final_lbl = agent_lbl
            source = f"DinGenPipe Reflective Resolution ({cand.get('challenge_type', 'Escalation')})"

        cand["arbitrated_label"] = agent_lbl
        cand["final_label"] = final_lbl
        cand["label"] = final_lbl
        cand["llm_label"] = agent_lbl
        cand["source"] = source
        cand["diagnostic_reason"] = reason
        cand["dinasor_prompt"] = dinasor_prompt
        cand["dinasor_raw_response"] = dinasor_raw
        cand["dingenerator_prompt"] = dingen_prompt
        cand["dingenerator_raw_response"] = dingen_raw
        arbitrated.append(cand)

    return {"arbitrated_spans": arbitrated}


def build_dinasor_dingen_subgraph() -> Any:
    """
    Builds and compiles the Dinasor & DinGenerator reflective multi-agent subgraph.
    """
    builder = StateGraph(UnifiedNERState)
    builder.add_node("enrich_challenges", enrich_challenges_node)
    builder.add_node("dinasor_hints", dinasor_hints_node)
    builder.add_node("dingen_resolution", dingen_resolution_node)
    builder.add_node("safeguard_arbitration", safeguard_arbitration_node)

    builder.add_edge(START, "enrich_challenges")
    builder.add_edge("enrich_challenges", "dinasor_hints")
    builder.add_edge("dinasor_hints", "dingen_resolution")
    builder.add_edge("dingen_resolution", "safeguard_arbitration")
    builder.add_edge("safeguard_arbitration", END)

    return builder.compile()


def build_dinasor_pipeline_graph() -> Any:
    """
    Builds and compiles the complete end-to-end SpanNER + DinGen LangGraph Pipeline:
    START -> spanner_inference -> novelty_scoring -> composite_gating -> dinasor_arbitration -> audit_eval -> END
    """
    from src.graphs.nodes.neural_nodes import spanner_inference_node, novelty_scoring_node
    from src.graphs.nodes.gating_nodes import composite_gating_node
    from src.graphs.nodes.audit_nodes import audit_eval_node

    dinasor_subgraph = build_dinasor_dingen_subgraph()

    pipeline_builder = StateGraph(UnifiedNERState)
    pipeline_builder.add_node("spanner_inference", spanner_inference_node)
    pipeline_builder.add_node("novelty_scoring", novelty_scoring_node)
    pipeline_builder.add_node("composite_gating", composite_gating_node)
    pipeline_builder.add_node("dinasor_arbitration", dinasor_subgraph)
    pipeline_builder.add_node("audit_eval", audit_eval_node)

    pipeline_builder.add_edge(START, "spanner_inference")
    pipeline_builder.add_edge("spanner_inference", "novelty_scoring")
    pipeline_builder.add_edge("novelty_scoring", "composite_gating")
    pipeline_builder.add_edge("composite_gating", "dinasor_arbitration")
    pipeline_builder.add_edge("dinasor_arbitration", "audit_eval")
    pipeline_builder.add_edge("audit_eval", END)

    return pipeline_builder.compile()
