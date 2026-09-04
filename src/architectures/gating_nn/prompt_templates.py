"""
Single Consolidated LLM Prompt Utilities for Gating NN Escalation Preparation.
Loads and dynamically formats prompts from:
  - Single_LLM_Prompts_For_Gating_NN/FINAL_v1_with_label.md
  - Single_LLM_Prompts_For_Gating_NN/FINAL_v2_no_label.md

Provides inlined abstract formatting and robust JSON response parsing with zero label arbitration.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

PROMPT_V1_PATH = os.path.join(PROJECT_ROOT, "src/prompts/gating_nn/FINAL_v1_with_label.md")
PROMPT_V2_PATH = os.path.join(PROJECT_ROOT, "src/prompts/gating_nn/FINAL_v2_no_label.md")

from src.common.label_mapping import canonicalize_label, to_new_label, ALL_KNOWN_LABELS

def load_prompt_template(version: str = "v1") -> str:
    """Load the raw markdown prompt template (v1 with label, v2 without label, or v1 with allowed new spans)."""
    ver = version.lower()
    if ver in ("v1_allowed_new_span", "v1_new_spans", "v1_allowed_new_spans"):
        target_filename = "FINAL_v1_allowed_new_span.md"
    elif ver in ("v1", "v1_with_label"):
        target_filename = "FINAL_v1_with_label.md"
    elif ver in ("v2", "v2_no_label"):
        target_filename = "FINAL_v2_no_label.md"
    else:
        target_filename = f"FINAL_{ver}.md"

    candidate_paths = [
        os.path.join(PROJECT_ROOT, "src/prompts/gating_nn", target_filename),
        os.path.join(PROJECT_ROOT, "src/prompts", target_filename),
        os.path.join(WORKSPACE_ROOT, "Single_LLM_Prompts_For_Gating_NN", target_filename),
        os.path.join(PROJECT_ROOT, f"../Single_LLM_Prompts_For_Gating_NN", target_filename),
    ]
    
    target_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            target_path = p
            break
            
    if not target_path:
        raise FileNotFoundError(f"Prompt template file '{target_filename}' not found in any standard path: {candidate_paths}")

    with open(target_path, "r", encoding="utf-8") as f:
        return f.read()

def format_inlined_abstract(text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Inserts inline square brackets '[span_text]' around detected entity spans.
    Processes spans in reverse order of start_char to prevent character offset corruption.
    """
    if not spans:
        return text

    # Sort spans by start_char descending
    sorted_spans = sorted(spans, key=lambda s: (s.get("start_char", s.get("start", 0)), s.get("end_char", s.get("end", 0))), reverse=True)
    
    # Filter out nested spans to prevent bracket duplication if any overlap remains
    non_overlapping = []
    last_start = float("inf")
    for s in sorted_spans:
        s_start = s.get("start_char", s.get("start", 0))
        s_end = s.get("end_char", s.get("end", 0))
        if s_end <= last_start:
            non_overlapping.append((s_start, s_end))
            last_start = s_start

    inlined_text = text
    for s_start, s_end in non_overlapping:
        inlined_text = inlined_text[:s_start] + "[" + inlined_text[s_start:s_end] + "]" + inlined_text[s_end:]

    return inlined_text

def format_entities_evidence_v1(spans: List[Dict[str, Any]]) -> str:
    """
    Formats detected spans with BERT evidence for Prompt v1 (with predicted label):
    - predicted_label
    - second_best_label
    - uncertainty
    - novelty_score
    - margin
    - p_background_o
    """
    lines = []
    for idx, s in enumerate(spans, 1):
        span_text = s.get("span_text", s.get("text", ""))
        pred_label = s.get("predicted_label", s.get("spanner_label", s.get("label", "O")))
        second_label = s.get("second_best_label", "O")
        u_score = float(s.get("uncertainty", s.get("u_score", 0.0)))
        nov_score = float(s.get("novelty_score", s.get("novelty", 0.0)))
        margin = float(s.get("margin", 0.0))
        p_o = float(s.get("p_background_o", s.get("prob_o", 0.0)))

        lines.append(f'{idx}. "{span_text}":')
        lines.append(f'   - predicted_label: {pred_label}')
        lines.append(f'   - second_best_label: {second_label}')
        lines.append(f'   - uncertainty: {u_score:.4f}')
        lines.append(f'   - novelty_score: {nov_score:.4f}')
        lines.append(f'   - margin: {margin:.4f}')
        lines.append(f'   - p_background_o: {p_o:.4f}')

    return "\n".join(lines)

def format_entities_evidence_v2(spans: List[Dict[str, Any]]) -> str:
    """
    Formats detected spans with difficulty evidence for Prompt v2 (without predicted label):
    - uncertainty
    - novelty_score
    - margin
    - p_background_o
    """
    lines = []
    for idx, s in enumerate(spans, 1):
        span_text = s.get("span_text", s.get("text", ""))
        u_score = float(s.get("uncertainty", s.get("u_score", 0.0)))
        nov_score = float(s.get("novelty_score", s.get("novelty", 0.0)))
        margin = float(s.get("margin", 0.0))
        p_o = float(s.get("p_background_o", s.get("prob_o", 0.0)))

        lines.append(f'{idx}. "{span_text}":')
        lines.append(f'   - uncertainty: {u_score:.4f}')
        lines.append(f'   - novelty_score: {nov_score:.4f}')
        lines.append(f'   - margin: {margin:.4f}')
        lines.append(f'   - p_background_o: {p_o:.4f}')

    return "\n".join(lines)

def build_single_call_prompt(text: str, spans: List[Dict[str, Any]], version: str = "v1") -> str:
    """
    Constructs the full dynamically-filled prompt for v1, v2, or v1_allowed_new_span.
    """
    template = load_prompt_template(version)
    inlined_abstract = format_inlined_abstract(text, spans)
    ver = version.lower()

    if ver in ("v1", "v1_with_label", "v1_allowed_new_span", "v1_new_spans", "v1_allowed_new_spans"):
        entities_str = format_entities_evidence_v1(spans)
        prompt = template.replace("{abstract}", inlined_abstract).replace("{entities_str}", entities_str)
    else:
        span_evidence_str = format_entities_evidence_v2(spans)
        prompt = template.replace("{abstract}", inlined_abstract).replace("{span_evidence_str}", span_evidence_str)

    return prompt

def normalize_llm_label(label: str) -> str:
    """Normalize LLM output label to canonical 10-class name or Non_LSF / O."""
    if not label:
        return "O"
    clean = label.strip().strip("'\"`")
    if clean.lower() in ("o", "non_lsf", "non-lsf", "none", "none of the above", "lifestyle_factor", "lsf_out_of_context"):
        return "O"
    
    # Try direct mapping / canonicalization
    canonical = canonicalize_label(clean, use_new_labels=True)
    if canonical != "O":
        return canonical
    
    # Fuzzy substring matching across known categories
    for cat in ALL_KNOWN_LABELS:
        if cat.lower() in clean.lower():
            return canonicalize_label(cat, use_new_labels=True)
            
    return clean

def parse_single_call_llm_response(
    raw_response: str,
    expected_spans: List[Dict[str, Any]],
    return_new_spans: bool = False
) -> Any:
    """
    Parses strict JSON response from the single LLM call.
    Returns list of dicts with entity, final_label, rationale, raw_llm_label.
    If return_new_spans is True, returns (aligned_results, new_discovered_entities).
    Zero arbitration: directly extracts the LLM's chosen category.
    """
    clean_resp = raw_response.strip()
    # Strip markdown fences if present
    if "```" in clean_resp:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_resp)
        if match:
            clean_resp = match.group(1).strip()

    parsed_entities = []
    new_discovered_entities = []
    try:
        data = json.loads(clean_resp)
        if isinstance(data, dict):
            items = data.get("labeled_entities", [])
            raw_new = data.get("new_discovered_entities", [])
            if isinstance(raw_new, list):
                for n_item in raw_new:
                    if isinstance(n_item, dict):
                        n_ent = n_item.get("entity", "").strip()
                        n_lbl = normalize_llm_label(n_item.get("final_label", n_item.get("label", n_item.get("category", "O"))))
                        n_rat = n_item.get("rationale", n_item.get("reasoning", "")).strip()
                        if n_ent and n_lbl != "O":
                            new_discovered_entities.append({
                                "entity": n_ent,
                                "final_label": n_lbl,
                                "raw_llm_label": n_lbl,
                                "rationale": n_rat,
                                "predicted_tag": "NEW_DISCOVERED"
                            })
        elif isinstance(data, list):
            items = data
        else:
            items = []
            
        for item in items:
            if isinstance(item, dict):
                ent_text = item.get("entity", "").strip()
                raw_label = item.get("final_label", item.get("label", "O"))
                norm_label = normalize_llm_label(raw_label)
                rationale = item.get("rationale", item.get("reasoning", "")).strip()
                pred_tag = item.get("predicted_tag", "")
                parsed_entities.append({
                    "entity": ent_text,
                    "final_label": norm_label,
                    "raw_llm_label": raw_label,
                    "rationale": rationale,
                    "predicted_tag": pred_tag
                })
    except Exception as e:
        # Fallback regex extraction if JSON had trailing commas or small syntax issues
        json_cand_match = re.findall(r'\{\s*"entity":\s*"([^"]+)",[\s\S]*?"final_label":\s*"([^"]+)"(?:,[\s\S]*?"rationale":\s*"([^"]*)")?\s*\}', clean_resp)
        if json_cand_match:
            for ent_text, raw_label, rationale in json_cand_match:
                parsed_entities.append({
                    "entity": ent_text.strip(),
                    "final_label": normalize_llm_label(raw_label),
                    "raw_llm_label": raw_label,
                    "rationale": rationale.strip() if rationale else "",
                    "predicted_tag": ""
                })

    # Align parsed entities to expected_spans
    aligned_results = []
    
    # Index parsed entities by text
    parsed_by_text = {}
    for p in parsed_entities:
        key = p["entity"].lower().strip()
        if key not in parsed_by_text:
            parsed_by_text[key] = []
        parsed_by_text[key].append(p)

    used_indices = set()
    for idx, span in enumerate(expected_spans):
        span_text = span.get("span_text", span.get("text", "")).lower().strip()
        matched_item = None
        if span_text in parsed_by_text and parsed_by_text[span_text]:
            matched_item = parsed_by_text[span_text].pop(0)
        elif idx < len(parsed_entities) and idx not in used_indices:
            # Fallback to positional index
            matched_item = parsed_entities[idx]
            used_indices.add(idx)

        if matched_item:
            aligned_results.append({
                "llm_label": matched_item["final_label"],
                "raw_llm_label": matched_item["raw_llm_label"],
                "llm_rationale": matched_item["rationale"],
                "llm_predicted_tag": matched_item.get("predicted_tag", "")
            })
        else:
            aligned_results.append({
                "llm_label": "O",
                "raw_llm_label": "O",
                "llm_rationale": "Missing in LLM response JSON",
                "llm_predicted_tag": ""
            })

    # If return_new_spans is enabled, collect any remaining unmatched positive entities as new discoveries
    if return_new_spans:
        for remaining_list in parsed_by_text.values():
            for rem_item in remaining_list:
                if rem_item.get("final_label", "O") != "O":
                    # Avoid duplicates
                    if not any(n["entity"].lower() == rem_item["entity"].lower() for n in new_discovered_entities):
                        new_discovered_entities.append(rem_item)
        return aligned_results, new_discovered_entities

    return aligned_results


# Compatibility alias
parse_llm_json_response = parse_single_call_llm_response
