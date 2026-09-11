"""
Cached SpanNER Candidates Loader.

Provides fast, deterministic loading of pre-computed SpanNER candidate spans,
logits, probabilities, uncertainties, and novelty metrics from data/cached/spanner_160_10_class/.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cached" / "spanner_160_10_class"


def get_cache_split_key(dataset_path_or_name: str) -> str:
    """
    Maps dataset directories or split strings to canonical cache split keys:
      '200/train' -> 200Abstracts train and val (160 abstracts)
      '400/train' -> 400Abstracts train (240 abstracts)
      '400/val'   -> 400Abstracts val (80 abstracts)
    """
    s = str(dataset_path_or_name).lower()
    if "400" in s:
        if "val" in s or "test" in s:
            return "400/val"
        return "400/train"
    elif "200" in s:
        return "200/train"
    return "200/train"


def format_candidate_for_pipeline(span_data: Dict[str, Any], doc_id: str = "") -> Dict[str, Any]:
    """
    Standardizes a cached span record to provide both flat legacy/pipeline keys
    and rich nested attributes with 100% backward compatibility.
    """
    b = span_data.get("boundaries", {})
    ev = span_data.get("spanner_evidence", {})
    gt = span_data.get("ground_truth", {})
    nov_b = span_data.get("novelty_breakdown", {})
    unc_m = span_data.get("uncertainty_metrics", {})
    s_text = span_data.get("span_text", "")

    st_c = b.get("start_char", span_data.get("start_char", 0))
    end_c = b.get("end_char", span_data.get("end_char", 0))
    p_lbl = ev.get("predicted_label", span_data.get("predicted_label", "O"))
    s_lbl = ev.get("second_best_label", span_data.get("second_best_label", "O"))
    top1_p = float(ev.get("top1_prob", span_data.get("top1_prob", span_data.get("prob", 0.0))))
    top2_p = float(ev.get("top2_prob", span_data.get("top2_prob", 0.0)))
    unc = float(ev.get("uncertainty", span_data.get("uncertainty", 0.0)))
    nov = float(ev.get("novelty_score", span_data.get("novelty_score", 0.0)))
    margin = float(ev.get("margin", span_data.get("margin", 0.0)))
    p_o = float(ev.get("p_background_o", span_data.get("p_background_o", 0.0)))
    gt_lbl = gt.get("gt_label", span_data.get("gt_label", "O"))

    return {
        # Core text and boundary attributes
        "span_id": span_data.get("span_id", f"{doc_id}_s000"),
        "abstract_id": doc_id or span_data.get("abstract_id", ""),
        "span_text": s_text,
        "entity": s_text,
        "start_char": st_c,
        "end_char": end_c,
        "token_start": b.get("token_start", span_data.get("token_start", 0)),
        "token_end": b.get("token_end", span_data.get("token_end", 0)),
        "token_length": b.get("token_length", span_data.get("token_length", 1)),
        "span_width": b.get("token_length", span_data.get("token_length", 1)),

        # SpanNER Neural Predictions & Evidence
        "predicted_label": p_lbl,
        "spanner_label": p_lbl,
        "second_best_label": s_lbl,
        "top1_prob": top1_p,
        "prob": top1_p,
        "top2_prob": top2_p,
        "margin": margin,
        "p_background_o": p_o,
        "uncertainty": unc,
        "u_score": unc,
        "novelty_score": nov,
        "novelty": nov,
        "gt_label": gt_lbl,

        # Rich multi-signal nested structures
        "boundaries": b,
        "ground_truth": gt,
        "spanner_evidence": ev,
        "novelty_breakdown": nov_b,
        "uncertainty_metrics": unc_m,
        "probabilities": span_data.get("probabilities", {}),
        "logits": span_data.get("logits", {}),
    }


def load_cached_spanner_candidates(
    split_key: str = "200/train",
    cache_dir: Optional[str] = None,
    doc_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Loads candidate spans for a specific split ('200/train', '400/train', '400/val').
    Returns a dict mapping doc_id -> {
        'text': abstract_text,
        'candidates': [formatted_candidate_dict, ...],
        'total_candidates': int
    }
    """
    base_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    split_dir = base_dir / split_key

    if not split_dir.exists():
        raise FileNotFoundError(f"Cache split directory not found: {split_dir}")

    manifest_path = split_dir / "all_candidates.json"
    raw_data: Dict[str, Any] = {}

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        # Fallback: load individual <doc_id>.json files
        for json_file in split_dir.glob("*.json"):
            if json_file.name == "all_candidates.json":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            doc_id = d.get("doc_id", json_file.stem)
            raw_data[doc_id] = d

    filter_set = set(doc_ids) if doc_ids is not None else None
    target_ids = doc_ids if doc_ids is not None else list(raw_data.keys())

    formatted_output = {}
    missing_ids = []
    for doc_id in target_ids:
        if filter_set and doc_id not in filter_set:
            continue
        doc_entry = raw_data.get(doc_id)
        if not doc_entry:
            missing_ids.append(doc_id)
            formatted_output[doc_id] = {
                "text": "",
                "candidates": [],
                "total_candidates": 0
            }
            continue

        text = doc_entry.get("text", "")
        raw_spans = doc_entry.get("spans", [])
        cands = [format_candidate_for_pipeline(s, doc_id=doc_id) for s in raw_spans]

        formatted_output[doc_id] = {
            "text": text,
            "candidates": cands,
            "total_candidates": len(cands)
        }

    if missing_ids:
        print(f"⚠️  [CACHE WARNING] {len(missing_ids)}/{len(target_ids)} requested abstracts were NOT found in cache '{split_key}'!")
        print(f"   (e.g., missing doc_ids: {missing_ids[:5]}...)")
        print(f"   Ensure DATA_DIR matches the cached split (e.g. 'data/new/400Abstracts/val' for '400/val').")

    return formatted_output
