"""
src/graphs/nodes/neural_nodes.py
Functional node adapters for neural span classification (SpanNER)
and vector novelty detection (ChromaDB / LOFNoveltyDetector).
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableConfig

from src.graphs.state import UnifiedNERState, CandidateSpan
from src.common.label_mapping import to_new_label, canonicalize_label
from src.architectures.gating_nn.span_features import run_nms


def compute_span_uncertainties(
    logits: torch.Tensor,
    spanner_model: Optional[torch.nn.Module] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    method: str = "mcd",
    mcd_passes: int = 5,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Computes uncertainty measures across candidate spans.
    Supports 'pe' (Prediction Entropy), 'lc' (Least Confidence),
    'margin' (Probability Margin), and 'mcd' (Monte Carlo Dropout).
    """
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    batch_size, num_spans, n_cls = logits.shape

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    top1_labels = sorted_indices[:, :, 0]
    top2_labels = sorted_indices[:, :, 1]
    p_background_o = probs[:, :, 0]
    margins = top1_probs - top2_probs

    norm_factor = math.log(max(num_classes, n_cls))

    act = method.lower()
    if act == "lc":
        u_scores = 1.0 - top1_probs
    elif act == "margin":
        u_scores = 1.0 - margins
    elif act == "mcd" and spanner_model is not None and input_ids is not None:
        try:
            spanner_model.train()
            mcd_probs_list = []
            with torch.no_grad():
                for _ in range(mcd_passes):
                    out = spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    mcd_probs_list.append(F.softmax(out["logits"], dim=-1))
            spanner_model.eval()
            mcd_stack = torch.stack(mcd_probs_list, dim=0) # [passes, B, spans, C]
            mcd_entropies = -torch.sum(mcd_stack * torch.log(mcd_stack + 1e-9), dim=-1)
            u_scores = torch.mean(mcd_entropies, dim=0) / norm_factor
            mean_mcd_probs = torch.mean(mcd_stack, dim=0)
            sorted_mean, sorted_mean_idx = torch.sort(mean_mcd_probs, dim=-1, descending=True)
            top1_probs = sorted_mean[:, :, 0]
            top2_probs = sorted_mean[:, :, 1]
            top1_labels = sorted_mean_idx[:, :, 0]
            top2_labels = sorted_mean_idx[:, :, 1]
            p_background_o = mean_mcd_probs[:, :, 0]
            margins = top1_probs - top2_probs
        except Exception:
            spanner_model.eval()
            raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            u_scores = raw_entropy / norm_factor
    else: # Default: Prediction Entropy
        raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        u_scores = raw_entropy / norm_factor

    return {
        "u_scores": u_scores,
        "top1_probs": top1_probs,
        "top2_probs": top2_probs,
        "top1_labels": top1_labels,
        "top2_labels": top2_labels,
        "margins": margins,
        "p_o": p_background_o,
    }


def spanner_inference_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    LangGraph Node: Evaluates input text through SpanNER neural classifier.
    Extracts candidate spans, probabilities, uncertainty (MCD/PE/Margin),
    and formats them into standard CandidateSpan structures.
    """
    text = state.get("text", "")
    if not text.strip():
        return {"candidate_spans": []}

    cfg = (config or {}).get("configurable", {})
    spanner_model = cfg.get("spanner_model")
    tokenizer = cfg.get("tokenizer")
    id2label = cfg.get("id2label", {
        0: 'O',
        1: 'Personal_care_products_and_cosmetic_procedures',
        2: 'Substance_use',
        3: 'Environmental_exposures',
        4: 'Mental_health_practices',
        5: 'Non_physical_leisure_time_activities',
        6: 'Nutrition',
        7: 'Physical_activities',
        8: 'Sleep',
        9: 'Socioeconomic_factors'
    })
    device = cfg.get("device", torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    uncertainty_method = cfg.get("uncertainty_method", "margin")
    mcd_passes = cfg.get("mcd_passes", 5)
    use_new_labels = cfg.get("use_new_labels", True)
    apply_nms = cfg.get("apply_nms", False)
    uncertainty_threshold = cfg.get("uncertainty_threshold", 0.35)

    # If already provided or running in mock mode without loaded weights
    if "candidate_spans" in state and state["candidate_spans"]:
        return {"candidate_spans": state["candidate_spans"]}

    if "precomputed_candidates" in state and state["precomputed_candidates"]:
        raw_cands = state["precomputed_candidates"]
        if isinstance(raw_cands, dict) and "candidates" in raw_cands:
            raw_cands = raw_cands["candidates"]
        formatted_cands: List[CandidateSpan] = []
        for c in raw_cands:
            s_text = c.get("span_text", c.get("entity", "")).strip()
            if not s_text:
                continue
            lbl = to_new_label(c.get("predicted_label", c.get("spanner_label", "O"))) if use_new_labels else c.get("predicted_label", c.get("spanner_label", "O"))
            top2_lbl = to_new_label(c.get("second_best_label", c.get("top2_label", "O"))) if use_new_labels else c.get("second_best_label", c.get("top2_label", "O"))
            formatted_cands.append({
                "start_char": int(c.get("start_char", c.get("char_start", 0))),
                "end_char": int(c.get("end_char", c.get("char_end", 0))),
                "span_text": s_text,
                "predicted_tag": lbl,
                "spanner_label": lbl,
                "top1_label": lbl,
                "top2_label": top2_lbl,
                "confidence": float(c.get("prob", c.get("confidence", 1.0))),
                "prob": float(c.get("prob", c.get("confidence", 1.0))),
                "top2_prob": float(c.get("top2_prob", 0.0)),
                "margin": float(c.get("margin", 1.0)),
                "p_o": float(c.get("p_background_o", c.get("p_o", 0.0))),
                "u_score": float(c.get("uncertainty", c.get("u_score", 0.0))),
                "novelty": float(c.get("novelty_score", c.get("novelty", 0.0))),
                "source": "spanner_cache"
            })
        if apply_nms:
            formatted_cands = run_nms(formatted_cands)
        return {"candidate_spans": formatted_cands}

    if spanner_model is None or tokenizer is None:
        # Fallback / Mock candidate generator for lightweight unit tests & decoupled execution
        words = text.split()
        mock_candidates: List[CandidateSpan] = []
        if len(words) > 2:
            start_c = text.find(words[1])
            end_c = start_c + len(words[1]) if start_c >= 0 else 10
            mock_candidates.append({
                "start_char": max(0, start_c),
                "end_char": end_c,
                "span_text": words[1],
                "predicted_tag": "Nutrition",
                "spanner_label": "Nutrition",
                "top1_label": "Nutrition",
                "top2_label": "Environmental_exposures",
                "confidence": 0.85,
                "prob": 0.85,
                "top2_prob": 0.10,
                "margin": 0.75,
                "p_o": 0.05,
                "u_score": 0.25,
                "source": "spanner"
            })
        return {"candidate_spans": mock_candidates}

    # Standard RoBERTa tokenization & SpanNER inference
    encoding = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"].cpu().numpy()

    with torch.no_grad():
        outputs = spanner_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        candidate_span_coords = outputs["candidate_spans"]

    unc_res = compute_span_uncertainties(
        logits=logits,
        spanner_model=spanner_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        method=uncertainty_method,
        mcd_passes=mcd_passes,
        num_classes=len(id2label)
    )

    pred_labels = unc_res["top1_labels"][0].cpu().numpy()
    max_probs = unc_res["top1_probs"][0].cpu().numpy()
    u_scores = unc_res["u_scores"][0].cpu().numpy()
    top2_labels = unc_res["top2_labels"][0].cpu().numpy()
    top2_probs = unc_res["top2_probs"][0].cpu().numpy()
    margins = unc_res["margins"][0].cpu().numpy()
    p_o = unc_res["p_o"][0].cpu().numpy()

    keep_mask = (pred_labels != 0) | (u_scores >= uncertainty_threshold)
    indices = np.where(keep_mask)[0]

    extracted_candidates: List[CandidateSpan] = []
    offsets_abstract = offsets[0]

    for s_idx in indices:
        start_tok, end_tok, _ = candidate_span_coords[s_idx]
        char_start = int(offsets_abstract[start_tok, 0])
        char_end = int(offsets_abstract[end_tok, 1])
        span_text = text[char_start:char_end].strip()
        if not span_text:
            continue

        raw_lbl = id2label.get(int(pred_labels[s_idx]), "O")
        top1_lbl = to_new_label(raw_lbl) if use_new_labels else raw_lbl
        raw_top2 = id2label.get(int(top2_labels[s_idx]), "O")
        top2_lbl = to_new_label(raw_top2) if use_new_labels else raw_top2

        extracted_candidates.append({
            "start_char": char_start,
            "end_char": char_end,
            "span_text": span_text,
            "predicted_tag": top1_lbl,
            "spanner_label": top1_lbl,
            "top1_label": top1_lbl,
            "top2_label": top2_lbl,
            "confidence": float(max_probs[s_idx]),
            "prob": float(max_probs[s_idx]),
            "top2_prob": float(top2_probs[s_idx]),
            "margin": float(margins[s_idx]),
            "p_o": float(p_o[s_idx]),
            "u_score": float(u_scores[s_idx]),
            "source": "spanner"
        })

    if apply_nms and extracted_candidates:
        extracted_candidates = run_nms(extracted_candidates)

    return {"candidate_spans": extracted_candidates}


def novelty_scoring_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    LangGraph Node: Calculates Local Outlier Factor (LOF) and kNN vector distance
    for candidate entity spans against the ChromaDB training vector space.
    """
    candidates = state.get("candidate_spans", [])
    if not candidates:
        return {"novelty_scored_spans": [], "candidate_spans": []}

    # If all candidates already have valid novelty scores (e.g. from precomputed cache), reuse directly
    if all("novelty" in c and c.get("novelty") is not None for c in candidates) and state.get("precomputed_candidates"):
        return {
            "novelty_scored_spans": candidates,
            "candidate_spans": candidates
        }

    cfg = (config or {}).get("configurable", {})
    novelty_scorer = cfg.get("novelty_scorer")

    scored_candidates: List[CandidateSpan] = []

    if novelty_scorer is None:
        # If no vector DB scorer provided, assign default novelty (e.g., 0.10)
        for c in candidates:
            c_copy = dict(c)
            if "novelty" not in c_copy:
                c_copy["novelty"] = 0.10
            scored_candidates.append(c_copy)
        return {
            "novelty_scored_spans": scored_candidates,
            "candidate_spans": scored_candidates
        }

    span_texts = [c["span_text"] for c in candidates]
    if hasattr(novelty_scorer, "embed"):
        span_vecs = novelty_scorer.embed(span_texts)
        novelty_scores = novelty_scorer.novelty(span_vecs)
    else:
        novelty_scores = novelty_scorer.novelty(span_texts)

    for i, c in enumerate(candidates):
        c_copy = dict(c)
        nov = float(novelty_scores[i]) if i < len(novelty_scores) else 0.10
        c_copy["novelty"] = nov

        # Retrieve contrastive exemplars if supported
        if hasattr(novelty_scorer, "get_contrastive_exemplars"):
            target_cls = [c_copy.get("top1_label", "")]
            top2 = c_copy.get("top2_label", "")
            if top2 and top2.lower() != "o" and top2 != target_cls[0]:
                target_cls.append(top2)
            try:
                emb = novelty_scorer.embed([c_copy["span_text"]])[0]
                c_copy["contrastive_exemplars"] = novelty_scorer.get_contrastive_exemplars(
                    emb, target_classes=target_cls, top_k_per_class=1
                )
            except Exception:
                c_copy["contrastive_exemplars"] = []

        scored_candidates.append(c_copy)

    return {
        "novelty_scored_spans": scored_candidates,
        "candidate_spans": scored_candidates
    }
