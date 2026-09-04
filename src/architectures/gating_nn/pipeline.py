"""
Gating Neural Network Architecture Pipeline.
Executes SpanNER candidate span extraction, multi-metric difficulty estimation,
and selective single-pass LLM escalation using Prompt v1 (with Neural Feature Evidence)
or Prompt v2 (blind difficulty evidence).
"""

from __future__ import annotations
import os
import re
import math
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
from src.common.llm_client import get_llm_provider, BaseLLMProvider

from src.architectures.gating_nn.prompt_templates import (
    load_prompt_template,
    format_inlined_abstract,
    format_entities_evidence_v1,
    format_entities_evidence_v2,
    build_single_call_prompt,
    parse_single_call_llm_response
)
from src.architectures.gating_nn.span_features import run_nms
from src.common.label_mapping import (
    CANONICAL_10_LABELS,
    CANONICAL_10_ID2LABEL,
    to_new_label,
    canonicalize_label
)


class GatingNNPipeline:
    """
    Unified Pipeline for the Gating Neural Network Architecture (Prompt v1 / v2 Single-Call).
    
    Features:
    1. Local SpanNER candidate extraction with full difficulty metrics:
       - Top-1 & Top-2 Predicted Classes
       - Confidence & Margin
       - Uncertainty (MCD, Predictive Entropy, Least Confidence)
       - Vector Novelty (LOF, Mahalanobis, Cosine kNN)
       - Background Probability P(O)
    2. Longest-Span First Non-Maximum Suppression (NMS).
    3. Selective Multi-Dimensional Gating Thresholding.
    4. Consolidated Inlined Prompt Escalation (Default: Prompt v1 With Label & Feature Evidence).
    5. Zero Arbitrary Corruption / High-Fidelity JSON Extraction.
    """

    def __init__(
        self,
        spanner_model: Optional[SpanNERModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        novelty_scorer: Optional[Any] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        prompt_version: str = "v1",
        device: Optional[torch.device] = None,
        id2label: Optional[Dict[int, str]] = None,
        use_new_labels: bool = True,
        output_folder: Optional[str] = None
    ):
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        )
        self.spanner_model = spanner_model
        self.tokenizer = tokenizer
        self.novelty_scorer = novelty_scorer
        self.llm_provider = llm_provider or get_default_llm_provider()
        self.prompt_version = prompt_version.lower()
        self.id2label = id2label or CANONICAL_10_ID2LABEL
        self.num_classes = len(self.id2label)
        self.use_new_labels = use_new_labels
        self.output_folder = output_folder

        if self.spanner_model is not None:
            self.spanner_model.to(self.device)
            self.spanner_model.eval()

        # Dynamic model name resolution for logging
        self.model_name = getattr(
            self.llm_provider,
            "model_name",
            getattr(self.llm_provider, "__class__", type("Obj", (), {"__name__": "GatingNN-LLM"})).__name__
        )

    def _resolve_with_gating_llm(
        self,
        text: str,
        uncertain_candidates: List[Dict[str, Any]],
        version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Escalates a list of uncertain candidate mentions for a single abstract
        using the consolidated single-call prompt (Prompt v1 or v2).
        """
        if not uncertain_candidates:
            return []

        v = (version or self.prompt_version).lower()
        prompt = build_single_call_prompt(text, uncertain_candidates, version=v)

        try:
            raw_response = self.llm_provider.generate(prompt)
        except Exception as e:
            print(f"⚠️ [GatingNN Pipeline] LLM generation error: {e}")
            raw_response = "{}"

        parsed_entities = parse_single_call_llm_response(raw_response, uncertain_candidates)

        for cand, parsed in zip(uncertain_candidates, parsed_entities):
            llm_lbl = parsed.get("llm_label", "O")
            rationale = parsed.get("llm_rationale", "")
            pred_tag = parsed.get("llm_predicted_tag", "")

            cand["llm_label"] = llm_lbl
            cand["label"] = llm_lbl
            cand["rationale"] = rationale
            cand["llm_rationale"] = rationale
            cand["llm_predicted_tag"] = pred_tag
            cand["raw_prompt"] = prompt
            cand["raw_response"] = raw_response

        return uncertain_candidates

    def _escalate_and_arbitrate_abstract(
        self,
        text: str,
        cand_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Universal compatibility hook for ablation and sweep engines.
        """
        return self._resolve_with_gating_llm(text, cand_list, version=self.prompt_version)

    def predict_abstract(
        self,
        text: str,
        uncertainty_threshold: float = 0.38,
        margin_threshold: float = 0.50,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        use_margin_safeguard: bool = True,
        mcd_passes: int = 5
    ) -> Dict[str, Any]:
        """
        End-to-end inference pipeline for a single raw abstract string:
        1. Tokenizes and extracts SpanNER candidate mentions and features.
        2. Applies Longest-Span First NMS.
        3. Computes vector novelty.
        4. Applies gating decision: confident mentions keep SpanNER labels,
           uncertain mentions escalate to single-pass LLM with Prompt v1.
        5. Returns structured prediction dictionary.
        """
        if not self.spanner_model or not self.tokenizer:
            raise ValueError("SpanNER model and tokenizer must be initialized to run predict_abstract.")

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].cpu().numpy()
        seq_len = int(attention_mask.sum().item())

        with torch.no_grad():
            out = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"][0]
            candidate_spans = out["candidate_spans"]
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0].cpu().numpy()
            top2_probs = sorted_probs[:, 1].cpu().numpy()
            top1_label_ids = sorted_indices[:, 0].cpu().numpy()
            top2_label_ids = sorted_indices[:, 1].cpu().numpy()
            margins = top1_probs - top2_probs
            p_o = probs[:, 0].cpu().numpy()

        # Uncertainty estimation
        u_method = uncertainty_method.lower()
        if u_method == "mcd":
            self.spanner_model.train()
            mcd_list = []
            with torch.no_grad():
                for _ in range(mcd_passes):
                    m_out = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    mcd_list.append(F.softmax(m_out["logits"][0], dim=-1))
            self.spanner_model.eval()
            mcd_stack = torch.stack(mcd_list, dim=0)
            mcd_entropies = -torch.sum(mcd_stack * torch.log(mcd_stack + 1e-9), dim=-1)
            uncertainties = (torch.mean(mcd_entropies, dim=0) / math.log(self.num_classes)).cpu().numpy()
        elif u_method == "pe":
            raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).cpu().numpy()
            uncertainties = raw_entropy / math.log(self.num_classes)
        elif u_method == "margin":
            uncertainties = 1.0 - margins
        else:  # 'lc'
            uncertainties = 1.0 - top1_probs

        raw_candidates = []
        for s_idx, (start, end, _) in enumerate(candidate_spans):
            if end >= seq_len:
                continue
            lbl_id = int(top1_label_ids[s_idx])
            if lbl_id == 0:
                continue

            char_start = int(offsets[start, 0])
            char_end = int(offsets[end, 1])
            span_text = text[char_start:char_end].strip()
            if not span_text:
                continue

            raw_label = self.id2label[lbl_id]
            spanner_label = to_new_label(raw_label) if self.use_new_labels else raw_label
            top2_raw = self.id2label[int(top2_label_ids[s_idx])]
            second_label = to_new_label(top2_raw) if self.use_new_labels else top2_raw

            raw_candidates.append({
                "span_text": span_text,
                "start_char": char_start,
                "end_char": char_end,
                "spanner_label": spanner_label,
                "predicted_label": spanner_label,
                "second_best_label": second_label,
                "prob": float(top1_probs[s_idx]),
                "top2_prob": float(top2_probs[s_idx]),
                "margin": float(margins[s_idx]),
                "uncertainty": float(uncertainties[s_idx]),
                "u_score": float(uncertainties[s_idx]),
                "p_background_o": float(p_o[s_idx]),
                "prob_o": float(p_o[s_idx])
            })

        # NMS
        nms_spans = run_nms(raw_candidates)

        # Novelty scoring
        if nms_spans and self.novelty_scorer is not None:
            texts_to_embed = [s["span_text"] for s in nms_spans]
            vecs = self.novelty_scorer.embed(texts_to_embed)
            nov_scores = self.novelty_scorer.novelty(vecs)
            for s, nov in zip(nms_spans, nov_scores):
                s["novelty_score"] = float(nov)
                s["novelty"] = float(nov)
        else:
            for s in nms_spans:
                s["novelty_score"] = 0.0
                s["novelty"] = 0.0

        # Gating filter
        uncertain_spans = []
        confident_spans = []
        for s in nms_spans:
            u_val = s["uncertainty"]
            nov_val = s["novelty_score"]

            if gating_mode == "or":
                unreliability = max(u_val, nov_val)
                escalate = (u_val >= uncertainty_threshold) or (nov_val >= 0.55)
            elif gating_mode == "and":
                unreliability = min(u_val, nov_val)
                escalate = (u_val >= uncertainty_threshold) and (nov_val >= 0.55)
            elif gating_mode == "prob_or":
                unreliability = 1.0 - (1.0 - nov_val) * (1.0 - u_val)
                escalate = unreliability >= uncertainty_threshold
            elif gating_mode == "copula":
                unreliability = float(np.clip(u_val + nov_val - 1.5 * (u_val * nov_val), 0.0, 1.0))
                escalate = unreliability >= uncertainty_threshold
            else:  # weighted
                unreliability = (w_novelty * nov_val) + (w_uncertainty * u_val)
                escalate = unreliability >= uncertainty_threshold

            s["unreliability"] = unreliability
            s["escalated_to_llm"] = escalate

            if escalate:
                uncertain_spans.append(s)
            else:
                s["label"] = s["spanner_label"]
                s["llm_label"] = "N/A"
                confident_spans.append(s)

        # Escalate uncertain spans
        if uncertain_spans:
            self._resolve_with_gating_llm(text, uncertain_spans, version=self.prompt_version)
            for u in uncertain_spans:
                llm_lbl = u.get("llm_label", u["spanner_label"])
                if use_margin_safeguard and llm_lbl == "O" and u["margin"] >= margin_threshold and u["spanner_label"] != "O" and u["p_background_o"] < 0.20:
                    u["label"] = u["spanner_label"]
                else:
                    u["label"] = llm_lbl

        final_entities = sorted(confident_spans + uncertain_spans, key=lambda x: x["start_char"])

        return {
            "text": text,
            "entities": final_entities,
            "total_candidates": len(nms_spans),
            "escalated_count": len(uncertain_spans),
            "local_count": len(confident_spans)
        }
