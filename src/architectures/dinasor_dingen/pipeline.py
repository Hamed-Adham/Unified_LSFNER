import os
import re
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

try:
    from src.architectures.dinasor_dingen.dinasor_agent import Dinasor
except ImportError:
    from .dinasor_agent import Dinasor
try:
    from src.architectures.dinasor_dingen.dingen_agent import DinGenerator
except ImportError:
    from .dingen_agent import DinGenerator
from src.common.label_mapping import canonicalize_label, to_new_label


def normalize_text(text: str) -> str:
    return text.strip().lower()


def run_nms_entities(entities: list) -> list:
    """
    Greedy Non-Maximum Suppression (NMS) for overlapping entities.
    Prioritizes highest confidence probability and longer span length.
    """
    sorted_ents = sorted(entities, key=lambda x: (x.get("prob", 0.0), x["end_char"] - x["start_char"]), reverse=True)
    kept = []
    for s in sorted_ents:
        overlap = False
        for k in kept:
            if not (s["end_char"] <= k["start_char"] or s["start_char"] >= k["end_char"]):
                overlap = True
                break
        if not overlap:
            kept.append(s)
    return sorted(kept, key=lambda x: x["start_char"])


def build_anchored_sentence(abstract: str, target_span: dict, anchor_spans: list) -> str:
    """
    Finds the sentence containing target_span and masks both the target and confident anchor entities.
    """
    t_start = target_span["start_char"]
    t_end = target_span["end_char"]
    t_text = target_span["span_text"]

    # Locate sentence start and end using punctuation followed by space
    s_start = 0
    for match in re.finditer(r'(?<=[.!?])\s+', abstract):
        if match.start() <= t_start:
            s_start = match.end()
        else:
            break

    s_end = len(abstract)
    for match in re.finditer(r'(?<=[.!?])\s+', abstract):
        if match.start() >= t_end:
            s_end = match.start()
            break

    sentence_raw = abstract[s_start:s_end]
    rel_t_start = t_start - s_start
    rel_t_end = t_end - s_start

    replacements = [
        {"start": rel_t_start, "end": rel_t_end, "replacement": f"[TARGET: {t_text}]"}
    ]

    for anc in anchor_spans:
        a_start = anc.get("start_char", 0)
        a_end = anc.get("end_char", 0)
        if a_start >= s_start and a_end <= s_end and not (a_start == t_start and a_end == t_end):
            anc_text = anc.get("span_text", anc.get("text", abstract[a_start:a_end]))
            anc_label = anc.get("label", anc.get("spanner_label", ""))
            replacements.append({
                "start": a_start - s_start,
                "end": a_end - s_start,
                "replacement": f"[ANCHOR: {anc_text} ({anc_label})]"
            })

    # Sort replacements descending by start position to avoid index shifting
    replacements.sort(key=lambda x: x["start"], reverse=True)
    res_str = sentence_raw
    for rep in replacements:
        s, e = rep["start"], rep["end"]
        res_str = res_str[:s] + rep["replacement"] + res_str[e:]

    return res_str.strip()


def build_anchored_abstract(abstract: str, uncertain_spans: list, confident_spans: list) -> str:
    """
    Builds full abstract where confident BERT spans are pre-filled as [ANCHOR: text (label)]
    and flagged uncertain spans are marked as [TARGET: text].
    """
    replacements = []
    for u in uncertain_spans:
        replacements.append({
            "start": u["start_char"],
            "end": u["end_char"],
            "replacement": f"[TARGET: {u['span_text']}]"
        })
    for c in confident_spans:
        label = c.get("label", c.get("spanner_label", ""))
        replacements.append({
            "start": c["start_char"],
            "end": c["end_char"],
            "replacement": f"[ANCHOR: {c['span_text']} ({label})]"
        })
    replacements.sort(key=lambda x: x["start"], reverse=True)
    res = abstract
    for r in replacements:
        res = res[:r["start"]] + r["replacement"] + res[r["end"]:]
    return res


def classify_challenge_type(
    top1_label: str,
    top1_prob: float,
    top2_label: str,
    top2_prob: float,
    margin: float,
    p_o: float,
    novelty: float,
    u_score: float
) -> str:
    """
    Categorizes the uncertainty challenge into one of 4 canonical escalation types.
    """
    if (top1_prob + top2_prob >= 0.70) and margin < 0.18:
        return "TOP2_COMPETITION"
    elif p_o >= 0.25:
        return "BACKGROUND_BORDERLINE"
    elif novelty >= 0.60 and u_score < 0.35:
        return "NOVEL_UNSEEN_TERM"
    elif novelty >= 0.50:
        return "BOUNDARY_AMBIGUITY"
    else:
        return "CATEGORY_AMBIGUITY"


class SpanNER_DinGenPipeline:
    def __init__(
        self,
        spanner_model: nn.Module,
        tokenizer: Any,
        novelty_scorer: Any,
        dinasor_agent: Dinasor,
        dingenerator_agent: DinGenerator,
        device: torch.device,
        id2label: dict,
        use_new_labels: bool = True,
        w_novelty: float = 0.20,
        w_uncertainty: float = 0.80,
        unreliability_threshold: float = 0.40,
        novelty_threshold: float = 0.60,
        margin_threshold: float = 0.50,
        gating_mode: str = "weighted",
        output_folder: str = "output",
        apply_nms: Optional[bool] = None,
        cache_file: Optional[str] = None,
        enable_cache: bool = True
    ):
        self.spanner_model = spanner_model
        self.tokenizer = tokenizer
        self.novelty_scorer = novelty_scorer
        self.dinasor_agent = dinasor_agent
        self.dingenerator_agent = dingenerator_agent
        self.device = device
        self.id2label = id2label
        self.num_classes = len(id2label)
        self.use_new_labels = use_new_labels
        self.w_novelty = w_novelty
        self.w_uncertainty = w_uncertainty
        self.unreliability_threshold = unreliability_threshold
        self.novelty_threshold = novelty_threshold
        self.margin_threshold = margin_threshold
        self.gating_mode = gating_mode.lower()
        self.output_folder = output_folder
        if apply_nms is None:
            apply_nms = os.getenv("APPLY_NMS", "False").strip().lower() in ("true", "1", "yes")
        self.apply_nms = apply_nms
        os.makedirs(self.output_folder, exist_ok=True)

        self.category_thresholds = {
            'Nutrition': 0.25,
            'Physical_activity': 0.25,
            'Physical_activities': 0.25,
            'Sleep': 0.35,
            'Socioeconomic_factors': 0.35,
            'Environmental_exposures': 0.35,
            'Drugs': 0.25,
            'Substance_use': 0.25,
            'Non_physical_leisure': 0.35,
            'Non_physical_leisure_time_activities': 0.35,
            'Mental_health_practices': 0.40,
            'Beauty_and_Cleaning': 0.40,
            'Personal_care_products_and_cosmetic_procedures': 0.40
        }

        self.enable_cache = enable_cache
        self.cache_file = cache_file or os.path.join(self.output_folder, ".llm_cache.json")
        self.llm_cache: Dict[str, Any] = {}
        if self.enable_cache:
            self._load_cache()

    def _get_cache_key(self, doc_id: str, span_text: str, start_char: int, end_char: int) -> str:
        clean_text = normalize_text(span_text)
        return f"{doc_id}::{start_char}_{end_char}::{clean_text}"

    def _load_cache(self):
        if not self.enable_cache or not self.cache_file:
            return
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.llm_cache = json.load(f)
                print(f"📦 [LLM Cache] Loaded {len(self.llm_cache)} cached LLM decisions from {self.cache_file}")
            except Exception as e:
                print(f"⚠️ [LLM Cache] Warning loading cache: {e}")
                self.llm_cache = {}
        else:
            self.llm_cache = {}

    def clear_cache(self):
        """Completely clears in-memory and on-disk LLM decision cache."""
        self.llm_cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print(f"🗑️ [LLM Cache] Removed persistent cache file: {self.cache_file}")
            except Exception as e:
                print(f"⚠️ [LLM Cache] Error removing cache file: {e}")

    def _save_cache(self):
        if not self.enable_cache or not self.cache_file:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            tmp_path = f"{self.cache_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.llm_cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.cache_file)
        except Exception as e:
            print(f"⚠️ [LLM Cache] Warning saving cache: {e}")

    def _compute_uncertainties(self, logits, input_ids=None, attention_mask=None, method="mcd", mcd_passes=5):
        """
        Computes uncertainty metrics and returns:
        (u_scores, top1_probs, top1_label_ids, margins, top2_probs, top2_label_ids, p_background_o)
        """
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, :, 0]
        top2_probs = sorted_probs[:, :, 1]
        top1_label_ids = sorted_indices[:, :, 0]
        top2_label_ids = sorted_indices[:, :, 1]
        p_background_o = probs[:, :, 0]
        margins = top1_probs - top2_probs

        act = method.lower()
        if act == "lc":
            u = 1.0 - top1_probs
            return u, top1_probs, top1_label_ids, margins, top2_probs, top2_label_ids, p_background_o
        elif act == "margin":
            u = 1.0 - margins
            return u, top1_probs, top1_label_ids, margins, top2_probs, top2_label_ids, p_background_o
        elif act == "mcd" and input_ids is not None:
            self.spanner_model.train()
            mcd_probs_list = []
            with torch.no_grad():
                for _ in range(mcd_passes):
                    out = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    mcd_probs_list.append(F.softmax(out["logits"], dim=-1))
            self.spanner_model.eval()
            mcd_stack = torch.stack(mcd_probs_list, dim=0)
            mcd_entropies = -torch.sum(mcd_stack * torch.log(mcd_stack + 1e-9), dim=-1)
            u_mcd_norm = torch.mean(mcd_entropies, dim=0) / math.log(self.num_classes)
            mean_mcd_probs = torch.mean(mcd_stack, dim=0)
            sorted_mean_probs, sorted_mean_indices = torch.sort(mean_mcd_probs, dim=-1, descending=True)
            mcd_top1_probs = sorted_mean_probs[:, :, 0]
            mcd_top2_probs = sorted_mean_probs[:, :, 1]
            mcd_top1_labels = sorted_mean_indices[:, :, 0]
            mcd_top2_labels = sorted_mean_indices[:, :, 1]
            mcd_p_o = mean_mcd_probs[:, :, 0]
            mcd_margins = mcd_top1_probs - mcd_top2_probs
            return u_mcd_norm, mcd_top1_probs, mcd_top1_labels, mcd_margins, mcd_top2_probs, mcd_top2_labels, mcd_p_o
        else:  # Default: Prediction Entropy (PE)
            raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            u_pe_norm = raw_entropy / math.log(self.num_classes)
            return u_pe_norm, top1_probs, top1_label_ids, margins, top2_probs, top2_label_ids, p_background_o

    def _resolve_with_dinasor_dingenerator(
        self,
        abstract: str,
        uncertain_candidates: list,
        confident_anchors: list = None,
        doc_id: str = None,
        use_contrastive_exemplars: bool = True,
        use_anchored_context: bool = True,
        use_restricted_top2: bool = True,
        use_margin_safeguard: bool = True
    ) -> list:
        """
        Stage 4a (Dinasor Reflection) + Stage 4b (DinGenerator Relabeling) + Stage 4c (Margin Safeguard)
        with Contrastive Exemplars and Anchored Contexts.
        """
        if not uncertain_candidates:
            return []

        # Check LLM decision cache
        cached_resolved = []
        uncached_candidates = []
        if self.enable_cache and doc_id:
            for c in uncertain_candidates:
                key = self._get_cache_key(doc_id, c["span_text"], c["start_char"], c["end_char"])
                if key in self.llm_cache:
                    cached_ent = dict(self.llm_cache[key])
                    for field in ["prob", "top1_prob", "top2_prob", "margin", "p_o", "p_background_o", "u_score", "uncertainty", "novelty", "unreliability"]:
                        if field in c:
                            cached_ent[field] = c[field]
                    cached_resolved.append(cached_ent)
                else:
                    uncached_candidates.append(c)

            if not uncached_candidates:
                # 100% cache hit! No LLM API calls needed.
                return sorted(cached_resolved, key=lambda x: x["start_char"])
        else:
            uncached_candidates = uncertain_candidates

        save_prefix = f"{doc_id}_" if doc_id else ""
        save_dinasor_path = f"{self.output_folder}/{save_prefix}dinasor_raw.txt" if getattr(self, "save_raw_files", False) else None
        save_dingen_path = f"{self.output_folder}/{save_prefix}dingenerator_raw.txt" if getattr(self, "save_raw_files", False) else None

        if confident_anchors is None:
            confident_anchors = []

        # Enrich each uncached uncertain candidate
        enriched_dinasor_payload = []
        for c in uncached_candidates:
            # 1. Anchored sentence
            anchored_sent = build_anchored_sentence(abstract, c, confident_anchors) if use_anchored_context else ""

            # 2. Challenge type
            ch_type = classify_challenge_type(
                top1_label=c["top1_label"],
                top1_prob=c["prob"],
                top2_label=c["top2_label"],
                top2_prob=c["top2_prob"],
                margin=c["margin"],
                p_o=c["p_o"],
                novelty=c.get("novelty", 0.0),
                u_score=c["u_score"]
            )

            # 3. Allowed categories for restricted decision space
            if ch_type == "TOP2_COMPETITION" and use_restricted_top2:
                allowed_cats = [c["top1_label"], c["top2_label"]]
            else:
                allowed_cats = []

            # 4. Contrastive Exemplars
            contrastive_exs = []
            if use_contrastive_exemplars and self.novelty_scorer is not None and hasattr(self.novelty_scorer, "get_contrastive_exemplars"):
                cand_vec = c.get("embedding", None)
                if cand_vec is None:
                    cand_vec = self.novelty_scorer.embed([c["span_text"]])[0]
                target_cls = [c["top1_label"]]
                if c["top2_label"] and c["top2_label"].lower() != "o" and c["top2_label"] != c["top1_label"]:
                    target_cls.append(c["top2_label"])
                contrastive_exs = self.novelty_scorer.get_contrastive_exemplars(
                    cand_vec, target_classes=target_cls, top_k_per_class=1
                )

            item_payload = {
                "entity": c["span_text"],
                "challenge_type": ch_type,
                "allowed_categories": allowed_cats,
                "spanner_evidence": {
                    "predicted_tag": c["spanner_label"],
                    "top1": f"{c['top1_label']} ({c['prob']:.4f})",
                    "top2": f"{c['top2_label']} ({c['top2_prob']:.4f})",
                    "margin": f"{c['margin']:.4f}",
                    "p_background_o": f"{c['p_o']:.4f}",
                    "uncertainty_score": f"{c['u_score']:.4f}",
                    "novelty_score": f"{c.get('novelty', 0.0):.4f}"
                },
                "contrastive_exemplars": contrastive_exs,
                "anchored_context": {
                    "target_sentence": anchored_sent
                }
            }
            enriched_dinasor_payload.append(item_payload)
            c["challenge_type"] = ch_type
            c["allowed_categories"] = allowed_cats
            c["contrastive_exemplars"] = contrastive_exs
            c["anchored_sentence"] = anchored_sent

        # Construct full document-level anchored abstract (confident spans pre-filled as [ANCHOR:...], uncertain as [TARGET:...])
        anchored_abstract = build_anchored_abstract(abstract, uncertain_candidates, confident_anchors) if use_anchored_context else abstract

        # Stage 4a: Run Dinasor Reflector
        hints_out = self.dinasor_agent.run(
            abstract=anchored_abstract,
            uncertain_entities=enriched_dinasor_payload,
            save_output_path=save_dinasor_path,
            return_prompt=True,
            return_raw=True
        )
        if isinstance(hints_out, tuple):
            if len(hints_out) == 3:
                hints, dinasor_prompt, dinasor_raw = hints_out
            elif len(hints_out) == 2:
                hints, dinasor_prompt = hints_out
                dinasor_raw = ""
            else:
                hints = hints_out[0]
                dinasor_prompt = ""
                dinasor_raw = ""
        else:
            hints, dinasor_prompt, dinasor_raw = hints_out, "", ""

        # Stage 4b: Run DinGenerator Relabeler
        gen_result = self.dingenerator_agent.run(
            abstract=anchored_abstract,
            hints=hints,
            uncertain_entities=enriched_dinasor_payload,
            save_output_path=save_dingen_path
        )

        reasoning = gen_result.get("reasoning", "")
        hints_applied = gen_result.get("hints_applied", [])
        labeled_list = gen_result.get("labeled_entities", [])
        dingen_prompt = gen_result.get("prompt", "")
        dingen_raw = gen_result.get("raw_response", "")

        # Map labeled entities by exact span text
        labeled_map = {}
        for item in labeled_list:
            ent_text = normalize_text(item.get("entity", ""))
            labeled_map[ent_text] = item

        resolved_entities = []
        for cand in uncached_candidates:
            span_text = cand["span_text"]
            norm_span = normalize_text(span_text)
            res_item = labeled_map.get(norm_span, {})
            raw_llm_label = res_item.get("label", "O")
            rationale = res_item.get("rationale", "Resolved by DinGenerator")

            # Canonicalize LLM label
            if raw_llm_label.lower() in ["non_lsf", "non-lsf", "none", "o", "non_lifestyle"]:
                llm_label = "O"
            else:
                llm_label = canonicalize_label(raw_llm_label, use_new_labels=self.use_new_labels)

            cand_spanner_label = canonicalize_label(cand["spanner_label"], use_new_labels=self.use_new_labels)
            cand_margin = float(cand.get("margin", cand.get("prob", 0.0)))
            cand_p_o = float(cand.get("p_o", 0.0))

            # Stage 4c: Top-1 vs Top-2 Margin Safeguard Veto
            # Protect high confidence entities with low background O probability from false LLM drops
            if use_margin_safeguard and llm_label == "O" and cand_margin >= self.margin_threshold and cand_spanner_label.lower() != "o" and cand_p_o < 0.20:
                final_label = cand_spanner_label
                decision_source = f"SpanNER Margin Veto (Decisive Margin: {cand_margin:.4f} >= {self.margin_threshold:.2f} & P(O): {cand_p_o:.2f} protected from LLM drop)"
            else:
                final_label = llm_label
                decision_source = f"DinGenPipe Reflective Resolution ({cand.get('challenge_type', 'Escalation')})"

            ent_dict = {
                "span_text": span_text,
                "start_char": cand["start_char"],
                "end_char": cand["end_char"],
                "label": final_label,
                "spanner_label": cand_spanner_label,
                "top1_label": cand.get("top1_label", cand_spanner_label),
                "top2_label": cand.get("top2_label", "N/A"),
                "prob": cand.get("prob", 0.0),
                "top1_prob": cand.get("top1_prob", cand.get("prob", 0.0)),
                "top2_prob": cand.get("top2_prob", 0.0),
                "margin": cand_margin,
                "p_o": cand_p_o,
                "p_background_o": cand_p_o,
                "u_score": cand["u_score"],
                "uncertainty": cand["u_score"],
                "novelty": cand.get("novelty", 0.0),
                "unreliability": cand["unreliability"],
                "llm_label": llm_label,
                "challenge_type": cand.get("challenge_type", "CATEGORY_AMBIGUITY"),
                "contrastive_exemplars": cand.get("contrastive_exemplars", []),
                "anchored_sentence": cand.get("anchored_sentence", ""),
                "escalated_to_llm": True,
                "source": decision_source,
                "dinasor_hints": hints,
                "dinasor_prompt": dinasor_prompt,
                "dinasor_raw_response": dinasor_raw,
                "dingenerator_prompt": dingen_prompt,
                "dingenerator_raw_response": dingen_raw,
                "reasoning": reasoning,
                "hints_applied": hints_applied,
                "rationale": rationale
            }
            resolved_entities.append(ent_dict)

            # Persist to LLM cache
            if self.enable_cache and doc_id:
                key = self._get_cache_key(doc_id, span_text, cand["start_char"], cand["end_char"])
                self.llm_cache[key] = ent_dict

        if self.enable_cache and doc_id and resolved_entities:
            self._save_cache()

        all_resolved = cached_resolved + resolved_entities
        return sorted(all_resolved, key=lambda x: x["start_char"])

    def predict_abstract(self, text: str, uncertainty_method: str = "mcd", doc_id: str = None, precomputed_candidates: Any = None) -> list:
        """Runs SpanNER recognition, LOF novelty, Gating, and DinGenPipe resolution on a single abstract."""
        cands_input = None
        if precomputed_candidates is not None:
            if isinstance(precomputed_candidates, dict):
                cands_input = precomputed_candidates
            else:
                cands_input = [precomputed_candidates]

        batch_res = self.predict_batch(
            [text],
            batch_size=1,
            uncertainty_method=uncertainty_method,
            doc_ids=[doc_id] if doc_id else None,
            verbose=False,
            precomputed_candidates=cands_input
        )
        return batch_res[0] if batch_res else []

    def predict_batch(
        self,
        texts: list,
        batch_size: int = 5,
        uncertainty_method: str = "mcd",
        doc_ids: list = None,
        verbose: bool = True,
        precomputed_candidates: Any = None
    ) -> list:
        """Runs full batched inference across multiple abstracts."""
        all_results = []
        total_abstracts = len(texts)
        total_batches = (total_abstracts + batch_size - 1) // batch_size

        if verbose:
            print("=" * 96)
            print(f"  🚀 SpanNER + DinGenPipe Batch Engine: Processing {total_abstracts} abstracts in {total_batches} batches (B={batch_size})")
            print(f"  ⚡ Uncertainty: {uncertainty_method.upper()} | Gating: {self.gating_mode.upper()} | Unreliability Thresh: {self.unreliability_threshold}")
            print("=" * 96)

        for chunk_start in range(0, total_abstracts, batch_size):
            chunk_texts = texts[chunk_start : chunk_start + batch_size]
            chunk_doc_ids = doc_ids[chunk_start : chunk_start + batch_size] if doc_ids else [f"doc_{i+1}" for i in range(chunk_start, chunk_start + len(chunk_texts))]
            current_bs = len(chunk_texts)
            batch_num = (chunk_start // batch_size) + 1

            if verbose:
                print(f"\n📦 [Batch {batch_num}/{total_batches}] Processing abstracts {chunk_start + 1}-{chunk_start + current_bs} of {total_abstracts}")

            chunk_pending = []

            if precomputed_candidates is not None:
                if verbose and chunk_start == 0:
                    print("  ⚡ [Stage 1: Precomputed Cache] Bypassing neural forward pass; consuming cached candidate spans directly.")
                for b_idx in range(current_bs):
                    global_idx = chunk_start + b_idx
                    doc_key = chunk_doc_ids[b_idx] if (doc_ids and b_idx < len(chunk_doc_ids)) else None
                    cands = None
                    if isinstance(precomputed_candidates, dict):
                        if doc_key and doc_key in precomputed_candidates:
                            cands = precomputed_candidates[doc_key]
                        elif global_idx in precomputed_candidates:
                            cands = precomputed_candidates[global_idx]
                        elif str(global_idx) in precomputed_candidates:
                            cands = precomputed_candidates[str(global_idx)]
                    elif isinstance(precomputed_candidates, list) and global_idx < len(precomputed_candidates):
                        cands = precomputed_candidates[global_idx]

                    if isinstance(cands, dict) and "candidates" in cands:
                        cands = cands["candidates"]
                    if cands is None:
                        cands = []

                    for s_idx, c in enumerate(cands):
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        if not s_text:
                            continue
                        c_start = int(c.get("start_char", c.get("char_start", 0)))
                        c_end = int(c.get("end_char", c.get("char_end", 0)))
                        raw_lbl = c.get("predicted_label", c.get("spanner_label", "O"))
                        spanner_lbl = to_new_label(raw_lbl) if self.use_new_labels else raw_lbl
                        raw_top2 = c.get("second_best_label", c.get("top2_label", "O"))
                        top2_lbl = to_new_label(raw_top2) if self.use_new_labels else raw_top2

                        prob_val = float(c.get("prob", c.get("confidence", 1.0)))
                        u_val = float(c.get("uncertainty", c.get("u_score", 0.0)))
                        margin_val = float(c.get("margin", prob_val))
                        top2_p = float(c.get("top2_prob", 0.0))
                        po_val = float(c.get("p_background_o", c.get("p_o", 0.0)))
                        nov_val = float(c.get("novelty_score", c.get("novelty", 0.0)))

                        lbl_id = 0 if spanner_lbl == "O" else 1
                        top2_lbl_id = 0 if top2_lbl == "O" else 1
                        for k, v in self.id2label.items():
                            if v == raw_lbl or to_new_label(v) == spanner_lbl:
                                lbl_id = k
                            if v == raw_top2 or to_new_label(v) == top2_lbl:
                                top2_lbl_id = k

                        chunk_pending.append({
                            "b_idx": b_idx,
                            "s_idx": s_idx,
                            "char_start": c_start,
                            "char_end": c_end,
                            "span_text": s_text,
                            "label_id": lbl_id,
                            "top2_label_id": top2_lbl_id,
                            "prob": prob_val,
                            "top2_prob": top2_p,
                            "u_score": u_val,
                            "margin": margin_val,
                            "p_o": po_val,
                            "novelty": nov_val,
                            "novelty_score": nov_val
                        })
            else:
                # Stage 1: SpanNER Tokenization & Forward Pass
                encoding = self.tokenizer(chunk_texts, padding=True, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                offsets = encoding["offset_mapping"].cpu().numpy()

                with torch.no_grad():
                    outputs = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    candidate_spans = outputs["candidate_spans"]

                u_scores, top1_probs, top1_label_ids, margins, top2_probs, top2_label_ids, p_background_o = self._compute_uncertainties(
                    logits,
                    input_ids=input_ids if uncertainty_method == "mcd" else None,
                    attention_mask=attention_mask if uncertainty_method == "mcd" else None,
                    method=uncertainty_method
                )

                # GPU Boolean Masking
                keep_mask = (top1_label_ids != 0) | (u_scores >= self.unreliability_threshold)
                b_indices, s_indices = torch.where(keep_mask)

                if len(b_indices) > 0:
                    b_idx_arr = b_indices.cpu().numpy()
                    s_idx_arr = s_indices.cpu().numpy()
                    label_id_arr = top1_label_ids[b_indices, s_indices].cpu().numpy()
                    top2_label_arr = top2_label_ids[b_indices, s_indices].cpu().numpy()
                    u_arr = u_scores[b_indices, s_indices].cpu().numpy()
                    prob_arr = top1_probs[b_indices, s_indices].cpu().numpy()
                    top2_prob_arr = top2_probs[b_indices, s_indices].cpu().numpy()
                    margin_arr = margins[b_indices, s_indices].cpu().numpy()
                    po_arr = p_background_o[b_indices, s_indices].cpu().numpy()

                    for i in range(len(b_idx_arr)):
                        b_idx = int(b_idx_arr[i])
                        s_idx = int(s_idx_arr[i])
                        start, end, _ = candidate_spans[s_idx]
                        char_start = int(offsets[b_idx, start, 0])
                        char_end = int(offsets[b_idx, end, 1])
                        span_text = chunk_texts[b_idx][char_start:char_end].strip()
                        if span_text:
                            chunk_pending.append({
                                "b_idx": b_idx,
                                "s_idx": s_idx,
                                "char_start": char_start,
                                "char_end": char_end,
                                "span_text": span_text,
                                "label_id": int(label_id_arr[i]),
                                "top2_label_id": int(top2_label_arr[i]),
                                "prob": float(prob_arr[i]),
                                "top2_prob": float(top2_prob_arr[i]),
                                "u_score": float(u_arr[i]),
                                "margin": float(margin_arr[i]),
                                "p_o": float(po_arr[i])
                            })

            # Stage 2: Parallel LOF Novelty Scoring
            if self.novelty_scorer is not None and chunk_pending:
                all_have_nov = all("novelty_score" in p and p.get("novelty_score") is not None for p in chunk_pending)
                if all_have_nov and precomputed_candidates is not None:
                    nov_scores = np.array([float(p["novelty_score"]) for p in chunk_pending], dtype=float)
                    if verbose:
                        print(f"  🔍 [Stage 2: LOF Novelty] Reusing {len(chunk_pending)} precomputed novelty scores.")
                else:
                    span_texts = [p["span_text"] for p in chunk_pending]
                    span_vecs = self.novelty_scorer.embed(span_texts)
                    nov_scores = self.novelty_scorer.novelty(span_vecs)
                    for p_idx, item in enumerate(chunk_pending):
                        item["embedding"] = span_vecs[p_idx]
            else:
                nov_scores = np.zeros(len(chunk_pending))

            # Stage 3: Gating Partitioning
            abstract_extracted = [[] for _ in range(current_bs)]
            abstract_uncertain = [[] for _ in range(current_bs)]

            for p_idx, item in enumerate(chunk_pending):
                b_idx = item["b_idx"]
                label_id = item["label_id"]
                prob = item["prob"]
                u_score = item["u_score"]
                novelty = float(nov_scores[p_idx])
                raw_label = self.id2label[label_id]
                spanner_label = to_new_label(raw_label) if self.use_new_labels else raw_label
                top2_raw = self.id2label[item["top2_label_id"]]
                top2_label = to_new_label(top2_raw) if self.use_new_labels else top2_raw

                unc_thresh = self.category_thresholds.get(spanner_label, self.unreliability_threshold)

                if self.gating_mode == "or":
                    escalate = (u_score >= unc_thresh) or (novelty >= self.novelty_threshold)
                    unreliability = max(u_score, novelty)
                elif self.gating_mode == "prob_or":
                    unreliability = 1.0 - (1.0 - novelty) * (1.0 - u_score)
                    escalate = unreliability >= self.unreliability_threshold
                elif self.gating_mode == "copula":
                    unreliability = float(np.clip(u_score + novelty - 1.5 * (u_score * novelty), 0.0, 1.0))
                    escalate = unreliability >= self.unreliability_threshold
                elif self.gating_mode == "and":
                    escalate = (u_score >= unc_thresh) and (novelty >= self.novelty_threshold)
                    unreliability = min(u_score, novelty)
                else:  # Default: Weighted convex combination
                    unreliability = self.w_novelty * novelty + self.w_uncertainty * u_score
                    escalate = unreliability >= self.unreliability_threshold

                if not escalate:
                    if label_id != 0:
                        abstract_extracted[b_idx].append({
                            "span_text": item["span_text"],
                            "start_char": item["char_start"],
                            "end_char": item["char_end"],
                            "label": spanner_label,
                            "spanner_label": spanner_label,
                            "top1_label": spanner_label,
                            "top2_label": top2_label,
                            "prob": prob,
                            "top1_prob": prob,
                            "top2_prob": item["top2_prob"],
                            "margin": item["margin"],
                            "p_o": item["p_o"],
                            "p_background_o": item["p_o"],
                            "u_score": u_score,
                            "uncertainty": u_score,
                            "novelty": novelty,
                            "unreliability": unreliability,
                            "llm_label": "❌ (Not Escalated)",
                            "escalated_to_llm": False,
                            "source": f"Primary SpanNER (High Confidence {uncertainty_method.upper()})"
                        })
                else:
                    abstract_uncertain[b_idx].append({
                        "span_text": item["span_text"],
                        "start_char": item["char_start"],
                        "end_char": item["char_end"],
                        "spanner_label": spanner_label,
                        "top1_label": spanner_label,
                        "top2_label": top2_label,
                        "prob": prob,
                        "top1_prob": prob,
                        "top2_prob": item["top2_prob"],
                        "u_score": u_score,
                        "uncertainty": u_score,
                        "margin": item["margin"],
                        "p_o": item["p_o"],
                        "p_background_o": item["p_o"],
                        "novelty": novelty,
                        "unreliability": unreliability,
                        "embedding": item.get("embedding", None)
                    })

            # Stage 4: Multi-Threaded Concurrent DinGenPipe Escalations (1 Thread per Abstract in Batch)
            escalation_tasks = [
                (b_idx, chunk_texts[b_idx], abstract_uncertain[b_idx], abstract_extracted[b_idx], chunk_doc_ids[b_idx])
                for b_idx in range(current_bs)
                if abstract_uncertain[b_idx]
            ]

            if escalation_tasks:
                max_workers = len(escalation_tasks)
                total_cands = sum(len(t[2]) for t in escalation_tasks)
                if verbose:
                    print(f"  🤖 [Stage 4: LLM Escalation] Escalating {total_cands} unreliable spans across {len(escalation_tasks)} abstracts concurrently ({max_workers} worker threads)...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_bidx = {
                        executor.submit(self._resolve_with_dinasor_dingenerator, text, cands, anchors, doc_id): (b_idx, doc_id, cands)
                        for b_idx, text, cands, anchors, doc_id in escalation_tasks
                    }
                    for future in future_to_bidx:
                        b_idx, doc_id, cands = future_to_bidx[future]
                        try:
                            res_ents = future.result()
                            abstract_extracted[b_idx].extend(res_ents)
                            if verbose:
                                num_esc = len(cands)
                                confirmed = sum(1 for e in res_ents if e.get("label") == e.get("spanner_label") and e.get("label", "O") != "O")
                                relabeled = sum(1 for e in res_ents if e.get("label") != e.get("spanner_label") and e.get("label", "O") != "O")
                                filtered_o = sum(1 for e in res_ents if e.get("label", "O") == "O")
                                newly_discovered = max(0, len(res_ents) - num_esc)
                                active_kept = len([e for e in res_ents if e.get("label", "O") != "O"])

                                detail_parts = []
                                if confirmed:
                                    detail_parts.append(f"{confirmed} confirmed")
                                if relabeled:
                                    detail_parts.append(f"{relabeled} relabeled")
                                if filtered_o:
                                    detail_parts.append(f"{filtered_o} filtered to Non-LSF")
                                if newly_discovered > 0:
                                    detail_parts.append(f"+{newly_discovered} new discovered")

                                summary_str = ", ".join(detail_parts) if detail_parts else f"{len(res_ents)} resolved"
                                print(f"     -> Abstract {doc_id}: {num_esc} escalated -> {summary_str} ({active_kept} active kept)")
                        except Exception as exc:
                            if verbose:
                                print(f"  ⚠️ Error resolving abstract {doc_id} with DinGenPipe: {exc}")
            else:
                if verbose:
                    print(f"  ✨ [Stage 4: LLM Escalation] 0 abstracts required escalation (100% accepted locally).")

            for b_idx in range(current_bs):
                ents = abstract_extracted[b_idx]
                if self.apply_nms:
                    ents = run_nms_entities(ents)
                else:
                    ents.sort(key=lambda x: x["start_char"])
                all_results.append(ents)

        return all_results
