from __future__ import annotations
import os
import sys
import time
import json
import math
import copy
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.architectures.dinasor_dingen.pipeline import (
    build_anchored_sentence,
    build_anchored_abstract,
    classify_challenge_type,
    SpanNER_DinGenPipeline
)
from src.common.label_mapping import canonicalize_label, to_new_label
from src.common.audit_logger import (
    create_run_directory,
    log_abstract_audit,
    log_run_summary,
    classify_audit_outcome,
    match_gt_label,
    normalize_text,
    to_float
)
from src.common.evaluation_framework import (
    evaluate_doc_4_formulations,
    aggregate_4_formulations,
    format_4_formulations_summary_box
)

# ==============================================================================
# 1. Persistent LLM Decision Cache
# ==============================================================================
class PersistentLLMDecisionCache:
    """
    Caches LLM reflection decisions keyed by:
    hash(doc_id, span_text, start_char, end_char, challenge_type, exemplars)
    Guarantees zero redundant API calls and 100% deterministic reproducibility.
    """
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"⚠️ Warning loading LLM cache: {e}")
                self.cache = {}

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def get_key(self, doc_id: str, span_text: str, start_char: int, end_char: int, challenge_type: str, ex: bool = True, anc: bool = True, top2: bool = True) -> str:
        if ex and anc and top2:
            return f"{doc_id}::{start_char}_{end_char}::{normalize_text(span_text)}::{challenge_type}"
        return f"{doc_id}::{start_char}_{end_char}::{normalize_text(span_text)}::{challenge_type}::ex{int(ex)}_anc{int(anc)}_top{int(top2)}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(key)

    def set(self, key: str, value: Dict[str, Any]):
        self.cache[key] = value


# ==============================================================================
# 2. Precomputed Multi-Metric Feature Store
# ==============================================================================
class PrecomputedFeatureCache:
    """
    Extracts all candidate spans and precomputes all uncertainty & novelty metrics
    in a single pass over the dataset.
    """
    def __init__(
        self,
        pipeline: SpanNER_DinGenPipeline,
        texts: List[str],
        doc_ids: List[str],
        doc_gts: List[List[Dict[str, Any]]],
        mcd_passes: int = 5
    ):
        self.pipeline = pipeline
        self.texts = texts
        self.doc_ids = doc_ids
        self.doc_gts = doc_gts
        self.mcd_passes = mcd_passes
        self.doc_candidates: List[List[Dict[str, Any]]] = []
        self._precompute_all()

    def _precompute_all(self):
        print(f"\n⚡ Precomputing multi-metric representations across {len(self.texts)} abstracts...")
        t0 = time.time()
        
        for doc_idx, (text, doc_id, gt_ents) in enumerate(zip(self.texts, self.doc_ids, self.doc_gts)):
            # Tokenize
            enc = self.pipeline.tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            input_ids = enc["input_ids"].to(self.pipeline.device)
            attention_mask = enc["attention_mask"].to(self.pipeline.device)
            offsets = enc["offset_mapping"][0].cpu().numpy()
            seq_len = int(attention_mask.sum().item())

            # Standard forward pass
            with torch.no_grad():
                out = self.pipeline.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"]
                candidate_spans = out["candidate_spans"]
                probs = F.softmax(logits, dim=-1)[0] # [Num_Spans, Num_Classes]

            # 1. Least Confidence & Margin
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0].cpu().numpy()
            top2_probs = sorted_probs[:, 1].cpu().numpy()
            top1_label_ids = sorted_indices[:, 0].cpu().numpy()
            top2_label_ids = sorted_indices[:, 1].cpu().numpy()
            margins = (top1_probs - top2_probs)
            p_o = probs[:, 0].cpu().numpy()
            u_lc = 1.0 - top1_probs
            u_margin = 1.0 - margins

            # 2. Prediction Entropy
            raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).cpu().numpy()
            u_pe = raw_entropy / math.log(self.pipeline.num_classes)

            # 3. Monte Carlo Dropout (MCD)
            self.pipeline.spanner_model.train()
            mcd_list = []
            with torch.no_grad():
                for _ in range(self.mcd_passes):
                    mcd_out = self.pipeline.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    mcd_list.append(F.softmax(mcd_out["logits"], dim=-1)[0])
            self.pipeline.spanner_model.eval()

            mcd_stack = torch.stack(mcd_list, dim=0) # [Passes, Num_Spans, Classes]
            mcd_entropies = -torch.sum(mcd_stack * torch.log(mcd_stack + 1e-9), dim=-1)
            u_mcd = (torch.mean(mcd_entropies, dim=0) / math.log(self.pipeline.num_classes)).cpu().numpy()
            mean_mcd_probs = torch.mean(mcd_stack, dim=0)
            mcd_sorted_probs, _ = torch.sort(mean_mcd_probs, dim=-1, descending=True)
            mcd_top1_p = mcd_sorted_probs[:, 0].cpu().numpy()
            mcd_top2_p = mcd_sorted_probs[:, 1].cpu().numpy()

            # Filter valid candidate mentions (non-O top1 predictions)
            doc_cands = []
            cand_texts = []
            for s_idx, (start, end, _) in enumerate(candidate_spans):
                if end >= seq_len:
                    continue
                lbl_id = int(top1_label_ids[s_idx])
                if lbl_id == 0:  # Skip O predictions
                    continue

                char_start = int(offsets[start, 0])
                char_end = int(offsets[end, 1])
                span_text = text[char_start:char_end].strip()
                if not span_text:
                    continue

                raw_label = self.pipeline.id2label[lbl_id]
                spanner_label = to_new_label(raw_label) if self.pipeline.use_new_labels else raw_label
                top2_raw = self.pipeline.id2label[int(top2_label_ids[s_idx])]
                top2_label = to_new_label(top2_raw) if self.pipeline.use_new_labels else top2_raw

                gt_lbl = match_gt_label(span_text, char_start, char_end, gt_ents)
                gt_canonical = canonicalize_label(gt_lbl, use_new_labels=self.pipeline.use_new_labels)

                cand_dict = {
                    "doc_id": doc_id,
                    "span_text": span_text,
                    "start_char": char_start,
                    "end_char": char_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "spanner_label": spanner_label,
                    "top1_label": spanner_label,
                    "top2_label": top2_label,
                    "top1_prob": float(top1_probs[s_idx]),
                    "top2_prob": float(top2_probs[s_idx]),
                    "prob": float(top1_probs[s_idx]),
                    "margin": float(margins[s_idx]),
                    "p_o": float(p_o[s_idx]),
                    "p_background_o": float(p_o[s_idx]),
                    "u_mcd": float(u_mcd[s_idx]),
                    "u_pe": float(u_pe[s_idx]),
                    "u_margin": float(u_margin[s_idx]),
                    "u_lc": float(u_lc[s_idx]),
                    "ground_truth": gt_canonical
                }
                doc_cands.append(cand_dict)
                cand_texts.append(span_text)

            # 4. Compute Vector Embeddings & Multi-Novelty Metrics
            if cand_texts and self.pipeline.novelty_scorer is not None:
                vecs = self.pipeline.novelty_scorer.embed(cand_texts)
                lof_scores = self.pipeline.novelty_scorer.novelty(vecs)
                
                # Compute Cosine-kNN Distance
                matrix = self.pipeline.novelty_scorer.matrix
                k_val = min(10, len(matrix))
                # Normalized cosine distance to k-nearest neighbors
                v_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
                m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
                sim_matrix = np.dot(v_norm, m_norm.T)
                # Top k similarities
                topk_sims = np.sort(sim_matrix, axis=1)[:, -k_val:]
                knn_novelty = np.clip(1.0 - np.mean(topk_sims, axis=1), 0.0, 1.0)

                # Mahalanobis / Centroid Distance
                mean_vec = np.mean(matrix, axis=0)
                diff = vecs - mean_vec
                mahal_dist = np.linalg.norm(diff, axis=1)
                # Normalize via 95th percentile
                max_d = np.percentile(mahal_dist, 95) if len(mahal_dist) > 0 else 1.0
                mahal_novelty = np.clip(mahal_dist / (max_d + 1e-9), 0.0, 1.0)

                for c_idx, c in enumerate(doc_cands):
                    c["embedding"] = vecs[c_idx]
                    c["nov_lof"] = float(lof_scores[c_idx])
                    c["nov_cosine_knn"] = float(knn_novelty[c_idx])
                    c["nov_mahalanobis"] = float(mahal_novelty[c_idx])
            else:
                for c in doc_cands:
                    c["nov_lof"] = 0.0
                    c["nov_cosine_knn"] = 0.0
                    c["nov_mahalanobis"] = 0.0

            self.doc_candidates.append(doc_cands)

        elapsed = time.time() - t0
        total_cands = sum(len(c) for c in self.doc_candidates)
        print(f"✓ Extracted {total_cands} total candidate mentions across {len(self.texts)} abstracts in {elapsed:.2f}s.")


# ==============================================================================
# 3. High-Speed Ablation Engine & Evaluator
# ==============================================================================
class AblationEngine:
    def __init__(
        self,
        pipeline: SpanNER_DinGenPipeline,
        feature_cache: PrecomputedFeatureCache,
        llm_cache: PersistentLLMDecisionCache,
        output_base_dir: str,
        dataset_name: str = "balanced_val_set",
        batch_size: int = 5
    ):
        self.pipeline = pipeline
        self.cache = feature_cache
        self.llm_cache = llm_cache
        self.output_base_dir = output_base_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.experiments_dir = output_base_dir
        os.makedirs(self.experiments_dir, exist_ok=True)

    def evaluate_configuration(
        self,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        unreliability_threshold: float = 0.38,
        margin_threshold: float = 0.50,
        use_contrastive_exemplars: bool = True,
        use_anchored_context: bool = True,
        use_restricted_top2: bool = True,
        use_margin_safeguard: bool = True,
        batch_size: Optional[int] = None,
        log_run_files: bool = True,
        test_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a configuration against the precomputed feature cache and persistent LLM decision store,
        processing in batches and escalating un-cached mentions concurrently with threads equal to the batch size.
        """
        all_audit_records = []
        all_abstract_metrics = []
        
        spanner_tp = Counter(); spanner_fp = Counter(); spanner_fn = Counter()
        hybrid_tp = Counter(); hybrid_fp = Counter(); hybrid_fn = Counter()
        classes_set = set()

        total_spans_evaluated = 0
        total_escalated = 0
        exact_span_correct = 0

        run_dir, abstracts_dir, run_id = None, None, None
        if log_run_files:
            ds_name = dataset_name or self.dataset_name
            t_name = test_name or "General Evaluation"
            r_name = run_name or config_name or f"{gating_mode}_w{int(w_novelty*100)}_t{int(unreliability_threshold*100)}"
            run_dir, abstracts_dir, run_id = create_run_directory(
                base_output_dir=self.output_base_dir,
                run_name=r_name,
                test_name=t_name,
                dataset_name=ds_name,
                use_timestamp=False
            )

        num_abstracts = len(self.cache.texts)
        bs = batch_size or self.batch_size

        for batch_start in range(0, num_abstracts, bs):
            batch_end = min(batch_start + bs, num_abstracts)
            batch_docs = []
            batch_escalation_tasks = []

            for doc_idx in range(batch_start, batch_end):
                text = self.cache.texts[doc_idx]
                doc_id = self.cache.doc_ids[doc_idx]
                gt_ents = self.cache.doc_gts[doc_idx]
                candidates = self.cache.doc_candidates[doc_idx]

                doc_preds = []
                doc_unc_spans = []
                doc_conf_spans = []

                # 1. Apply Gating Filter
                for cand in candidates:
                    total_spans_evaluated += 1
                    u_val = cand.get(f"u_{uncertainty_method}", cand["u_mcd"])
                    nov_val = cand.get(f"nov_{novelty_metric}", cand["nov_lof"])
                    spanner_lbl = cand["spanner_label"]
                    gt_lbl = cand["ground_truth"]

                    # Gating Condition
                    if gating_mode == "or":
                        unreliability = max(u_val, nov_val)
                        escalate = (u_val >= unreliability_threshold) or (nov_val >= 0.55)
                    elif gating_mode == "and":
                        unreliability = min(u_val, nov_val)
                        escalate = (u_val >= unreliability_threshold) and (nov_val >= 0.55)
                    elif gating_mode == "prob_or":
                        unreliability = 1.0 - (1.0 - nov_val) * (1.0 - u_val)
                        escalate = unreliability >= unreliability_threshold
                    elif gating_mode == "copula":
                        unreliability = float(np.clip(u_val + nov_val - 1.5 * (u_val * nov_val), 0.0, 1.0))
                        escalate = unreliability >= unreliability_threshold
                    else:  # Default: Weighted convex
                        unreliability = (w_novelty * nov_val) + (w_uncertainty * u_val)
                        escalate = unreliability >= unreliability_threshold

                    # Build Challenge Type & Anchored Context
                    ch_type = classify_challenge_type(
                        top1_label=cand["top1_label"],
                        top1_prob=cand["top1_prob"],
                        top2_label=cand["top2_label"],
                        top2_prob=cand["top2_prob"],
                        margin=cand["margin"],
                        p_o=cand["p_o"],
                        novelty=nov_val,
                        u_score=u_val
                    )

                    cand_copy = copy.deepcopy(cand)
                    cand_copy["unreliability"] = unreliability
                    cand_copy["u_score"] = u_val
                    cand_copy["novelty"] = nov_val
                    cand_copy["challenge_type"] = ch_type
                    cand_copy["escalated_to_llm"] = escalate

                    if escalate:
                        total_escalated += 1
                        doc_unc_spans.append(cand_copy)
                    else:
                        cand_copy["label"] = spanner_lbl
                        cand_copy["llm_label"] = "N/A"
                        doc_conf_spans.append(cand_copy)
                        doc_preds.append(cand_copy)

                dinasor_hints = []
                dingen_res = {"labeled_entities": []}
                unc_to_query = []
                anchored_abstract = ""

                if doc_unc_spans:
                    anchored_abstract = build_anchored_abstract(text, doc_unc_spans, doc_conf_spans) if use_anchored_context else text
                    for u_cand in doc_unc_spans:
                        cache_key = self.llm_cache.get_key(
                            doc_id=doc_id,
                            span_text=u_cand["span_text"],
                            start_char=u_cand["char_start"],
                            end_char=u_cand["char_end"],
                            challenge_type=u_cand["challenge_type"],
                            ex=use_contrastive_exemplars,
                            anc=use_anchored_context,
                            top2=use_restricted_top2
                        )
                        cached_val = self.llm_cache.get(cache_key)

                        if cached_val is not None:
                            u_cand["llm_label"] = cached_val["llm_label"]
                            u_cand["rationale"] = cached_val.get("rationale", "")
                            u_cand["dinasor_hints"] = cached_val.get("dinasor_hints", [])
                            u_cand["dinasor_prompt"] = cached_val.get("dinasor_prompt", "")
                            u_cand["dinasor_raw_response"] = cached_val.get("dinasor_raw_response", "")
                            u_cand["dingenerator_prompt"] = cached_val.get("dingenerator_prompt", "")
                            u_cand["dingenerator_raw_response"] = cached_val.get("dingenerator_raw_response", "")
                            dinasor_hints = cached_val.get("dinasor_hints", [])
                            dingen_res["labeled_entities"].append({
                                "entity": u_cand["span_text"],
                                "label": cached_val["llm_label"],
                                "rationale": cached_val.get("rationale", "")
                            })
                        else:
                            unc_to_query.append(u_cand)

                    if unc_to_query:
                        batch_escalation_tasks.append((doc_idx, doc_id, text, anchored_abstract, unc_to_query, doc_conf_spans))

                batch_docs.append({
                    "doc_idx": doc_idx,
                    "text": text,
                    "doc_id": doc_id,
                    "gt_ents": gt_ents,
                    "doc_preds": doc_preds,
                    "doc_unc_spans": doc_unc_spans,
                    "doc_conf_spans": doc_conf_spans,
                    "dinasor_hints": dinasor_hints,
                    "dingen_res": dingen_res,
                    "anchored_abstract": anchored_abstract
                })

            # Execute concurrent multi-threaded LLM escalations for un-cached spans in this batch
            # Threads equal to number of abstracts in the batch that require escalation (up to batch_size)
            if batch_escalation_tasks:
                max_workers = len(batch_escalation_tasks)
                total_mentions = sum(len(t[4]) for t in batch_escalation_tasks)
                batch_num = (batch_start // bs) + 1
                total_batches = (num_abstracts + bs - 1) // bs
                print(f"    🤖 [Batch {batch_num}/{total_batches}] Concurrently escalating {total_mentions} mentions across {max_workers} abstracts ({max_workers} worker threads)...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    def _call_escalate(t_item):
                        _, doc_id_t, text_t, anch_t, unc_t, conf_t = t_item
                        if hasattr(self.pipeline, "_resolve_with_dinasor_dingenerator"):
                            return self.pipeline._resolve_with_dinasor_dingenerator(
                                abstract=anch_t if use_anchored_context else text_t,
                                uncertain_candidates=unc_t,
                                confident_anchors=conf_t if use_anchored_context else [],
                                doc_id=doc_id_t,
                                use_contrastive_exemplars=use_contrastive_exemplars,
                                use_anchored_context=use_anchored_context,
                                use_restricted_top2=use_restricted_top2,
                                use_margin_safeguard=use_margin_safeguard
                            )
                        elif hasattr(self.pipeline, "_resolve_with_gating_llm"):
                            return self.pipeline._resolve_with_gating_llm(
                                text=text_t,
                                uncertain_candidates=unc_t
                            )
                        elif hasattr(self.pipeline, "_escalate_and_arbitrate_abstract"):
                            return self.pipeline._escalate_and_arbitrate_abstract(
                                text=text_t,
                                cand_list=unc_t
                            )
                        elif hasattr(self.pipeline, "llm_provider") and self.pipeline.llm_provider is not None:
                            # Direct LinkNER single-pass arbitration fallback
                            from src.architectures.hybrid_linkner.prompt_templates import build_lsf_linkner_batch_prompt
                            prompt = build_lsf_linkner_batch_prompt(text_t, unc_t, use_new_labels=getattr(self.pipeline, "use_new_labels", True))
                            resp = self.pipeline.llm_provider.generate(prompt)
                            # Return unc_t with default labels
                            return unc_t
                        return unc_t

                    future_to_task = {
                        executor.submit(_call_escalate, task): (task[0], task[1], task[4])
                        for task in batch_escalation_tasks
                    }
                    for future in future_to_task:
                        d_idx, doc_id, unc_to_query = future_to_task[future]
                        try:
                            resolved = future.result()
                            for r_item in resolved:
                                cache_key = self.llm_cache.get_key(
                                    doc_id=doc_id,
                                    span_text=r_item["span_text"],
                                    start_char=r_item["start_char"],
                                    end_char=r_item["end_char"],
                                    challenge_type=r_item.get("challenge_type", "NONE"),
                                    ex=use_contrastive_exemplars,
                                    anc=use_anchored_context,
                                    top2=use_restricted_top2
                                )
                                self.llm_cache.set(cache_key, {
                                    "llm_label": r_item.get("llm_label", r_item["label"]),
                                    "rationale": r_item.get("rationale", r_item.get("llm_rationale", "")),
                                    "dinasor_hints": r_item.get("dinasor_hints", []),
                                    "dinasor_prompt": r_item.get("dinasor_prompt", r_item.get("raw_prompt", "")),
                                    "dinasor_raw_response": r_item.get("dinasor_raw_response", r_item.get("raw_response", "")),
                                    "dingenerator_prompt": r_item.get("dingenerator_prompt", ""),
                                    "dingenerator_raw_response": r_item.get("dingenerator_raw_response", "")
                                })
                        except Exception as exc:
                            print(f"  ⚠️ Error resolving abstract {doc_id} with pipeline {self.pipeline.__class__.__name__}: {exc}")
                self.llm_cache.save()

            # Finalize predictions, safeguards, metrics & audits for each document in the batch
            for b_doc in batch_docs:
                doc_id = b_doc["doc_id"]
                text = b_doc["text"]
                gt_ents = b_doc["gt_ents"]
                doc_preds = b_doc["doc_preds"]
                doc_unc_spans = b_doc["doc_unc_spans"]
                doc_conf_spans = b_doc["doc_conf_spans"]
                dinasor_hints = b_doc["dinasor_hints"]
                dingen_res = b_doc["dingen_res"]
                anchored_abstract = b_doc["anchored_abstract"]

                for u_cand in doc_unc_spans:
                    if "llm_label" not in u_cand or u_cand.get("llm_label") is None:
                        cache_key = self.llm_cache.get_key(
                            doc_id=doc_id,
                            span_text=u_cand["span_text"],
                            start_char=u_cand["char_start"],
                            end_char=u_cand["char_end"],
                            challenge_type=u_cand["challenge_type"],
                            ex=use_contrastive_exemplars,
                            anc=use_anchored_context,
                            top2=use_restricted_top2
                        )
                        cached_val = self.llm_cache.get(cache_key)
                        if cached_val is not None:
                            u_cand["llm_label"] = cached_val["llm_label"]
                            u_cand["rationale"] = cached_val.get("rationale", "")
                            u_cand["dinasor_hints"] = cached_val.get("dinasor_hints", [])
                            u_cand["dinasor_prompt"] = cached_val.get("dinasor_prompt", "")
                            u_cand["dinasor_raw_response"] = cached_val.get("dinasor_raw_response", "")
                            u_cand["dingenerator_prompt"] = cached_val.get("dingenerator_prompt", "")
                            u_cand["dingenerator_raw_response"] = cached_val.get("dingenerator_raw_response", "")
                            dinasor_hints = cached_val.get("dinasor_hints", [])
                            dingen_res["labeled_entities"].append({
                                "entity": u_cand["span_text"],
                                "label": cached_val["llm_label"],
                                "rationale": cached_val.get("rationale", "")
                            })

                    llm_lbl = u_cand.get("llm_label", u_cand["spanner_label"])
                    cand_margin = u_cand["margin"]
                    cand_po = u_cand["p_o"]
                    
                    if use_margin_safeguard and llm_lbl == "O" and cand_margin >= margin_threshold and u_cand["spanner_label"] != "O" and cand_po < 0.20:
                        final_lbl = u_cand["spanner_label"]
                    else:
                        final_lbl = llm_lbl

                    u_cand["label"] = final_lbl
                    doc_preds.append(u_cand)

                # Accumulate Performance Metrics
                doc_spanner_tp = 0; doc_spanner_fp = 0; doc_spanner_fn = 0
                doc_hybrid_tp = 0; doc_hybrid_fp = 0; doc_hybrid_fn = 0
                covered_gt_spanner = set(); covered_gt_hybrid = set()

                for p in doc_preds:
                    s_lbl = p["spanner_label"]
                    h_lbl = p["label"]
                    gt_lbl = p["ground_truth"]
                    is_esc = p.get("escalated_to_llm", False)
                    outcome_info = classify_audit_outcome(
                        spanner_label=s_lbl,
                        llm_label=p.get("llm_label", s_lbl),
                        final_label=h_lbl,
                        gt_label=gt_lbl
                    )
                    outcome = outcome_info[0] if isinstance(outcome_info, tuple) else outcome_info

                    all_audit_records.append({
                        "doc_id": doc_id,
                        "span_text": p["span_text"],
                        "ground_truth": gt_lbl,
                        "spanner_label": s_lbl,
                        "hybrid_label": h_lbl,
                        "outcome": outcome,
                        "escalated": is_esc
                    })

                    # Exact accuracy
                    if h_lbl == gt_lbl:
                        exact_span_correct += 1

                    # SpanNER Metrics
                    if s_lbl != "O":
                        classes_set.add(s_lbl)
                        if s_lbl == gt_lbl:
                            spanner_tp[s_lbl] += 1
                            doc_spanner_tp += 1
                        else:
                            spanner_fp[s_lbl] += 1
                            doc_spanner_fp += 1

                    # Hybrid Metrics
                    if h_lbl != "O":
                        classes_set.add(h_lbl)
                        if h_lbl == gt_lbl:
                            hybrid_tp[h_lbl] += 1
                            doc_hybrid_tp += 1
                        else:
                            hybrid_fp[h_lbl] += 1
                            doc_hybrid_fp += 1

                    use_new = getattr(self.pipeline, "use_new_labels", True)
                    for g_idx, g in enumerate(gt_ents):
                        g_can = canonicalize_label(g.get("label", "O"), use_new_labels=use_new)
                        if g_can != "O":
                            if s_lbl == g_can and (p["char_start"] == g.get("start_char") or normalize_text(p["span_text"]) == normalize_text(g.get("text", ""))):
                                covered_gt_spanner.add(g_idx)
                            if h_lbl == g_can and (p["char_start"] == g.get("start_char") or normalize_text(p["span_text"]) == normalize_text(g.get("text", ""))):
                                covered_gt_hybrid.add(g_idx)

                use_new = getattr(self.pipeline, "use_new_labels", True)
                doc_spanner_fn = len([g for i, g in enumerate(gt_ents) if i not in covered_gt_spanner and canonicalize_label(g.get("label", "O"), use_new_labels=use_new) != "O"])
                doc_hybrid_fn = len([g for i, g in enumerate(gt_ents) if i not in covered_gt_hybrid and canonicalize_label(g.get("label", "O"), use_new_labels=use_new) != "O"])

                for g in gt_ents:
                    g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new)
                    if g_lbl != "O":
                        classes_set.add(g_lbl)

                def calc_s_prf(tp, fp, fn):
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                    return p, r, f

                s_p, s_r, s_f = calc_s_prf(doc_spanner_tp, doc_spanner_fp, doc_spanner_fn)
                h_p, h_r, h_f = calc_s_prf(doc_hybrid_tp, doc_hybrid_fp, doc_hybrid_fn)

                # Compute all 4 evaluation formulations for this document
                doc_preds_spanner = [{"span_text": p["span_text"], "start_char": p["char_start"], "end_char": p["char_end"], "label": p["spanner_label"]} for p in doc_preds]
                doc_preds_hybrid = [{"span_text": p["span_text"], "start_char": p["char_start"], "end_char": p["char_end"], "label": p["label"]} for p in doc_preds]
                spanner_4 = evaluate_doc_4_formulations(doc_preds_spanner, gt_ents, use_new_labels=use_new)
                hybrid_4 = evaluate_doc_4_formulations(doc_preds_hybrid, gt_ents, use_new_labels=use_new)

                all_abstract_metrics.append({
                    "doc_id": doc_id,
                    "spanner": {"tp": doc_spanner_tp, "fp": doc_spanner_fp, "fn": doc_spanner_fn, "p": s_p, "r": s_r, "f1": s_f},
                    "hybrid": {"tp": doc_hybrid_tp, "fp": doc_hybrid_fp, "fn": doc_hybrid_fn, "p": h_p, "r": h_r, "f1": h_f},
                    "spanner_4": spanner_4,
                    "hybrid_4": hybrid_4
                })

                # Individual abstract logs
                if log_run_files and abstracts_dir:
                    log_abstract_audit(
                        doc_id=doc_id,
                        text=text,
                        gt_ents=gt_ents,
                        preds=doc_preds,
                        dinasor_hints=dinasor_hints,
                        dingen_results=dingen_res,
                        anchored_abstract=anchored_abstract if doc_unc_spans else text,
                        config={
                            "GATING_MODE": gating_mode,
                            "W_NOVELTY": w_novelty,
                            "W_UNCERTAINTY": w_uncertainty,
                            "METHOD": uncertainty_method,
                            "UNRELIABILITY_THRESHOLD": unreliability_threshold,
                            "ARBITRATION_MARGIN_THRESHOLD": margin_threshold,
                            "MODEL_NAME": getattr(getattr(self.pipeline, "dingenerator_agent", None), "model_name", getattr(getattr(self.pipeline, "llm_provider", None), "model_name", getattr(self.pipeline, "model_name", "SpanNER-Hybrid"))),
                            "DATA_DIR": self.dataset_name
                        },
                        abstracts_dir=abstracts_dir
                    )

        # Global Metrics
        tot_s_tp = sum(spanner_tp.values()); tot_s_fp = sum(spanner_fp.values()); tot_s_fn = sum(m["spanner"]["fn"] for m in all_abstract_metrics)
        tot_h_tp = sum(hybrid_tp.values()); tot_h_fp = sum(hybrid_fp.values()); tot_h_fn = sum(m["hybrid"]["fn"] for m in all_abstract_metrics)

        def calc_prf(tp, fp, fn):
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            return prec, rec, f1

        s_micro_p, s_micro_r, s_micro_f1 = calc_prf(tot_s_tp, tot_s_fp, tot_s_fn)
        h_micro_p, h_micro_r, h_micro_f1 = calc_prf(tot_h_tp, tot_h_fp, tot_h_fn)

        s_p_list, s_r_list, s_f_list = [], [], []
        h_p_list, h_r_list, h_f_list = [], [], []
        per_class_results = []

        for cls in sorted(list(classes_set)):
            c_s_tp = spanner_tp[cls]; c_s_fp = spanner_fp[cls]
            c_h_tp = hybrid_tp[cls]; c_h_fp = hybrid_fp[cls]
            c_fn = sum(1 for r in all_audit_records if r["ground_truth"] == cls and r["hybrid_label"] != cls)
            c_s_fn = sum(1 for r in all_audit_records if r["ground_truth"] == cls and r["spanner_label"] != cls)

            c_s_p, c_s_r, c_s_f = calc_prf(c_s_tp, c_s_fp, c_s_fn)
            c_h_p, c_h_r, c_h_f = calc_prf(c_h_tp, c_h_fp, c_fn)

            s_p_list.append(c_s_p); s_r_list.append(c_s_r); s_f_list.append(c_s_f)
            h_p_list.append(c_h_p); h_r_list.append(c_h_r); h_f_list.append(c_h_f)

            per_class_results.append({
                "class": cls,
                "spanner_f1": c_s_f,
                "hybrid_f1": c_h_f,
                "delta": c_h_f - c_s_f,
                "tp": c_h_tp, "fp": c_h_fp, "fn": c_fn
            })

        s_macro_p = float(np.mean(s_p_list)) if s_p_list else 0.0
        s_macro_r = float(np.mean(s_r_list)) if s_r_list else 0.0
        s_macro_f1 = float(np.mean(s_f_list)) if s_f_list else 0.0

        h_macro_p = float(np.mean(h_p_list)) if h_p_list else 0.0
        h_macro_r = float(np.mean(h_r_list)) if h_r_list else 0.0
        h_macro_f1 = float(np.mean(h_f_list)) if h_f_list else 0.0

        span_accuracy = (exact_span_correct / total_spans_evaluated) if total_spans_evaluated > 0 else 0.0
        local_saved_pct = ((total_spans_evaluated - total_escalated) / total_spans_evaluated * 100) if total_spans_evaluated > 0 else 0.0
        escalation_pct = (total_escalated / total_spans_evaluated * 100) if total_spans_evaluated > 0 else 0.0

        outcomes_dict = dict(Counter(r["outcome"] for r in all_audit_records))

        if log_run_files and run_dir:
            log_run_summary(
                all_audit_records=all_audit_records,
                all_abstract_metrics=all_abstract_metrics,
                config={
                    "GATING_MODE": gating_mode,
                    "W_NOVELTY": w_novelty,
                    "W_UNCERTAINTY": w_uncertainty,
                    "METHOD": uncertainty_method,
                    "UNRELIABILITY_THRESHOLD": unreliability_threshold,
                    "ARBITRATION_MARGIN_THRESHOLD": margin_threshold,
                    "MODEL_NAME": getattr(getattr(self.pipeline, "dingenerator_agent", None), "model_name", getattr(getattr(self.pipeline, "llm_provider", None), "model_name", getattr(self.pipeline, "model_name", "SpanNER-Hybrid")))
                },
                execution_time=0.05,
                run_dir=run_dir
            )

        agg_spanner_4 = aggregate_4_formulations([m["spanner_4"] for m in all_abstract_metrics if "spanner_4" in m])
        agg_hybrid_4 = aggregate_4_formulations([m["hybrid_4"] for m in all_abstract_metrics if "hybrid_4" in m])

        return {
            "config": {
                "gating_mode": gating_mode,
                "w_novelty": w_novelty,
                "w_uncertainty": w_uncertainty,
                "uncertainty_method": uncertainty_method,
                "novelty_metric": novelty_metric,
                "threshold": unreliability_threshold,
                "margin_threshold": margin_threshold,
                "use_contrastive_exemplars": use_contrastive_exemplars,
                "use_anchored_context": use_anchored_context,
                "use_restricted_top2": use_restricted_top2,
                "use_margin_safeguard": use_margin_safeguard
            },
            "metrics": {
                "micro_precision": h_micro_p,
                "micro_recall": h_micro_r,
                "micro_f1": h_micro_f1,
                "macro_precision": h_macro_p,
                "macro_recall": h_macro_r,
                "macro_f1": h_macro_f1,
                "span_accuracy": span_accuracy,
                "spanner_micro_f1": s_micro_f1,
                "spanner_macro_f1": s_macro_f1,
                "delta_micro_f1": h_micro_f1 - s_micro_f1,
                "delta_macro_f1": h_macro_f1 - s_macro_f1
            },
            "formulations_4": {
                "spanner": agg_spanner_4,
                "hybrid": agg_hybrid_4
            },
            "efficiency": {
                "total_spans": total_spans_evaluated,
                "escalated_spans": total_escalated,
                "escalation_rate_pct": escalation_pct,
                "local_saved_pct": local_saved_pct
            },
            "per_class": per_class_results,
            "outcomes": outcomes_dict,
            "audit_records": all_audit_records,
            "abstract_metrics": all_abstract_metrics,
            "run_dir": run_dir,
            "abstracts_dir": abstracts_dir
        }

    # ==========================================================================
    # 4. Master Ablation Sweeps (with Dynamic Cumulative Parameter Injection)
    # ==========================================================================
    def run_weight_combination_sweep(
        self,
        step_size: float = 0.10,
        default_threshold: float = 0.38,
        gating_mode: str = "weighted",
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        test_name: str = "Weight Combination Sensitivity",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 1: w_nov from 0.0 to 1.0 in steps of 0.10"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 1: {test_name} (Dataset: {ds_name}, Step: {step_size}) ---")
        results = []
        weights = np.arange(0.0, 1.0 + 1e-5, step_size)
        for w_nov in weights:
            w_nov = round(float(w_nov), 2)
            w_unc = round(1.0 - w_nov, 2)
            run_name = f"run_nov_{w_nov:.2f}_unc_{w_unc:.2f}"
            res = self.evaluate_configuration(
                gating_mode=gating_mode,
                w_novelty=w_nov,
                w_uncertainty=w_unc,
                uncertainty_method=uncertainty_method,
                novelty_metric=novelty_metric,
                unreliability_threshold=default_threshold,
                log_run_files=log_run_files,
                test_name=test_name,
                dataset_name=ds_name,
                run_name=run_name
            )
            results.append(res)
            print(f"  • w_nov={w_nov:.2f}, w_unc={w_unc:.2f} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Esc: {res['efficiency']['escalation_rate_pct']:5.1f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    def run_gating_operator_sweep(
        self,
        default_threshold: float = 0.38,
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        test_name: str = "Gating Fusion Operators",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 2: Gating Operators (weighted, or, and, prob_or, copula) using best discovered weights"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 2: {test_name} (Dataset: {ds_name}, w_nov={w_novelty:.2f}, w_unc={w_uncertainty:.2f}) ---")
        operators = ["weighted", "or", "and", "prob_or", "copula"]
        results = []
        for op in operators:
            run_name = f"run_operator_{op}"
            res = self.evaluate_configuration(
                gating_mode=op,
                w_novelty=w_novelty,
                w_uncertainty=w_uncertainty,
                uncertainty_method=uncertainty_method,
                novelty_metric=novelty_metric,
                unreliability_threshold=default_threshold,
                log_run_files=log_run_files,
                test_name=test_name,
                dataset_name=ds_name,
                run_name=run_name
            )
            results.append(res)
            print(f"  • Operator: {op:<10} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Acc: {res['metrics']['span_accuracy']*100:6.2f}% | Esc: {res['efficiency']['escalation_rate_pct']:5.1f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    def run_uncertainty_methods_sweep(
        self,
        default_threshold: float = 0.38,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        novelty_metric: str = "lof",
        test_name: str = "Uncertainty Estimators",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 3: Uncertainty Estimators (mcd, pe, margin, lc) using best discovered operator & weights"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 3: {test_name} (Dataset: {ds_name}, Operator: {gating_mode.upper()}) ---")
        methods = ["mcd", "pe", "margin", "lc"]
        results = []
        for m in methods:
            run_name = f"run_unc_{m}"
            res = self.evaluate_configuration(
                uncertainty_method=m,
                gating_mode=gating_mode,
                w_novelty=w_novelty,
                w_uncertainty=w_uncertainty,
                novelty_metric=novelty_metric,
                unreliability_threshold=default_threshold,
                log_run_files=log_run_files,
                test_name=test_name,
                dataset_name=ds_name,
                run_name=run_name
            )
            results.append(res)
            print(f"  • Method: {m:<8} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Precision: {res['metrics']['micro_precision']*100:6.2f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    def run_novelty_metrics_sweep(
        self,
        default_threshold: float = 0.38,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        test_name: str = "Novelty Density Estimators",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 4: Novelty Estimators (lof, mahalanobis, cosine_knn) using best discovered configurations"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 4: {test_name} (Dataset: {ds_name}, Method: {uncertainty_method.upper()}) ---")
        metrics = ["lof", "mahalanobis", "cosine_knn"]
        results = []
        for nov in metrics:
            run_name = f"run_nov_{nov}"
            res = self.evaluate_configuration(
                novelty_metric=nov,
                gating_mode=gating_mode,
                w_novelty=w_novelty,
                w_uncertainty=w_uncertainty,
                uncertainty_method=uncertainty_method,
                unreliability_threshold=default_threshold,
                log_run_files=log_run_files,
                test_name=test_name,
                dataset_name=ds_name,
                run_name=run_name
            )
            results.append(res)
            print(f"  • Novelty Metric: {nov:<12} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Acc: {res['metrics']['span_accuracy']*100:6.2f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    def run_threshold_pareto_sweep(
        self,
        step_size: float = 0.05,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        test_name: str = "Threshold Sensitivity",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 5: Threshold Sweep [0.20, 0.60] for Pareto Frontier using all best base configurations"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 5: {test_name} & Pareto Sweep (Dataset: {ds_name}, Step: {step_size}) ---")
        thresholds = np.arange(0.20, 0.60 + 1e-5, step_size)
        results = []
        for t_val in thresholds:
            t_val = round(float(t_val), 2)
            run_name = f"run_thresh_{t_val:.2f}"
            res = self.evaluate_configuration(
                gating_mode=gating_mode,
                w_novelty=w_novelty,
                w_uncertainty=w_uncertainty,
                uncertainty_method=uncertainty_method,
                novelty_metric=novelty_metric,
                unreliability_threshold=t_val,
                log_run_files=log_run_files,
                test_name=test_name,
                dataset_name=ds_name,
                run_name=run_name
            )
            results.append(res)
            print(f"  • Thresh: {t_val:.2f} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Esc Rate: {res['efficiency']['escalation_rate_pct']:5.1f}% | Saved: {res['efficiency']['local_saved_pct']:5.1f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    def run_component_ablation_matrix(
        self,
        gating_mode: str = "weighted",
        w_novelty: float = 0.50,
        w_uncertainty: float = 0.50,
        uncertainty_method: str = "mcd",
        novelty_metric: str = "lof",
        unreliability_threshold: float = 0.38,
        margin_threshold: float = 0.50,
        test_name: str = "Architectural Component Ablation",
        dataset_name: Optional[str] = None,
        log_run_files: bool = True
    ) -> List[Dict[str, Any]]:
        """Sweep 6: Feature Component Ablation Switches from optimal configuration baseline"""
        ds_name = dataset_name or self.dataset_name
        print(f"\n--- Running Sweep 6: {test_name} (Dataset: {ds_name}, Baseline Tau: {unreliability_threshold:.2f}) ---")
        configs = [
            # 1. Reference Baselines
            ("SpanNER Baseline Only (Zero Escalations)", "run_comp_local_spanner_only", True, True, True, True),
            ("Vanilla Hybrid (All 4 Mechanisms OFF)", "run_comp_vanilla_hybrid", False, False, False, False),

            # 2. Single-Component Additions (+1 from Vanilla)
            ("+ Context Anchoring Only", "run_comp_plus_context_anchoring", False, True, False, False),
            ("+ Contrastive Exemplars Only", "run_comp_plus_contrastive_exemplars", True, False, False, False),
            ("+ Restricted Top-2 Only", "run_comp_plus_restricted_top2", False, False, True, False),
            ("+ Margin Safeguard Veto Only", "run_comp_plus_margin_safeguard", False, False, False, True),

            # 3. Pairwise Combinations (+2 Pair Synergies)
            ("+ Context Anchoring & Restricted Top-2", "run_comp_plus_anch_top2", False, True, True, False),
            ("+ Context Anchoring & Margin Safeguard", "run_comp_plus_anch_veto", False, True, False, True),
            ("+ Restricted Top-2 & Margin Safeguard", "run_comp_plus_top2_veto", False, False, True, True),
            ("+ Context Anchoring & Contrastive Exemplars", "run_comp_plus_exem_anch", True, True, False, False),

            # 4. Triple Combinations / Leave-One-Out (-1 from Full)
            ("w/o Contrastive Exemplars (Anch + Top2 + Veto)", "run_comp_no_contrastive_exemplars", False, True, True, True),
            ("w/o Context Anchoring (Exem + Top2 + Veto)", "run_comp_no_anchored_context", True, False, True, True),
            ("w/o Restricted Top-2 (Exem + Anch + Veto)", "run_comp_no_restricted_top2", True, True, False, True),
            ("w/o Margin Safeguard (Exem + Anch + Top2)", "run_comp_no_margin_safeguard", True, True, True, False),

            # 5. Full Optimal System
            ("Full LinkNER System (All 4 Components ON)", "run_comp_full_pipeline", True, True, True, True)
        ]
        results = []
        for name, run_name, ex, anc, top2, veto in configs:
            if "SpanNER Baseline" in name:
                res = self.evaluate_configuration(
                    gating_mode=gating_mode,
                    w_novelty=w_novelty,
                    w_uncertainty=w_uncertainty,
                    uncertainty_method=uncertainty_method,
                    novelty_metric=novelty_metric,
                    unreliability_threshold=1.00,
                    margin_threshold=margin_threshold,
                    log_run_files=log_run_files,
                    test_name=test_name,
                    dataset_name=ds_name,
                    run_name=run_name
                )
            else:
                res = self.evaluate_configuration(
                    gating_mode=gating_mode,
                    w_novelty=w_novelty,
                    w_uncertainty=w_uncertainty,
                    uncertainty_method=uncertainty_method,
                    novelty_metric=novelty_metric,
                    unreliability_threshold=unreliability_threshold,
                    margin_threshold=margin_threshold,
                    use_contrastive_exemplars=ex,
                    use_anchored_context=anc,
                    use_restricted_top2=top2,
                    use_margin_safeguard=veto,
                    log_run_files=log_run_files,
                    test_name=test_name,
                    dataset_name=ds_name,
                    run_name=run_name
                )
            res["component_name"] = name
            results.append(res)
            print(f"  • {name:<45} | Macro F1: {res['metrics']['macro_f1']*100:6.2f}% | Micro F1: {res['metrics']['micro_f1']*100:6.2f}% | Acc: {res['metrics']['span_accuracy']*100:6.2f}% | Saved Log: {res.get('run_dir', 'N/A')}")
        return results

    # ==========================================================================
    # 5. Master Report Exporter
    # ==========================================================================
    def export_master_ablation_report(
        self,
        weight_results: List[Dict[str, Any]],
        operator_results: List[Dict[str, Any]],
        unc_results: List[Dict[str, Any]],
        nov_results: List[Dict[str, Any]],
        thresh_results: List[Dict[str, Any]],
        comp_results: List[Dict[str, Any]],
        dataset_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Exports comprehensive human-readable master report + CSVs + optimal configuration JSON.
        """
        ds_name = dataset_name or self.dataset_name
        ablation_dir = os.path.join(self.output_base_dir, "Master Reports", ds_name)
        sweeps_dir = os.path.join(ablation_dir, "sweeps")
        os.makedirs(sweeps_dir, exist_ok=True)

        report_txt_path = os.path.join(ablation_dir, "ablation_summary_report.txt")
        report_json_path = os.path.join(ablation_dir, "ablation_summary_metrics.json")
        optimal_json_path = os.path.join(ablation_dir, "optimal_configuration.json")

        # Find best configuration across sweeps by Macro F1
        all_evals = weight_results + operator_results + unc_results + nov_results + thresh_results + comp_results
        best_cfg = max(all_evals, key=lambda x: x["metrics"]["macro_f1"])

        with open(optimal_json_path, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=4)

        # Helper for table formatting
        def fmt_row(items: List[str], widths: List[int]) -> str:
            return " | ".join(f"{it:<{w}}" for it, w in zip(items, widths))

        lines = []
        lines.append("=" * 110)
        lines.append("🔬 COMPREHENSIVE ABLATION STUDY MASTER REPORT")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Dataset: {ds_name} | Total Sweeps Evaluated: {len(all_evals)}")
        lines.append("=" * 110)
        lines.append("")

        lines.append("-" * 110)
        lines.append("🏆 1. RECOMMENDED OPTIMAL CONFIGURATION")
        lines.append("-" * 110)
        c = best_cfg["config"]
        m = best_cfg["metrics"]
        lines.append(f"  • Gating Operator     : {c['gating_mode'].upper()}")
        lines.append(f"  • Weight Combination  : w_novelty={c['w_novelty']:.2f}, w_uncertainty={c['w_uncertainty']:.2f}")
        lines.append(f"  • Uncertainty Metric  : {c['uncertainty_method'].upper()} (MCD)")
        lines.append(f"  • Novelty Metric      : {c['novelty_metric'].upper()} (LOF)")
        lines.append(f"  • Threshold (Tau)     : {c['threshold']:.2f}")
        lines.append(f"  • Margin Veto Thresh  : {c['margin_threshold']:.2f}")
        lines.append(f"  • Best Macro F1-Score : {m['macro_f1']*100:.2f}% (SpanNER Gain: {m['delta_macro_f1']*100:+.2f}%)")
        lines.append(f"  • Best Micro F1-Score : {m['micro_f1']*100:.2f}% (SpanNER Gain: {m['delta_micro_f1']*100:+.2f}%)")
        lines.append(f"  • Micro Precision / R : {m['micro_precision']*100:.2f}% / {m['micro_recall']*100:.2f}%")
        lines.append(f"  • Span-Level Accuracy : {m['span_accuracy']*100:.2f}%")
        lines.append(f"  • Local Computation % : {best_cfg['efficiency']['local_saved_pct']:.2f}% saved locally")
        lines.append(f"  • Run Log Location    : {best_cfg.get('run_dir', 'N/A')}")
        lines.append("")

        # Section 2: Weight Sweeps
        lines.append("-" * 110)
        lines.append("2. WEIGHT COMBINATION SENSITIVITY (w_novelty vs. w_uncertainty)")
        lines.append("-" * 110)
        w_headers = ["w_nov", "w_unc", "Macro F1", "Micro F1", "Micro Prec", "Micro Rec", "Accuracy", "Esc Rate %", "Saved %"]
        w_widths = [7, 7, 10, 10, 12, 11, 10, 12, 10]
        lines.append(fmt_row(w_headers, w_widths))
        lines.append("-" * 110)
        for r in weight_results:
            c_r = r["config"]; m_r = r["metrics"]; e_r = r["efficiency"]
            lines.append(fmt_row([
                f"{c_r['w_novelty']:.2f}", f"{c_r['w_uncertainty']:.2f}",
                f"{m_r['macro_f1']*100:6.2f}%", f"{m_r['micro_f1']*100:6.2f}%",
                f"{m_r['micro_precision']*100:6.2f}%", f"{m_r['micro_recall']*100:6.2f}%",
                f"{m_r['span_accuracy']*100:6.2f}%", f"{e_r['escalation_rate_pct']:6.2f}%",
                f"{e_r['local_saved_pct']:6.2f}%"
            ], w_widths))
        lines.append("")

        # Section 3: Gating Operator Comparison
        lines.append("-" * 110)
        lines.append("3. GATING FUSION OPERATORS COMPARISON")
        lines.append("-" * 110)
        op_headers = ["Operator", "Macro F1", "Micro F1", "Micro Prec", "Micro Rec", "Accuracy", "Esc Rate %"]
        op_widths = [15, 10, 10, 12, 11, 10, 12]
        lines.append(fmt_row(op_headers, op_widths))
        lines.append("-" * 110)
        for r in operator_results:
            c_r = r["config"]; m_r = r["metrics"]; e_r = r["efficiency"]
            lines.append(fmt_row([
                c_r["gating_mode"].upper(),
                f"{m_r['macro_f1']*100:6.2f}%", f"{m_r['micro_f1']*100:6.2f}%",
                f"{m_r['micro_precision']*100:6.2f}%", f"{m_r['micro_recall']*100:6.2f}%",
                f"{m_r['span_accuracy']*100:6.2f}%", f"{e_r['escalation_rate_pct']:6.2f}%"
            ], op_widths))
        lines.append("")

        # Section 4: Architectural Component Contribution Matrix
        lines.append("-" * 110)
        lines.append("4. ARCHITECTURAL COMPONENT ABLATION MATRIX (Feature Contributions)")
        lines.append("-" * 110)
        comp_headers = ["Configuration / Component Switch", "Macro F1", "Micro F1", "Micro Prec", "Accuracy", "Delta F1"]
        comp_widths = [45, 10, 10, 12, 10, 12]
        lines.append(fmt_row(comp_headers, comp_widths))
        lines.append("-" * 110)
        for r in comp_results:
            name = r.get("component_name", "Component")
            m_r = r["metrics"]
            lines.append(fmt_row([
                name,
                f"{m_r['macro_f1']*100:6.2f}%", f"{m_r['micro_f1']*100:6.2f}%",
                f"{m_r['micro_precision']*100:6.2f}%", f"{m_r['span_accuracy']*100:6.2f}%",
                f"{m_r['delta_macro_f1']*100:+6.2f}%"
            ], comp_widths))
        lines.append("=" * 110)

        # Write Summary Files
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_configuration": best_cfg,
                "weights_sweep": weight_results,
                "gating_operators_sweep": operator_results,
                "uncertainty_methods_sweep": unc_results,
                "novelty_metrics_sweep": nov_results,
                "threshold_pareto_sweep": thresh_results,
                "component_ablations": comp_results
            }, f, indent=4)

        # Export CSVs
        import csv
        with open(os.path.join(sweeps_dir, "01_weight_sweep.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["w_novelty", "w_uncertainty", "macro_f1", "micro_f1", "micro_precision", "micro_recall", "accuracy", "escalation_rate_pct"])
            for r in weight_results:
                writer.writerow([r["config"]["w_novelty"], r["config"]["w_uncertainty"], r["metrics"]["macro_f1"], r["metrics"]["micro_f1"], r["metrics"]["micro_precision"], r["metrics"]["micro_recall"], r["metrics"]["span_accuracy"], r["efficiency"]["escalation_rate_pct"]])

        with open(os.path.join(sweeps_dir, "05_threshold_pareto.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "macro_f1", "micro_f1", "micro_precision", "micro_recall", "accuracy", "escalation_rate_pct", "local_saved_pct"])
            for r in thresh_results:
                writer.writerow([r["config"]["threshold"], r["metrics"]["macro_f1"], r["metrics"]["micro_f1"], r["metrics"]["micro_precision"], r["metrics"]["micro_recall"], r["metrics"]["span_accuracy"], r["efficiency"]["escalation_rate_pct"], r["efficiency"]["local_saved_pct"]])

        with open(os.path.join(sweeps_dir, "06_component_ablation.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "macro_f1", "micro_f1", "micro_precision", "micro_recall", "accuracy", "delta_macro_f1"])
            for r in comp_results:
                writer.writerow([r.get("component_name", ""), r["metrics"]["macro_f1"], r["metrics"]["micro_f1"], r["metrics"]["micro_precision"], r["metrics"]["micro_recall"], r["metrics"]["span_accuracy"], r["metrics"]["delta_macro_f1"]])

        return report_txt_path, report_json_path
