"""
JEV Gating Pipeline
===================

Executes the JEV System One gating algorithm comparing BERT/SpanNER candidate predictions
against JEV System One decisions.

Gating Algorithm Rule:
- Equality Check: `bert_label == jev_label`
  * If EQUAL: Gated locally (confident). Prediction accepted directly without LLM inference cost.
  * If NOT EQUAL: Escalated to LLM (disputed/ambiguous). Enriched with complete candidate details,
    context, and JEV decision priors for downstream LLM arbitration (e.g., ACE Generator / Dinasor-DinGen).

Output Format:
- `all_spans`: complete list of spans with assigned gating metadata.
"""

import os
import json
import logging
import re
import time
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.label_mapping import canonicalize_label, CANONICAL_10_LABELS
from src.common.audit_logger import classify_10_outcome_str
from src.common.llm_client import get_llm_provider
from src.architectures.jev_gating.jev_call import (
    JevDecisionResult,
    load_or_fetch_doc_jev_decisions,
    call_batch_jev_decisions,
    extract_sentence_context,
    format_guidelines_from_json,
    format_guideline_examples_from_json,
)

logger = logging.getLogger(__name__)


def normalize_category(label: Optional[str]) -> str:
    """Normalizes label string to canonical categories or 'O'."""
    if not label:
        return "O"
    clean = str(label).strip().replace(" ", "_")
    if clean.lower() in ("o", "non_lsf", "non-lsf", "none", "background", "other", "null", ""):
        return "O"
    return canonicalize_label(clean, use_new_labels=True)


class JevGatingPipeline:
    """
    Pipeline that runs JEV decision scoring and applies equality gating on candidate spans.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 10,
        prompt_template_path: Optional[Union[str, Path]] = None,
        questions_path: Optional[Union[str, Path]] = None,
        guidebook_path: Optional[Union[str, Path]] = None,
        guidelines: Optional[str] = None,
        examples_path: Optional[Union[str, Path]] = None,
        guideline_examples: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_workers = max_workers
        self.prompt_template_path = prompt_template_path
        self.questions_path = questions_path
        self.guidebook_path = guidebook_path
        self.guidelines = guidelines
        self.examples_path = examples_path
        self.guideline_examples = guideline_examples
        self.api_key = api_key

    def gate_document_candidates(
        self,
        doc_id: str,
        abstract_text: str,
        candidates: List[Dict[str, Any]],
        use_cache: bool = True,
        prompt_template_path: Optional[Union[str, Path]] = None,
        questions_path: Optional[Union[str, Path]] = None,
        guidebook_path: Optional[Union[str, Path]] = None,
        guidelines: Optional[str] = None,
        examples_path: Optional[Union[str, Path]] = None,
        guideline_examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs JEV decision classification and performs equality gating on candidate spans.
        """
        effective_cache = self.cache_dir if use_cache else None
        eff_prompt_path = prompt_template_path or self.prompt_template_path
        eff_questions_path = questions_path or self.questions_path
        eff_guidebook = guidebook_path or self.guidebook_path
        eff_guidelines = guidelines or self.guidelines
        
        eff_examples_path = examples_path or self.examples_path
        if not eff_examples_path and eff_guidebook:
            cand_ex = Path(eff_guidebook).parent / f"{Path(eff_guidebook).stem}_examples.json"
            if cand_ex.exists():
                eff_examples_path = cand_ex
        eff_examples = guideline_examples or self.guideline_examples

        # Fetch or compute JEV decisions for candidate spans
        jev_decisions_map = load_or_fetch_doc_jev_decisions(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans=candidates,
            cache_dir=effective_cache,
            max_workers=self.max_workers,
            prompt_template_path=eff_prompt_path,
            questions_path=eff_questions_path,
            api_key=self.api_key,
            guidelines=eff_guidelines,
            guidebook_path=eff_guidebook,
            guideline_examples=eff_examples,
            examples_path=eff_examples_path
        )

        local_remained: List[Dict[str, Any]] = []
        escale_to_llm: List[Dict[str, Any]] = []
        all_spans: List[Dict[str, Any]] = []

        for idx, cand in enumerate(candidates, 1):
            span_id = str(cand.get("id", cand.get("span_id", idx)))
            span_text = cand.get("span_text", cand.get("entity", cand.get("text", "")))
            
            # Boundaries
            boundaries = cand.get("boundaries", {})
            start_char = boundaries.get("start_char", cand.get("start_char", 0))
            end_char = boundaries.get("end_char", cand.get("end_char", len(span_text)))

            # Neural BERT / SpanNER evidence (flat or nested under spanner_evidence)
            sp_ev = cand.get("spanner_evidence", {})
            bert_label_raw = (
                cand.get("bert_predicted_label") or
                cand.get("predicted_label") or
                cand.get("spanner_label") or
                sp_ev.get("predicted_label") or
                "O"
            )
            bert_label = normalize_category(bert_label_raw)
            second_best = normalize_category(cand.get("second_best_label") or sp_ev.get("second_best_label", "O"))
            
            uncertainty = float(cand.get("uncertainty") or sp_ev.get("uncertainty", 0.0))
            novelty = float(cand.get("novelty_score") or cand.get("novelty") or sp_ev.get("novelty_score", 0.0))
            margin = float(cand.get("margin") or sp_ev.get("margin", 0.0))
            prob = float(cand.get("prob") or sp_ev.get("top1_prob", 0.0))

            # Ground truth if available
            gt_info = cand.get("ground_truth", {})
            gt_label = normalize_category(gt_info.get("gt_label") or cand.get("gt_label", "O"))

            # JEV System One Decision lookup (by span_id, 1-based index, or span_text)
            jev_dec = (
                jev_decisions_map.get(span_id) or
                jev_decisions_map.get(str(idx)) or
                jev_decisions_map.get(int(idx) if isinstance(idx, int) else None)
            )
            if not jev_dec:
                # Fallback search by span_text
                for d in jev_decisions_map.values():
                    if d.span_text.strip().lower() == span_text.strip().lower():
                        jev_dec = d
                        break

            if jev_dec:
                jev_label_raw = jev_dec.predicted_category
                jev_label = normalize_category(jev_label_raw)
                jev_is_lsf_prob = jev_dec.is_lsf_entity
                jev_uncertainty = jev_dec.gating_uncertainty
                jev_probs = jev_dec.category_probabilities
                jev_selected_guidelines = jev_dec.selected_guidelines
            else:
                jev_label = bert_label
                jev_is_lsf_prob = 1.0 if bert_label != "O" else 0.0
                jev_uncertainty = 0.0
                jev_probs = {bert_label: 1.0}
                jev_selected_guidelines = {}

            # Sentence Context
            sentence_ctx = extract_sentence_context(abstract_text, start_char, end_char)

            # Equality Gating Rule: bert_label == jev_label
            is_equal = (bert_label == jev_label)
            escalated = not is_equal

            if is_equal:
                gating_status = "LOCAL_RETAINED"
                gating_reason = f"Consensus Agreement: Both BERT and JEV agree on '{bert_label}'"
                assigned_label = bert_label
            else:
                gating_status = "ESCALATE_TO_LLM"
                gating_reason = f"Category Discrepancy: BERT proposed '{bert_label}' but JEV scored '{jev_label}' (diff={uncertainty:.2f})"
                assigned_label = bert_label  # Prior before LLM arbitration

            # Detailed record with all parameters passing to Generator LLM
            detailed_span = {
                "id": int(span_id) if span_id.isdigit() else span_id,
                "span_text": span_text,
                "start_char": start_char,
                "end_char": end_char,
                "sentence_context": sentence_ctx,
                "bert_predicted_label": bert_label,
                "spanner_label": bert_label,
                "second_best_label": second_best,
                "prob": prob,
                "uncertainty": uncertainty,
                "novelty_score": novelty,
                "margin": margin,
                "jev_predicted_label": jev_label,
                "jev_is_lsf_probability": jev_is_lsf_prob,
                "jev_gating_uncertainty": jev_uncertainty,
                "jev_category_probabilities": jev_probs,
                "jev_selected_guidelines": jev_selected_guidelines,
                "ground_truth_label": gt_label,
                "gating_status": gating_status,
                "gating_reason": gating_reason,
                "escalated_to_llm": escalated,
                "assigned_label": assigned_label
            }

            all_spans.append(detailed_span)
            if escalated:
                escale_to_llm.append(detailed_span)
            else:
                local_remained.append(detailed_span)

        total_spans = len(candidates)
        esc_count = len(escale_to_llm)
        rem_count = len(local_remained)
        esc_rate = (esc_count / max(1, total_spans)) * 100.0

        return {
            "doc_id": doc_id,
            "abstract_text": abstract_text,
            "summary": {
                "total_candidate_spans": total_spans,
                "local_remained_count": rem_count,
                "escalated_to_llm_count": esc_count,
                "local_retained_pct": round((rem_count / max(1, total_spans)) * 100.0, 2),
                "escalation_rate_pct": round(esc_rate, 2),
                "gating_rule": "bert_label == jev_label"
            },
            "local_remained": local_remained,
            "escale_to_llm": escale_to_llm,
            "all_spans": all_spans
        }


def run_jev_gating_pipeline(
    doc_id: str,
    abstract_text: str,
    candidates: List[Dict[str, Any]],
    cache_dir: Optional[Union[str, Path]] = None,
    max_workers: int = 10,
    prompt_template_path: Optional[Union[str, Path]] = None,
    questions_path: Optional[Union[str, Path]] = None,
    guidebook_path: Optional[Union[str, Path]] = None,
    guidelines: Optional[str] = None,
    examples_path: Optional[Union[str, Path]] = None,
    guideline_examples: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience functional API to run JEV gating on a single document.
    """
    pipeline = JevGatingPipeline(
        cache_dir=cache_dir,
        max_workers=max_workers,
        prompt_template_path=prompt_template_path,
        questions_path=questions_path,
        guidebook_path=guidebook_path,
        guidelines=guidelines,
        examples_path=examples_path,
        guideline_examples=guideline_examples
    )
    return pipeline.gate_document_candidates(
        doc_id=doc_id,
        abstract_text=abstract_text,
        candidates=candidates
    )


def build_gated_anchored_abstract(
    abstract: str,
    uncertain_spans: List[Dict[str, Any]],
    confident_spans: List[Dict[str, Any]]
) -> str:
    """
    Inserts anchors into abstract:
    - [id: {id}, span: {text}] for uncertain candidate spans to be evaluated by Generator
    - [ANCHOR: {text} ({label})] for high-confidence local spans serving as semantic context
    """
    events = []
    for c in uncertain_spans:
        cid = c["id"]
        s_text = c["span_text"]
        start_char = c.get("start_char", 0)
        end_char = c.get("end_char", len(s_text))

        if start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue

        span_len = end_char - start_char
        events.append((end_char, 0, span_len, cid, "]"))
        events.append((start_char, 1, -span_len, cid, f"[id: {cid}, span: "))

    for c in confident_spans:
        s_text = c["span_text"]
        lbl = c["bert_predicted_label"]
        start_char = c.get("start_char", 0)
        end_char = c.get("end_char", len(s_text))

        if start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue

        span_len = end_char - start_char
        events.append((end_char, 0, span_len, -1, f" ({lbl})]"))
        events.append((start_char, 1, -span_len, -1, "[ANCHOR: "))

    # Sort events: descending by character offset
    events.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    res = abstract
    for pos, _, _, _, tag in events:
        res = res[:pos] + tag + res[pos:]
    return res


def format_candidate_spans_with_jev_json(candidate_spans: List[Dict[str, Any]]) -> str:
    """Formats candidate spans with both BERT and JEV diagnostic evidence into JSON string."""
    evidence_list = []
    for c in candidate_spans:
        top_probs = dict(sorted(c.get("jev_category_probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:3])
        evidence_list.append({
            "id": c["id"],
            "span_text": c["span_text"],
            "bert_prior": {
                "predicted_label": c.get("bert_predicted_label", "O"),
                "confidence": round(float(c.get("prob", 0.0)), 4),
                "margin": round(float(c.get("margin", 0.0)), 4),
            },
            "jev_decision": {
                "predicted_label": c.get("jev_predicted_label", "O"),
                "is_lsf_entity_score": round(float(c.get("jev_is_lsf_probability", 0.0)), 4),
                "uncertainty_score": round(float(c.get("jev_gating_uncertainty", 0.0)), 4),
                "top_probabilities": top_probs,
                "selected_guidelines": c.get("jev_selected_guidelines", {})
            }
        })
    return json.dumps(evidence_list, indent=2)


def parse_ace_generator_response(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts reasoning block and JSON resolutions from ACE Generator response."""
    reasoning = ""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    json_str = raw_text
    json_block = re.search(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", raw_text, re.DOTALL)
    if json_block:
        json_str = json_block.group(1)
    else:
        bracket_match = re.search(r"(\[\s*\{.*\}\s*\])", raw_text, re.DOTALL)
        if bracket_match:
            json_str = bracket_match.group(1)

    parsed_items = []
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            parsed_items = parsed
    except Exception:
        item_matches = re.findall(r"\{[^{}]*\"id\"[^{}]*\}", raw_text, re.DOTALL)
        for im in item_matches:
            try:
                obj = json.loads(im)
                parsed_items.append(obj)
            except Exception:
                continue

    return reasoning, parsed_items


def generate_jev_ace_doc_report(
    doc_id: str,
    abstract_text: str,
    spans_data: List[Dict[str, Any]],
    out_file: Union[str, Path],
    reasoning: str = "",
    llm_raw_response: Optional[str] = None
) -> None:
    """Writes detailed per-document text report."""
    total_spans = len(spans_data)
    escalated_count = sum(1 for s in spans_data if s.get("escalated_to_llm", False))
    confident_count = total_spans - escalated_count

    lines = [
        "=" * 80,
        f"JEV-GATED ACE GENERATOR REPORT: Document {doc_id}",
        "=" * 80,
        f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Spans           : {total_spans}",
        f"Agreed (Local BERT)   : {confident_count} ({confident_count/max(1, total_spans)*100:.1f}%)",
        f"Disagreed (Escalated) : {escalated_count} ({escalated_count/max(1, total_spans)*100:.1f}%)",
        "",
        "-" * 80,
        "DOCUMENT ABSTRACT TEXT:",
        "-" * 80,
        abstract_text.strip(),
        "",
        "=" * 80,
        "CANDIDATE SPAN EVALUATIONS & ARBITRATION DETAILS:",
        "=" * 80,
    ]

    for s in spans_data:
        idx = s["id"]
        s_text = s["span_text"]
        s_start, s_end = s["start_char"], s["end_char"]
        gt_lbl = s.get("ground_truth_label", "O")
        bert_lbl = s.get("bert_predicted_label", "O")
        jev_lbl = s.get("jev_predicted_label", "O")
        final_lbl = s.get("assigned_label", bert_lbl)
        esc = s.get("escalated_to_llm", False)
        gating_status = "ESCALATED TO LLM" if esc else "LOCAL RETAINED"
        g_sels = s.get("jev_selected_guidelines", {})
        g_sel_str = ", ".join(f"{k}: {v}" for k, v in g_sels.items()) if g_sels else "None"

        lines.extend([
            f"\n[SPAN {idx}/{total_spans}] \"{s_text}\" (Offsets: [{s_start}:{s_end}])",
            f"  * Ground Truth (GT)   : {gt_lbl}",
            f"  * BERT Prior          : {bert_lbl} [Conf: {s.get('prob', 0.0):.4f}, Margin: {s.get('margin', 0.0):.4f}]",
            f"  * JEV System One      : {jev_lbl} [P(LSF): {s.get('jev_is_lsf_probability', 0.0):.4f}, Uncert: {s.get('jev_gating_uncertainty', 0.0):.4f}]",
            f"  * JEV Selected Rules  : {g_sel_str}",
            f"  * Gating Decision     : {gating_status}",
            f"  * Final Assigned Label: {final_lbl}",
            f"  * Status / Correctness: BERT Correct: {gt_lbl == bert_lbl} | Final Correct: {gt_lbl == final_lbl}",
            "-" * 80
        ])

    if llm_raw_response:
        lines.extend([
            "\n" + "=" * 80,
            "ACE GENERATOR RAW LLM RESPONSE:",
            "=" * 80,
            llm_raw_response
        ])

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_jev_gated_ace_pipeline_on_dataset(
    input_cache_dir: Union[str, Path],
    output_report_dir: Union[str, Path],
    jev_cache_dir: Union[str, Path],
    prompt_template_path: Optional[Union[str, Path]] = None,
    questions_path: Optional[Union[str, Path]] = None,
    ace_prompt_path: Optional[Union[str, Path]] = None,
    guidebook_path: Optional[Union[str, Path]] = None,
    guidelines: Optional[str] = None,
    examples_path: Optional[Union[str, Path]] = None,
    guideline_examples: Optional[str] = None,
    use_cache: bool = True,
    llm_backend: str = "9router",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs full end-to-end dataset arbitration: JEV Gating + Single-turn ACE Generator v3 LLM arbitration.
    """
    input_dir = Path(input_cache_dir)
    out_dir = Path(output_report_dir)
    j_cache_dir = Path(jev_cache_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    j_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt templates
    ace_template = ""
    if ace_prompt_path and Path(ace_prompt_path).exists():
        ace_template = Path(ace_prompt_path).read_text(encoding="utf-8")

    guidebook_content = ""
    if guidebook_path and Path(guidebook_path).exists():
        guidebook_content = Path(guidebook_path).read_text(encoding="utf-8").strip()

    llm = get_llm_provider(llm_backend)

    json_files = sorted([f for f in input_dir.glob("*.json") if f.stem.isdigit()])
    if limit:
        json_files = json_files[:limit]

    eff_examples_path = examples_path
    if not eff_examples_path and guidebook_path:
        cand_ex = Path(guidebook_path).parent / f"{Path(guidebook_path).stem}_examples.json"
        if cand_ex.exists():
            eff_examples_path = cand_ex

    pipeline = JevGatingPipeline(
        cache_dir=j_cache_dir if use_cache else None,
        prompt_template_path=prompt_template_path,
        questions_path=questions_path,
        guidebook_path=guidebook_path,
        guidelines=guidelines,
        examples_path=eff_examples_path,
        guideline_examples=guideline_examples
    )

    all_doc_results = []
    logger.info(f"Starting JEV-Gated ACE arbitration across {len(json_files)} documents...")

    for idx, f_path in enumerate(json_files, 1):
        with open(f_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_id = doc_data.get("doc_id", f_path.stem)
        abstract_text = doc_data.get("text", "")
        candidates = doc_data.get("spans", [])

        # Run JEV Gating step
        gate_res = pipeline.gate_document_candidates(
            doc_id=doc_id,
            abstract_text=abstract_text,
            candidates=candidates,
            use_cache=use_cache,
            prompt_template_path=prompt_template_path,
            questions_path=questions_path,
            guidebook_path=guidebook_path,
            guidelines=guidelines,
            examples_path=eff_examples_path,
            guideline_examples=guideline_examples
        )

        all_spans = gate_res["all_spans"]
        uncertain_spans = gate_res["escale_to_llm"]
        confident_spans = gate_res["local_remained"]

        raw_response = None
        reasoning_text = ""

        if uncertain_spans and ace_template:
            anchored_abs = build_gated_anchored_abstract(abstract_text, uncertain_spans, confident_spans)
            cand_json = format_candidate_spans_with_jev_json(uncertain_spans)

            prompt_content = ace_template
            prompt_content = prompt_content.replace("{abstract}", anchored_abs)
            prompt_content = prompt_content.replace("{candidate_spans_str}", cand_json)
            prompt_content = prompt_content.replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', guidebook_content)
            prompt_content = prompt_content.replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidebook}", guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidelines}", guidebook_content)
            prompt_content = prompt_content.replace("{{ABSTRACT_TEXT}}", anchored_abs)
            prompt_content = prompt_content.replace("{{CANDIDATE_SPANS_JSON}}", cand_json)
            prompt_content = prompt_content.replace("{{DYNAMIC_GUIDEBOOK}}", guidebook_content)

            try:
                raw_response = llm.generate(prompt=prompt_content, temperature=0.0, max_tokens=6128)
            except Exception as e:
                logger.error(f"LLM call failed for doc {doc_id}: {e}")
                raw_response = "[]"

            reasoning_text, parsed_items = parse_ace_generator_response(raw_response or "")
            res_by_id = {item.get("id"): item for item in parsed_items if "id" in item}

            for s in uncertain_spans:
                sid = s["id"]
                if sid in res_by_id:
                    res_obj = res_by_id[sid]
                    assigned = normalize_category(res_obj.get("category", res_obj.get("predicted_label", "O")))
                    s["assigned_label"] = assigned
                    s["final_label"] = assigned
                else:
                    s["assigned_label"] = s["bert_predicted_label"]
                    s["final_label"] = s["bert_predicted_label"]

        # Generate and save text report for this document
        doc_report_file = out_dir / f"{doc_id}.txt"
        generate_jev_ace_doc_report(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans_data=all_spans,
            out_file=doc_report_file,
            reasoning=reasoning_text,
            llm_raw_response=raw_response
        )

        all_doc_results.append({
            "doc_id": doc_id,
            "total_spans": len(all_spans),
            "escalated_count": len(uncertain_spans),
            "local_count": len(confident_spans),
        })

    logger.info(f"Arbitration completed across {len(json_files)} documents. Reports saved to {out_dir}")
    return {
        "output_report_dir": str(out_dir),
        "total_documents": len(all_doc_results),
        "total_escalated_spans": sum(d["escalated_count"] for d in all_doc_results),
        "total_local_spans": sum(d["local_count"] for d in all_doc_results)
    }
