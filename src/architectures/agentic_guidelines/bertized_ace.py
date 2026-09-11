"""
BERTized Agentic Context Engineering (ACE) Pipeline.
Coordinates 3 specialized agents operating in a post-BERT candidate proposal setting:
  1. Generator (ACE_Generator_v2): Refines SpanNER candidate labels using dynamic guidelines and BERT evidence.
  2. Reflector (ACE_Reflector_v2): Diagnoses Generator errors exclusively on in-scope BERT candidate spans,
     tracing root causes and deriving key insights.
  3. Curator (ACE_Curator_v2): Screens insights against Base Rules and genericity tests, formulating ADD/MODIFY operations.
Managed by DynamicGuidelinesManager for persistent vector-indexed guideline storage and automated pruning.
"""

import os
import re
import json
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.common.llm_client import generate_llm_response as call_llm
from src.common.label_mapping import canonicalize_label
from src.common.audit_logger import classify_10_outcome_str
from src.architectures.agentic_guidelines.manager import DynamicGuidebookManager, DynamicGuidelinesManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines"


def _retry_agent_call(fn, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs):
    """Executes an agent call with exponential backoff for transient LLM/gateway errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
    raise last_err


def clean_json_string(raw_str: str) -> str:
    """Strips markdown code fences and cleans trailing commas for robust JSON parsing."""
    cleaned = raw_str.strip()
    if "```" in cleaned:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
        if match:
            cleaned = match.group(1).strip()
    return cleaned


def format_candidate_alignment_text(
    candidate_spans: List[Dict[str, Any]],
    generator_output: Dict[str, Any],
    use_outcome_class: bool = True
) -> str:
    """
    Builds the Candidate Span Alignment table matching the Reflector prompt format:
    [Span Text] [Tab] [Ground_Truth] [Tab] [BERT_LABEL] [Tab] [GENERATOR_LABEL] [Tab] [Outcome_Class]
    """
    ans_by_text = {}
    for a in generator_output.get("final_answer", []):
        ans_by_text[a.get("entity", "").strip().lower()] = a

    if use_outcome_class:
        alignment_rows = [
            "Span Text\tGround_Truth\tBERT_LABEL\tGENERATOR_LABEL\tOutcome_Class"
        ]
    else:
        alignment_rows = [
            "Span Text\tGround_Truth_Tag\tFinal_Label\tBERT_Predicted_Label\tComparison_Label"
        ]

    for idx, c in enumerate(candidate_spans):
        s_text = c.get("span_text", c.get("entity", ""))
        gt_tag = canonicalize_label(c.get("gt_label", "O"), use_new_labels=True)
        bert_pred = canonicalize_label(c.get("predicted_label", c.get("spanner_label", "O")), use_new_labels=True)

        ans = ans_by_text.get(s_text.strip().lower())
        if not ans and idx < len(generator_output.get("final_answer", [])):
            ans = generator_output.get("final_answer")[idx]

        final_lbl = canonicalize_label(ans.get("final_label", bert_pred) if ans else bert_pred, use_new_labels=True)

        comp = classify_10_outcome_str(
            gt_label=gt_tag,
            spanner_label=bert_pred,
            final_label=final_lbl,
            escalated=True
        )

        if use_outcome_class:
            alignment_rows.append(f"{s_text}\t{gt_tag}\t{bert_pred}\t{final_lbl}\t{comp}")
        else:
            alignment_rows.append(f"{s_text}\t{gt_tag}\t{final_lbl}\t{bert_pred}\t{comp}")

    return "\n".join(alignment_rows)


import csv

# ==============================================================================
# 1. BertizedGenerator (ACE_Generator_v2)
# ==============================================================================
class BertizedGenerator:
    """
    Generator agent operating in post-BERT correction setting.
    Arbitrates candidate spans using BERT evidence (predicted_label, uncertainty, novelty, margin)
    and current dynamic guidelines.
    """
    def __init__(self, model_name: str = "gemini-cli", backend: str = "api", prompt_path: Optional[str] = None, max_tokens: int = 16384):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        default_prompt = PROMPTS_DIR / "ACE_Generator_v3_bob.txt" if (PROMPTS_DIR / "ACE_Generator_v3_bob.txt").exists() else PROMPTS_DIR / "ACE_Generator_v2.txt"
        file_path = Path(prompt_path) if prompt_path else default_prompt
        with open(file_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def run(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        dynamicGuidelines: str = "",
        save_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Generator reasoning on BERT candidate spans.
        """
        # Format candidate spans with BERT evidence for prompt
        cand_list = []
        for c in candidate_spans:
            s_text = c.get("span_text", c.get("entity", ""))
            p_lbl = c.get("predicted_label", c.get("spanner_label", "O"))
            s_lbl = c.get("second_best_label", "O")
            unc = float(c.get("uncertainty", c.get("u_score", 0.0)))
            nov = float(c.get("novelty_score", c.get("novelty", 0.0)))
            margin = float(c.get("margin", 0.0))
            cand_list.append({
                "entity": s_text,
                "predicted_label": p_lbl,
                "second_best_label": s_lbl,
                "uncertainty": round(unc, 4),
                "novelty_score": round(nov, 4),
                "margin": round(margin, 4)
            })

        candidate_spans_str = json.dumps(cand_list, indent=2)
        guidelines_str = dynamicGuidelines.strip() if dynamicGuidelines and dynamicGuidelines.strip() else "No dynamicGuidelines available"

        prompt = self.template.replace("{abstract}", abstract) \
                              .replace("{candidate_spans_str}", candidate_spans_str) \
                              .replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', guidelines_str) \
                              .replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', guidelines_str) \
                              .replace("{dynamicGuidebook}", guidelines_str) \
                              .replace("{dynamicGuidelines}", guidelines_str)

        response = call_llm(prompt, self.model_name, "Generator", self.backend, max_tokens=self.max_tokens)

        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write(response)

        # Parse JSON output
        clean_resp = clean_json_string(response)
        try:
            parsed = json.loads(clean_resp)
        except Exception:
            # Fallback regex extraction
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', clean_resp)
            reasoning = reasoning_match.group(1) if reasoning_match else clean_resp
            bullets = re.findall(r'"([A-Z]{2,4}-\d{4})"', clean_resp)
            parsed = {"reasoning": reasoning, "bullet_ids": bullets, "final_answer": []}

        reasoning = parsed.get("reasoning", "")
        bullet_ids = parsed.get("bullet_ids", [])
        final_ans = parsed.get("final_answer", [])

        # Standardize final_answer items
        standardized_answers = []
        if isinstance(final_ans, list):
            for item in final_ans:
                if isinstance(item, dict):
                    ent_text = item.get("entity", "").strip()
                    bert_lbl = item.get("bert_predicted_label", item.get("predicted_label", "O"))
                    final_lbl = canonicalize_label(item.get("final_label", "O"), use_new_labels=True)
                    rationale = item.get("rationale", "")
                    standardized_answers.append({
                        "entity": ent_text,
                        "bert_predicted_label": bert_lbl,
                        "final_label": final_lbl,
                        "rationale": rationale
                    })

        return {
            "reasoning": reasoning,
            "bullet_ids": bullet_ids,
            "final_answer": standardized_answers,
            "raw_response": response
        }


# ==============================================================================
# 2. BertizedReflector (ACE_Reflector_v2)
# ==============================================================================
class BertizedReflector:
    """
    Reflector agent diagnosing Generator errors exclusively on in-scope BERT candidate spans.
    Produces error identification table, root cause analysis, and candidate key insight.
    """
    def __init__(self, model_name: str = "gemini-cli", backend: str = "api", prompt_path: Optional[str] = None, max_tokens: int = 16384):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        default_prompt = PROMPTS_DIR / "ACE_Reflector_v3_bob.txt" if (PROMPTS_DIR / "ACE_Reflector_v3_bob.txt").exists() else PROMPTS_DIR / "ACE_Reflector_v2.txt"
        file_path = Path(prompt_path) if prompt_path else default_prompt
        with open(file_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def run(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        generator_output: Dict[str, Any],
        dynamicGuidelines: Optional[Dict[str, Any]] = None,
        save_output_path: Optional[str] = None,
        alignment_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Reflector diagnosis by building the Candidate Span Alignment table.
        Format: [Span Text] [Tab] [Ground_Truth] [Tab] [BERT_LABEL] [Tab] [GENERATOR_LABEL] [Tab] [Outcome_Class]
        """
        if alignment_text is None:
            use_legacy = ("Comparison_Label" in self.template and "Outcome_Class" not in self.template)
            alignment_text = format_candidate_alignment_text(
                candidate_spans=candidate_spans,
                generator_output=generator_output,
                use_outcome_class=not use_legacy
            )

        used_bullet_ids = generator_output.get("bullet_ids", [])
        bullet_ids_str = json.dumps(used_bullet_ids) if used_bullet_ids else "None"

        prompt = self.template.replace("{alignment_text}", alignment_text) \
                              .replace('{predicted.get("reasoning", "")}', generator_output.get("reasoning", "")) \
                              .replace('{predicted.get("bullet_ids", "")}', bullet_ids_str)

        response = call_llm(prompt, self.model_name, "Reflector", self.backend, max_tokens=self.max_tokens)

        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write(response)

        clean_resp = clean_json_string(response)
        try:
            parsed = json.loads(clean_resp)
        except Exception:
            parsed = {
                "reasoning": clean_resp,
                "error_identification": "None",
                "root_cause_analysis": "None",
                "correct_approach": "None",
                "key_insight": "None",
                "key_insight_section": "ANF",
                "bullet_tags": []
            }

        return parsed


# ==============================================================================
# 3. BertizedCurator (ACE_Curator_v2)
# ==============================================================================
class BertizedCurator:
    """
    Curator agent screening candidate insights against Base Rules and 4 genericity tests.
    Produces ADD or MODIFY operations (or Safe-Fail empty operations).
    """
    def __init__(self, model_name: str = "gemini-cli", backend: str = "api", prompt_path: Optional[str] = None, max_tokens: int = 16384):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        default_prompt = PROMPTS_DIR / "ACE_Curator_v3_bob.txt" if (PROMPTS_DIR / "ACE_Curator_v3_bob.txt").exists() else PROMPTS_DIR / "ACE_Curator_v2.txt"
        file_path = Path(prompt_path) if prompt_path else default_prompt
        with open(file_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def run(
        self,
        dynamicGuidelines: str,
        reflection: Dict[str, Any],
        save_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Curator policy validation.
        """
        key_insight = reflection.get("key_insight", "None")
        if not key_insight or key_insight.strip().lower() in ("none", "null", ""):
            return {
                "reasoning": "Safe-Fail: Reflector produced no key_insight to curate.",
                "operations": []
            }

        reflection_str = json.dumps(reflection, indent=2)
        guidelines_str = dynamicGuidelines.strip() if dynamicGuidelines and dynamicGuidelines.strip() else "No dynamicGuidelines available"

        prompt = self.template.replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', guidelines_str) \
                              .replace("{dynamicGuidebook}", guidelines_str) \
                              .replace("{dynamicGuidelines}", guidelines_str) \
                              .replace("{json.dumps(reflection, indent=2)}", reflection_str) \
                              .replace("{reflection}", reflection_str)

        response = call_llm(prompt, self.model_name, "Curator", self.backend, max_tokens=self.max_tokens)

        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write(response)

        clean_resp = clean_json_string(response)
        try:
            parsed = json.loads(clean_resp)
        except Exception:
            parsed = {
                "reasoning": clean_resp,
                "operations": []
            }

        if "operations" not in parsed or not isinstance(parsed["operations"], list):
            parsed["operations"] = []

        return parsed


# ==============================================================================
# 4. BertizedACEPipeline (Unified Orchestrator)
# ==============================================================================
class BertizedACEPipeline:
    """
    End-to-End BERTized Agentic Context Engineering Pipeline:
      - Uses SpanNER + LOF to propose spans and extract BERT evidence.
      - Executes Generator_v2 -> Reflector_v2 -> Curator_v2 for training / curation.
      - Executes Generator_v2 for high-precision inference on unseen test abstracts.
    """
    def __init__(
        self,
        model_name: str = "gemini-cli",
        backend: str = "api",
        dynamicGuidelines_path: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        spanner_pipeline=None,
        max_tokens: int = 16384,
        generator_prompt_path: Optional[str] = None,
        reflector_prompt_path: Optional[str] = None,
        curator_prompt_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        self._lock = threading.RLock()
        self.generator = BertizedGenerator(model_name=model_name, backend=backend, prompt_path=generator_prompt_path, max_tokens=max_tokens)
        self.reflector = BertizedReflector(model_name=model_name, backend=backend, prompt_path=reflector_prompt_path, max_tokens=max_tokens)
        self.curator = BertizedCurator(model_name=model_name, backend=backend, prompt_path=curator_prompt_path, max_tokens=max_tokens)
        self.manager = DynamicGuidebookManager(
            guidebook_file_path=dynamicGuidelines_path,
            chroma_persist_directory=chroma_persist_directory
        )
        self.spanner = spanner_pipeline

    def save_guidebook(self, output_path: Optional[str] = None) -> bool:
        """Saves current dynamic guidebook JSON to disk with automatic timestamped backup."""
        with self._lock:
            return self.manager.save_dynamicGuidebook(backup=True)

    save_guidelines = save_guidebook

    def export_guidelines_markdown(self, output_path: str) -> str:
        """Exports human-readable Markdown summary of all active dynamic guidelines."""
        with self._lock:
            md_content = self.manager.get_dynamicGuidelines_for_generator_md()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return md_content

    def export_guidelines_csv(self, output_path: str) -> None:
        """Exports tabular CSV of all active guidelines with usage & helpful/harmful metrics."""
        with self._lock:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "section", "supercategory", "guideline", "usage_count", "helpful", "harmful", "modifications"])
                for bullet, supercategory, section_name in self.manager.store.iter_bullets():
                    metrics = bullet.get("usage_metrics", {})
                    writer.writerow([
                        bullet.get("id", bullet.get("bullet_id", "")),
                        section_name,
                        supercategory,
                        bullet.get("guideline", bullet.get("content", "")),
                        metrics.get("usage_count", bullet.get("usage_count", 0)),
                        metrics.get("helpful", bullet.get("helpful", 0)),
                        metrics.get("harmful", bullet.get("harmful", 0)),
                        metrics.get("modification_count", bullet.get("modification_count", 0))
                    ])

    def train_abstract(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        file_name: str = "abstract",
        save_outputs_dir: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Executes one full ACE curation cycle for a training abstract (Thread-Safe):
          1. Generator_v2 classifies candidate spans (instant intra-batch guideline visibility).
          2. Reflector_v2 analyzes errors & forms insight.
          3. Curator_v2 validates insight and formulates operations.
          4. Manager updates dynamicGuidelines store & bullet metrics with immediate persistence.
        """
        # Output paths
        gen_out = os.path.join(save_outputs_dir, f"{file_name}_generator.txt") if save_outputs_dir else None
        ref_out = os.path.join(save_outputs_dir, f"{file_name}_reflector.txt") if save_outputs_dir else None
        cur_out = os.path.join(save_outputs_dir, f"{file_name}_curator.txt") if save_outputs_dir else None

        # 1. Generator Step (Instant intra-batch visibility: reads fresh guidelines right before run)
        with self._lock:
            curr_guidelines_md = self.manager.get_dynamicGuidelines_for_generator_md()

        predicted = _retry_agent_call(
            self.generator.run,
            abstract=abstract,
            candidate_spans=candidate_spans,
            dynamicGuidelines=curr_guidelines_md,
            save_output_path=gen_out
        )

        # Track bullet usage under lock
        used_bullets = predicted.get("bullet_ids", [])
        if used_bullets:
            with self._lock:
                self.manager.increment_usage_counts(used_bullets)

        # 2. Reflector Step
        reflection = _retry_agent_call(
            self.reflector.run,
            abstract=abstract,
            candidate_spans=candidate_spans,
            generator_output=predicted,
            save_output_path=ref_out
        )

        # Apply bullet tag metrics (helpful/harmful) under lock
        bullet_tags = reflection.get("bullet_tags", [])
        if isinstance(bullet_tags, list):
            with self._lock:
                for b in bullet_tags:
                    if isinstance(b, dict) and "id" in b and "tag" in b:
                        b_id = b["id"]
                        tag = b["tag"].lower()
                        if tag == "helpful":
                            self.manager.update_bullet_metrics([b_id], helpful_delta=1)
                        elif tag == "harmful":
                            self.manager.update_bullet_metrics([b_id], harmful_delta=1)

        # 3. Curator Step (Fetch candidate guidelines under lock)
        query_text = reflection.get("key_insight") or reflection.get("root_cause_analysis") or abstract
        with self._lock:
            curator_guidelines_md = self.manager.get_dynamicGuidelines_for_curator_md(query_text=query_text)

        curator_output = _retry_agent_call(
            self.curator.run,
            dynamicGuidelines=curator_guidelines_md,
            reflection=reflection,
            save_output_path=cur_out
        )

        # 4. Process Curator Operations into Manager under lock & persist immediately
        if curator_output.get("operations"):
            with self._lock:
                self.manager.process_curator_operations(
                    curator_output=curator_output,
                    abstract=abstract,
                    reflection=reflection,
                    file_name=file_name
                )
                self.manager.save_dynamicGuidebook(backup=False)

        return predicted, reflection, curator_output

    def predict_abstract(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        file_name: str = "abstract",
        save_outputs_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Generator inference on unseen test abstract candidate spans using frozen guidelines (Thread-Safe).
        Returns arbitrated entity predictions with character offsets and confidence metadata.
        """
        gen_out = os.path.join(save_outputs_dir, f"{file_name}_generator.txt") if save_outputs_dir else None
        with self._lock:
            curr_guidelines_md = self.manager.get_dynamicGuidelines_for_generator_md()

        predicted = _retry_agent_call(
            self.generator.run,
            abstract=abstract,
            candidate_spans=candidate_spans,
            dynamicGuidelines=curr_guidelines_md,
            save_output_path=gen_out
        )

        ans_by_text = {}
        for a in predicted.get("final_answer", []):
            ans_by_text[a.get("entity", "").strip().lower()] = a

        final_predictions = []
        for idx, c in enumerate(candidate_spans):
            s_text = c.get("span_text", c.get("entity", ""))
            ans = ans_by_text.get(s_text.strip().lower())
            if not ans and idx < len(predicted.get("final_answer", [])):
                ans = predicted.get("final_answer")[idx]

            final_lbl = ans.get("final_label", c.get("predicted_label", "O")) if ans else c.get("predicted_label", "O")
            rationale = ans.get("rationale", "") if ans else "SpanNER Proposer Baseline"

            final_predictions.append({
                "span_text": s_text,
                "start_char": c.get("start_char", 0),
                "end_char": c.get("end_char", 0),
                "label": final_lbl,
                "spanner_label": c.get("predicted_label", c.get("spanner_label", "O")),
                "ace_label": final_lbl,
                "uncertainty": float(c.get("uncertainty", c.get("u_score", 0.0))),
                "novelty": float(c.get("novelty_score", c.get("novelty", 0.0))),
                "margin": float(c.get("margin", 0.0)),
                "rationale": rationale,
                "source": f"ACE Generator (Bullets: {', '.join(predicted.get('bullet_ids', [])) or 'Base Rules'})"
            })

        return final_predictions

    def train_batch_parallel(
        self,
        batch_items: List[Dict[str, Any]],
        max_workers: int = 5,
        save_outputs_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs a batch of abstracts through train_abstract in parallel using ThreadPoolExecutor.
        Each item in batch_items must provide: 'abstract' (or 'text'), 'candidates', and 'file_name' (or 'doc_id').
        Returns list of results in matching input order.
        """
        results = [None] * len(batch_items)

        def _worker(idx_item):
            idx, item = idx_item
            text = item.get("abstract", item.get("text", ""))
            cands = item.get("candidates", [])
            doc_id = item.get("file_name", item.get("doc_id", f"doc_{idx}"))
            pred, refl, cur = self.train_abstract(
                abstract=text,
                candidate_spans=cands,
                file_name=doc_id,
                save_outputs_dir=save_outputs_dir
            )
            return idx, {
                "doc_id": doc_id,
                "predicted": pred,
                "reflection": refl,
                "curator_output": cur,
                "spans_count": len(cands),
                "bullet_ids_used": pred.get("bullet_ids", []),
                "key_insight": refl.get("key_insight", "None"),
                "operations": cur.get("operations", [])
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, (i, itm)) for i, itm in enumerate(batch_items)]
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx] = res

        return results

    def predict_batch_parallel(
        self,
        batch_items: List[Dict[str, Any]],
        max_workers: int = 5,
        save_outputs_dir: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs a batch of abstracts through predict_abstract in parallel using ThreadPoolExecutor.
        Returns list of prediction lists in matching input order.
        """
        results = [None] * len(batch_items)

        def _worker(idx_item):
            idx, item = idx_item
            text = item.get("abstract", item.get("text", ""))
            cands = item.get("candidates", [])
            doc_id = item.get("file_name", item.get("doc_id", f"doc_{idx}"))
            preds = self.predict_abstract(
                abstract=text,
                candidate_spans=cands,
                file_name=doc_id,
                save_outputs_dir=save_outputs_dir
            )
            return idx, preds

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, (i, itm)) for i, itm in enumerate(batch_items)]
            for fut in as_completed(futures):
                idx, preds = fut.result()
                results[idx] = preds
        return results
