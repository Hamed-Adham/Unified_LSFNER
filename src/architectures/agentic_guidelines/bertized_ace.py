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

from src.common.llm_client import generate_llm_response as call_llm, last_call_metadata
from src.common.langfuse_tracker import get_langfuse_tracker
from src.common.label_mapping import canonicalize_label
from src.common.audit_logger import classify_10_outcome_str
from src.architectures.agentic_guidelines.manager import DynamicGuidebookManager, DynamicGuidelinesManager
from src.architectures.agentic_guidelines.gating import ACEGating, build_gated_anchored_abstract

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines"


def _retry_agent_call(fn, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs):
    """Executes an agent call with exponential backoff for transient LLM/gateway errors."""
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("No attempts were executed in _retry_agent_call.")


def clean_json_string(raw_str: str) -> str:
    """Strips markdown code fences and cleans trailing commas for robust JSON parsing."""
    cleaned = raw_str.strip()
    if "```" in cleaned:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
        if match:
            cleaned = match.group(1).strip()
    return cleaned


def build_anchored_abstract(abstract: str, candidate_spans: List[Dict[str, Any]]) -> str:
    """
    Inserts candidate span anchors into the abstract text using format:
    [id: {id}, span: ...]
    Supports nested and overlapping spans using boundary offsets or string searching.
    """
    events = []  # list of (pos, event_type, priority, id, tag)
    # event_type: 0 for close, 1 for open (so at same pos, close comes before open)
    # For open at same pos: longer span (larger span_len) should open first -> priority = -span_len
    # For close at same pos: shorter span (smaller span_len) should close first -> priority = span_len

    for idx, c in enumerate(candidate_spans, 1):
        cid = c.get("id", idx)
        s_text = c.get("span_text", c.get("entity", ""))
        if not s_text:
            continue

        start_char = None
        end_char = None
        if "boundaries" in c and isinstance(c["boundaries"], dict):
            start_char = c["boundaries"].get("start_char")
            end_char = c["boundaries"].get("end_char")
        if start_char is None or end_char is None:
            start_char = c.get("start_char")
            end_char = c.get("end_char")

        # Fallback to search in abstract if offsets not provided or invalid
        if start_char is None or end_char is None or start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue

        span_len = end_char - start_char
        # Close event: type 0, priority = span_len (shorter span closes first)
        events.append((end_char, 0, span_len, cid, "]"))
        # Open event: type 1, priority = -span_len (longer span opens first)
        events.append((start_char, 1, -span_len, cid, f"[id: {cid}, span: "))

    # Sort events by position ascending, event_type ascending (0=close before 1=open), priority ascending
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    out = []
    curr_pos = 0
    for pos, etype, prio, cid, tag in events:
        if pos > curr_pos:
            out.append(abstract[curr_pos:pos])
            curr_pos = pos
        out.append(tag)
    if curr_pos < len(abstract):
        out.append(abstract[curr_pos:])

    return "".join(out)


def format_candidate_alignment_text(
    candidate_spans: List[Dict[str, Any]],
    generator_output: Dict[str, Any],
    use_outcome_class: bool = True
) -> str:
    """
    Builds the Candidate Span Alignment table matching the Reflector prompt format:
    ID [Tab] [Span Text] [Tab] [Ground_Truth] [Tab] [BERT_LABEL] [Tab] [GENERATOR_LABEL] [Tab] [Outcome_Class]
    """
    ans_by_id = {}
    ans_by_text = {}
    gen_answers = generator_output.get("llm_answer", generator_output.get("final_answer", []))
    for a in gen_answers:
        if "id" in a:
            ans_by_id[a["id"]] = a
        ans_by_text[a.get("entity", "").strip().lower()] = a

    if use_outcome_class:
        alignment_rows = [
            "ID\tSpan Text\tGround_Truth\tBERT_LABEL\tGENERATOR_LABEL\tOutcome_Class"
        ]
    else:
        alignment_rows = [
            "ID\tSpan Text\tGround_Truth_Tag\tFinal_Label\tBERT_Predicted_Label\tComparison_Label"
        ]

    for idx, c in enumerate(candidate_spans, 1):
        cid = c.get("id", idx)
        s_text = c.get("span_text", c.get("entity", ""))
        gt_tag = canonicalize_label(c.get("gt_label", "O"), use_new_labels=True)
        bert_pred = canonicalize_label(c.get("predicted_label", c.get("spanner_label", "O")), use_new_labels=True)

        ans = ans_by_id.get(cid)
        if not ans:
            ans = ans_by_text.get(s_text.strip().lower())
        if not ans and idx - 1 < len(gen_answers):
            ans = gen_answers[idx - 1]

        raw_final = ans.get("llm_predicted_label", ans.get("final_label", bert_pred)) if ans else bert_pred
        final_lbl = canonicalize_label(raw_final, use_new_labels=True)

        comp = classify_10_outcome_str(
            gt_label=gt_tag,
            spanner_label=bert_pred,
            final_label=final_lbl,
            escalated=True
        )

        if use_outcome_class:
            alignment_rows.append(f"{cid}\t{s_text}\t{gt_tag}\t{bert_pred}\t{final_lbl}\t{comp}")
        else:
            alignment_rows.append(f"{cid}\t{s_text}\t{gt_tag}\t{final_lbl}\t{bert_pred}\t{comp}")

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
        save_output_path: Optional[str] = None,
        anchored_abstract: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Generator reasoning on BERT candidate spans.
        """
        # Format candidate spans with BERT evidence for prompt & assign IDs
        cand_list = []
        for idx, c in enumerate(candidate_spans, 1):
            cid = c.get("id", idx)
            c["id"] = cid
            s_text = c.get("span_text", c.get("entity", ""))
            p_lbl = c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O")))
            s_lbl = c.get("second_best_label", "O")
            unc = float(c.get("uncertainty", c.get("u_score", 0.0)))
            nov = float(c.get("novelty_score", c.get("novelty", 0.0)))
            margin = float(c.get("margin", 0.0))
            cand_list.append({
                "id": cid,
                "entity": s_text,
                "bert_predicted_label": p_lbl,
                "second_best_label": s_lbl,
                "uncertainty": round(unc, 4),
                "novelty_score": round(nov, 4),
                "margin": round(margin, 4)
            })

        if anchored_abstract is None:
            anchored_abstract = build_anchored_abstract(abstract, candidate_spans)
        candidate_spans_str = json.dumps(cand_list, indent=2)
        guidelines_str = dynamicGuidelines.strip() if dynamicGuidelines and dynamicGuidelines.strip() else "No dynamicGuidelines available"

        prompt = self.template.replace("{abstract}", anchored_abstract) \
                              .replace("{candidate_spans_str}", candidate_spans_str) \
                              .replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', guidelines_str) \
                              .replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', guidelines_str) \
                              .replace("{dynamicGuidebook}", guidelines_str) \
                              .replace("{dynamicGuidelines}", guidelines_str)

        tracker = get_langfuse_tracker()
        with tracker.trace_agent(
            agent_name="ACE-Generator",
            model_name=self.model_name,
            prompt=prompt,
            metadata={"backend": self.backend, "candidate_count": len(candidate_spans)}
        ) as gen_obs:
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
                parsed = {"reasoning": reasoning, "bullet_ids": bullets, "llm_answer": [], "final_answer": []}

            if isinstance(parsed, list):
                parsed = {
                    "reasoning": "Generator output parsed from list.",
                    "bullet_ids": [],
                    "llm_answer": parsed,
                    "final_answer": parsed
                }
            elif not isinstance(parsed, dict):
                parsed = {
                    "reasoning": str(parsed),
                    "bullet_ids": [],
                    "llm_answer": [],
                    "final_answer": []
                }

            reasoning = parsed.get("reasoning", "")
            bullet_ids = parsed.get("bullet_ids", [])
            final_ans = parsed.get("llm_answer", parsed.get("final_answer", []))

            # Standardize final_answer items
            standardized_answers = []
            if isinstance(final_ans, list):
                for idx, item in enumerate(final_ans):
                    if isinstance(item, dict):
                        cid = item.get("id")
                        ent_text = item.get("entity", "").strip()
                        bert_lbl = item.get("bert_predicted_label", item.get("predicted_label", "O"))
                        final_lbl = canonicalize_label(
                            item.get("llm_predicted_label", item.get("final_label", "O")),
                            use_new_labels=True
                        )
                        rationale = item.get("rationale", "")
                        ans_entry = {
                            "entity": ent_text,
                            "bert_predicted_label": bert_lbl,
                            "llm_predicted_label": final_lbl,
                            "final_label": final_lbl,
                            "rationale": rationale
                        }
                        if cid is not None:
                            ans_entry["id"] = cid
                        standardized_answers.append(ans_entry)

            result = {
                "reasoning": reasoning,
                "bullet_ids": bullet_ids,
                "llm_answer": standardized_answers,
                "final_answer": standardized_answers,
                "raw_response": response,
                "anchored_abstract": anchored_abstract
            }

            gen_obs.update(
                output=result,
                usage_details={
                    "input": last_call_metadata.get("prompt_tokens", 0),
                    "output": last_call_metadata.get("completion_tokens", 0),
                    "total": last_call_metadata.get("total_tokens", 0)
                },
                metadata={
                    "bullet_ids": bullet_ids,
                    "classified_count": len(standardized_answers),
                    "latency_seconds": last_call_metadata.get("latency_seconds", 0.0)
                }
            )

            return result


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
        Format: ID [Tab] [Span Text] [Tab] [Ground_Truth] [Tab] [BERT_LABEL] [Tab] [GENERATOR_LABEL] [Tab] [Outcome_Class]
        """
        anchored_abstract = generator_output.get("anchored_abstract")
        if not anchored_abstract:
            anchored_abstract = build_anchored_abstract(abstract, candidate_spans)

        if alignment_text is None:
            use_legacy = ("Comparison_Label" in self.template and "Outcome_Class" not in self.template)
            alignment_text = format_candidate_alignment_text(
                candidate_spans=candidate_spans,
                generator_output=generator_output,
                use_outcome_class=not use_legacy
            )

        used_bullet_ids = generator_output.get("bullet_ids", [])
        bullet_ids_str = json.dumps(used_bullet_ids) if used_bullet_ids else "None"

        prompt = self.template.replace("{abstract}", anchored_abstract) \
                              .replace("{alignment_text}", alignment_text) \
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
                "bullet_tags": []
            }

        # Normalize parsed into a standardized dictionary
        if isinstance(parsed, list):
            if not parsed:
                # 0 errors diagnosed: Generator was 100% accurate on in-scope candidates
                parsed = {
                    "reasoning": "No errors diagnosed on in-scope candidate spans.",
                    "error_identification": "None",
                    "root_cause_analysis": "None",
                    "correct_approach": "None",
                    "key_insight": "None",
                    "bullet_tags": [],
                    "diagnostic_items": []
                }
            else:
                all_bullet_tags = []
                key_insights = []
                root_causes = []
                reasonings = []
                error_identifications = []
                correct_approaches = []

                for item in parsed:
                    if isinstance(item, dict):
                        if "bullet_tags" in item and isinstance(item["bullet_tags"], list):
                            all_bullet_tags.extend(item["bullet_tags"])
                        if item.get("key_insight"):
                            key_insights.append(str(item["key_insight"]))
                        if item.get("root_cause_analysis"):
                            root_causes.append(str(item["root_cause_analysis"]))
                        if item.get("reasoning"):
                            reasonings.append(str(item["reasoning"]))
                        if item.get("error_identification"):
                            error_identifications.append(str(item["error_identification"]))
                        if item.get("correct_approach"):
                            correct_approaches.append(str(item["correct_approach"]))

                parsed = {
                    "reasoning": " | ".join(reasonings) if reasonings else clean_resp,
                    "error_identification": "\n".join(error_identifications) if error_identifications else "None",
                    "root_cause_analysis": " | ".join(root_causes) if root_causes else "None",
                    "correct_approach": " | ".join(correct_approaches) if correct_approaches else "None",
                    "key_insight": " | ".join(key_insights) if key_insights else "None",
                    "bullet_tags": all_bullet_tags,
                    "diagnostic_items": parsed
                }
        elif not isinstance(parsed, dict):
            parsed = {
                "reasoning": str(parsed),
                "error_identification": "None",
                "root_cause_analysis": "None",
                "correct_approach": "None",
                "key_insight": "None",
                "bullet_tags": []
            }
        else:
            if "bullet_tags" not in parsed or not isinstance(parsed["bullet_tags"], list):
                parsed["bullet_tags"] = []

        return parsed


# ==============================================================================
# 3. BertizedCurator (ACE_Curator_v2)
# ==============================================================================
class BertizedCurator:
    """
    Curator agent screening candidate insights against Base Rules and genericity tests.
    Supports both 'per_abstract' (single operation) and 'per_span' (multi-operation) modes.
    Produces ADD or MODIFY operations (or Safe-Fail empty operations).
    """
    def __init__(
        self,
        model_name: str = "gemini-cli",
        backend: str = "api",
        prompt_path: Optional[str] = None,
        max_tokens: int = 16384,
        curator_mode: str = "per_span"
    ):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        self.curator_mode = curator_mode

        if prompt_path:
            file_path = Path(prompt_path)
        else:
            if curator_mode == "per_span":
                file_path = PROMPTS_DIR / "ACE_Curator_v3_per_span.txt"
                if not file_path.exists():
                    file_path = PROMPTS_DIR / "ACE_Curator_v3_bob.txt"
            else:
                file_path = PROMPTS_DIR / "ACE_Curator_v3_bob.txt"
                if not file_path.exists():
                    file_path = PROMPTS_DIR / "ACE_Curator_v2.txt"

        self.prompt_path = str(file_path)
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
        # Extract diagnostic items array from reflection
        diag_items = reflection.get("diagnostic_items", []) if isinstance(reflection, dict) else (reflection if isinstance(reflection, list) else [])

        # Check if there are any valid diagnosed errors
        has_valid_items = False
        if isinstance(diag_items, list) and len(diag_items) > 0:
            for item in diag_items:
                if isinstance(item, dict) and item.get("key_insight") and item.get("key_insight").strip().lower() not in ("none", "null", ""):
                    has_valid_items = True
                    break
        elif isinstance(reflection, dict):
            k_ins = reflection.get("key_insight", "None")
            has_valid_items = bool(k_ins and k_ins.strip().lower() not in ("none", "null", ""))

        if not has_valid_items:
            return {
                "reasoning": "Safe-Fail: Reflector diagnosed no erroneous spans or valid key_insights to curate.",
                "operations": []
            }

        # Format reflection as the clean diagnostic array
        reflection_payload = diag_items if (isinstance(diag_items, list) and len(diag_items) > 0) else reflection
        reflection_str = json.dumps(reflection_payload, indent=2)

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

        if isinstance(parsed, list):
            parsed = {
                "reasoning": "Curator output parsed from list.",
                "operations": parsed
            }
        elif not isinstance(parsed, dict):
            parsed = {
                "reasoning": str(parsed),
                "operations": []
            }

        if "operations" not in parsed or not isinstance(parsed["operations"], list):
            parsed["operations"] = []

        # Enforce at most 1 operation if operating in per_abstract mode
        if self.curator_mode == "per_abstract" and len(parsed["operations"]) > 1:
            parsed["operations"] = parsed["operations"][:1]

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
        curator_mode: str = "per_span",
        enable_gating_train: bool = True,
        enable_gating_test: bool = True,
        gating_mode: str = "prob_or",
        unreliability_threshold: float = 0.35,
        novelty_threshold: float = 0.55,
        w_uncertainty: float = 0.70,
        w_novelty: float = 0.30,
        include_confident_as_readonly: bool = False,
        enable_margin_safeguard_veto: bool = True,
        margin_threshold: float = 0.50,
        p_o_threshold: float = 0.20,
    ):
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        self.curator_mode = curator_mode
        self._lock = threading.RLock()
        self.generator = BertizedGenerator(model_name=model_name, backend=backend, prompt_path=generator_prompt_path, max_tokens=max_tokens)
        self.reflector = BertizedReflector(model_name=model_name, backend=backend, prompt_path=reflector_prompt_path, max_tokens=max_tokens)
        self.curator = BertizedCurator(model_name=model_name, backend=backend, prompt_path=curator_prompt_path, max_tokens=max_tokens, curator_mode=curator_mode)
        self.manager = DynamicGuidebookManager(
            guidebook_file_path=dynamicGuidelines_path,
            chroma_persist_directory=chroma_persist_directory
        )
        self.spanner = spanner_pipeline
        self.enable_gating_train = bool(enable_gating_train)
        self.enable_gating_test = bool(enable_gating_test)
        self.gating = ACEGating(
            gating_mode=gating_mode,
            unreliability_threshold=unreliability_threshold,
            novelty_threshold=novelty_threshold,
            w_uncertainty=w_uncertainty,
            w_novelty=w_novelty,
            enable_margin_safeguard_veto=enable_margin_safeguard_veto,
            margin_threshold=margin_threshold,
            p_o_threshold=p_o_threshold,
            include_confident_as_readonly=include_confident_as_readonly,
        )

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
                writer.writerow([
                    "id", "ground_truth_label", "bert_label", "span", "guideline",
                    "usage_count", "helpful", "harmful", "modifications"
                ])
                for g in self.manager.store.iter_guidelines():
                    metrics = g.get("usage_metrics", {})
                    writer.writerow([
                        g.get("id", g.get("bullet_id", "")),
                        g.get("ground_truth_label", "O"),
                        g.get("bert_label", ""),
                        g.get("span", ""),
                        g.get("guideline", g.get("content", "")),
                        metrics.get("usage_count", g.get("usage_count", 0)),
                        metrics.get("helpful", g.get("helpful", 0)),
                        metrics.get("harmful", g.get("harmful", 0)),
                        metrics.get("modification_count", g.get("modification_count", len(g.get("history", []))))
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
        Applies selective candidate gating if enable_gating_train is True.
        """
        # Output paths
        gen_out = os.path.join(save_outputs_dir, f"{file_name}_generator.txt") if save_outputs_dir else None
        ref_out = os.path.join(save_outputs_dir, f"{file_name}_reflector.txt") if save_outputs_dir else None
        cur_out = os.path.join(save_outputs_dir, f"{file_name}_curator.txt") if save_outputs_dir else None

        # Determine target spans based on gating configuration
        if self.enable_gating_train:
            escalated_cands, confident_cands = self.gating.filter_candidates(candidate_spans)
            if not escalated_cands:
                # Fast-path safe-fail: zero ambiguous spans escalated
                predicted = {
                    "reasoning": "Gating bypass: all candidate spans classified with high SpanNER confidence.",
                    "bullet_ids": [],
                    "llm_answer": [],
                    "final_answer": [],
                    "raw_response": "{}",
                    "anchored_abstract": abstract,
                    "escalated_count": 0
                }
                reflection = {
                    "reasoning": "Gating bypass: zero candidate spans escalated.",
                    "error_identification": "None",
                    "root_cause_analysis": "None",
                    "correct_approach": "None",
                    "key_insight": "None",
                    "bullet_tags": []
                }
                curator_output = {
                    "reasoning": "Gating bypass: zero candidate spans escalated.",
                    "operations": []
                }
                return predicted, reflection, curator_output

            target_cands = escalated_cands
            anchored_abstract = build_gated_anchored_abstract(
                abstract=abstract,
                escalated_spans=escalated_cands,
                confident_spans=confident_cands,
                include_confident=self.gating.include_confident_as_readonly
            )
        else:
            target_cands = candidate_spans
            anchored_abstract = build_anchored_abstract(abstract, candidate_spans)

        # 1. Generator Step (Instant intra-batch visibility: reads fresh guidelines right before run)
        with self._lock:
            curr_guidelines_md = self.manager.get_dynamicGuidelines_for_generator_md()

        predicted = _retry_agent_call(
            self.generator.run,
            abstract=abstract,
            candidate_spans=target_cands,
            dynamicGuidelines=curr_guidelines_md,
            save_output_path=gen_out,
            anchored_abstract=anchored_abstract
        )
        predicted["escalated_count"] = len(target_cands)

        # Track bullet usage under lock
        used_bullets = predicted.get("bullet_ids", [])
        if used_bullets:
            with self._lock:
                self.manager.increment_usage_counts(used_bullets)

        # 2. Reflector Step (Evaluates on target candidate spans)
        reflection = _retry_agent_call(
            self.reflector.run,
            abstract=abstract,
            candidate_spans=target_cands,
            generator_output=predicted,
            save_output_path=ref_out
        )

        # Defensive guard for reflection
        if not isinstance(reflection, dict):
            if isinstance(reflection, list):
                all_tags = [t for it in reflection if isinstance(it, dict) for t in it.get("bullet_tags", [])]
                reflection = {
                    "reasoning": "Converted list",
                    "bullet_tags": all_tags,
                    "key_insight": "None" if not reflection else (reflection[0].get("key_insight", "None") if isinstance(reflection[0], dict) else "None")
                }
            else:
                reflection = {"reasoning": str(reflection), "bullet_tags": [], "key_insight": "None"}

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
        if self.curator_mode == "per_span" and reflection.get("diagnostic_items"):
            insights = [it.get("key_insight") for it in reflection["diagnostic_items"] if it.get("key_insight") and it.get("key_insight").strip().lower() not in ("none", "null", "")]
            query_text = " ".join(insights) if insights else (reflection.get("key_insight") or abstract)
        else:
            query_text = reflection.get("key_insight") or reflection.get("root_cause_analysis") or abstract

        with self._lock:
            curator_guidelines_md = self.manager.get_dynamicGuidelines_for_curator_md(query_text=query_text)

        curator_output = _retry_agent_call(
            self.curator.run,
            dynamicGuidelines=curator_guidelines_md,
            reflection=reflection,
            save_output_path=cur_out
        )

        # Defensive guard for curator_output
        if not isinstance(curator_output, dict):
            if isinstance(curator_output, list):
                curator_output = {"reasoning": "Converted list", "operations": curator_output}
            else:
                curator_output = {"reasoning": str(curator_output), "operations": []}

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
        Supports selective candidate gating and margin safeguard veto.
        Returns arbitrated entity predictions with character offsets and confidence metadata.
        """
        gen_out = os.path.join(save_outputs_dir, f"{file_name}_generator.txt") if save_outputs_dir else None

        if self.enable_gating_test:
            escalated_cands, confident_cands = self.gating.filter_candidates(candidate_spans)

            # Fast-path: 0 spans escalated to LLM
            if not escalated_cands:
                final_predictions = []
                for c in confident_cands:
                    s_text = c.get("span_text", c.get("entity", ""))
                    s_lbl = canonicalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))), use_new_labels=True)
                    final_predictions.append({
                        "span_text": s_text,
                        "start_char": c.get("start_char", 0),
                        "end_char": c.get("end_char", 0),
                        "label": s_lbl,
                        "spanner_label": s_lbl,
                        "ace_label": s_lbl,
                        "uncertainty": float(c.get("uncertainty", c.get("u_score", 0.0))),
                        "novelty": float(c.get("novelty_score", c.get("novelty", 0.0))),
                        "margin": float(c.get("margin", 0.0)),
                        "rationale": "High-confidence SpanNER prediction (bypassed LLM escalation)",
                        "source": "SpanNER Confident Baseline (Gating Fast-Path)",
                        "escalated_to_llm": False
                    })
                final_predictions.sort(key=lambda x: (x["start_char"], x["end_char"]))
                return final_predictions

            target_cands = escalated_cands
            anchored_abstract = build_gated_anchored_abstract(
                abstract=abstract,
                escalated_spans=escalated_cands,
                confident_spans=confident_cands,
                include_confident=self.gating.include_confident_as_readonly
            )
        else:
            target_cands = candidate_spans
            confident_cands = []
            anchored_abstract = build_anchored_abstract(abstract, candidate_spans)

        with self._lock:
            curr_guidelines_md = self.manager.get_dynamicGuidelines_for_generator_md()

        predicted = _retry_agent_call(
            self.generator.run,
            abstract=abstract,
            candidate_spans=target_cands,
            dynamicGuidelines=curr_guidelines_md,
            save_output_path=gen_out,
            anchored_abstract=anchored_abstract
        )

        ans_by_id = {}
        ans_by_text = {}
        pred_answers = predicted.get("llm_answer", predicted.get("final_answer", []))
        for a in pred_answers:
            if "id" in a:
                ans_by_id[a["id"]] = a
            ans_by_text[a.get("entity", "").strip().lower()] = a

        # 1. Process escalated candidate predictions
        escalated_predictions = []
        for idx, c in enumerate(target_cands):
            cid = c.get("id")
            s_text = c.get("span_text", c.get("entity", ""))
            ans = ans_by_id.get(cid) if cid is not None else None
            if not ans:
                ans = ans_by_text.get(s_text.strip().lower())
            if not ans and idx < len(pred_answers):
                ans = pred_answers[idx]

            s_lbl = canonicalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))), use_new_labels=True)
            raw_target = ans.get("llm_predicted_label", ans.get("final_label", s_lbl)) if ans else s_lbl
            raw_final_lbl = canonicalize_label(raw_target, use_new_labels=True)
            rationale = ans.get("rationale", "") if ans else "SpanNER Proposer Baseline"

            # Apply Margin Safeguard Veto
            final_lbl, veto_triggered, veto_reason = self.gating.apply_margin_veto(c, raw_final_lbl)
            final_lbl = canonicalize_label(final_lbl, use_new_labels=True)

            if veto_triggered:
                source = f"SpanNER Margin Veto ({veto_reason})"
                rationale = f"Veto applied: {veto_reason}. Original LLM: {raw_final_lbl}. {rationale}"
            else:
                used_b = predicted.get("bullet_ids", [])
                source = f"ACE Generator (Bullets: {', '.join(used_b) or 'Base Rules'})"

            escalated_predictions.append({
                "span_text": s_text,
                "start_char": c.get("start_char", 0),
                "end_char": c.get("end_char", 0),
                "label": final_lbl,
                "spanner_label": s_lbl,
                "ace_label": raw_final_lbl,
                "uncertainty": float(c.get("uncertainty", c.get("u_score", 0.0))),
                "novelty": float(c.get("novelty_score", c.get("novelty", 0.0))),
                "margin": float(c.get("margin", 0.0)),
                "rationale": rationale,
                "source": source,
                "escalated_to_llm": True
            })

        # 2. Process confident candidate pass-through predictions
        confident_predictions = []
        for c in confident_cands:
            s_text = c.get("span_text", c.get("entity", ""))
            s_lbl = canonicalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))), use_new_labels=True)
            confident_predictions.append({
                "span_text": s_text,
                "start_char": c.get("start_char", 0),
                "end_char": c.get("end_char", 0),
                "label": s_lbl,
                "spanner_label": s_lbl,
                "ace_label": s_lbl,
                "uncertainty": float(c.get("uncertainty", c.get("u_score", 0.0))),
                "novelty": float(c.get("novelty_score", c.get("novelty", 0.0))),
                "margin": float(c.get("margin", 0.0)),
                "rationale": "High-confidence SpanNER prediction (gated without LLM escalation)",
                "source": "SpanNER Confident Baseline (Gating Retained)",
                "escalated_to_llm": False
            })

        all_predictions = escalated_predictions + confident_predictions
        all_predictions.sort(key=lambda x: (x["start_char"], x["end_char"]))
        return all_predictions

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
                "predicted": pred if isinstance(pred, dict) else {},
                "reflection": refl if isinstance(refl, dict) else {},
                "curator_output": cur if isinstance(cur, dict) else {},
                "spans_count": len(cands),
                "escalated_count": pred.get("escalated_count", len(cands)) if isinstance(pred, dict) else len(cands),
                "bullet_ids_used": pred.get("bullet_ids", []) if isinstance(pred, dict) else [],
                "key_insight": refl.get("key_insight", "None") if isinstance(refl, dict) else "None",
                "operations": cur.get("operations", []) if isinstance(cur, dict) else []
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
