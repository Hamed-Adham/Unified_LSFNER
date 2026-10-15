"""
Jev Gating & Classification Module (TypeSafe AI via OpenRouter Decisions API)
=============================================================================

This module interfaces with the Jev non-autoregressive "System One" decision model
via OpenRouter's `/api/alpha/decisions` endpoint.

For each candidate span (from SpanNER / BERT output in `data/cached`), Jev performs
a fast, calibrated single-pass evaluation for:
1. `is_lsf_entity` (noul binary probability: LSF vs O / background)
2. `category` (choice over the 9 Lifestyle Factor classes + Non_LSF with full distribution)
3. 4 guideline selection questions (choice over dynamic guideline IDs + "None")
4. `gating_uncertainty` (score: calibrated difficulty / need for LLM escalation)
"""

import os
import re
import copy
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Default Constants for 9Router System One JEV Gateway
DEFAULT_JEV_URL = os.getenv("JEV_ENDPOINT", os.getenv("JEV_URL", os.getenv("JEV_BASE_URL", "http://localhost:20128/v1/systemone")))
DEFAULT_JEV_MODEL = os.getenv("JEV_MODEL", "oc/jev-1.13-free")
DEFAULT_JEV_API_KEY = (
    os.getenv("JEV_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("NINEROUTER_OPENAI_API_KEY")
    or "sk-1f62f661c62bee6b-ktgzgu-eeb720fa"
)

# Paths to Prompt Templates and Questions
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "jev_gating"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "jev_state.txt"
FULL_DG_PROMPT_PATH = PROMPTS_DIR / "jev_state_full_dg.txt"
DEFAULT_QUESTIONS_PATH = PROMPTS_DIR / "jev_questions.json"
STAGE1_PROMPT_PATH = PROMPTS_DIR / "jev_stage1_state.txt"
STAGE1_QUESTIONS_PATH = PROMPTS_DIR / "jev_stage1_questions.json"
STAGE2_PROMPT_PATH = PROMPTS_DIR / "jev_stage2_state.txt"
STAGE2_QUESTIONS_PATH = PROMPTS_DIR / "jev_stage2_questions.json"


@dataclass
class JevDecisionResult:
    """Structured decision output from Jev System One."""
    span_id: str
    span_text: str
    is_lsf_entity: float  # Noul probability (0.0 to 1.0)
    predicted_category: str  # Top choice among 9 LSF classes or Non_LSF
    category_probabilities: Dict[str, float]  # Full probability distribution
    gating_uncertainty: float  # Calibrated difficulty/uncertainty score (0.0 to 2.0)
    should_escalate_to_llm: bool  # Recommended LLM escalation decision
    selected_guideline: str = "None"  # Most relevant guideline (choice)
    guideline_scores: Dict[str, float] = field(default_factory=dict)  # Relevance score/probability for each guideline
    selected_guidelines: Dict[str, str] = field(default_factory=dict)  # Legacy alias
    raw_response: Dict[str, Any] = field(default_factory=dict)


def load_jev_state_template(prompt_path: Optional[Union[str, Path]] = None, use_full_dg: bool = False) -> str:
    """Loads the state prompt template from the prompts directory."""
    if prompt_path:
        path = Path(prompt_path)
    elif use_full_dg and FULL_DG_PROMPT_PATH.exists():
        path = FULL_DG_PROMPT_PATH
    else:
        path = DEFAULT_PROMPT_PATH

    if not path.exists():
        if FULL_DG_PROMPT_PATH.exists():
            return FULL_DG_PROMPT_PATH.read_text(encoding="utf-8")
        # Fallback default template if file is missing
        return (
            "Target Candidate Mention: \"{span_text}\"\n"
            "Candidate Prior Label: {spanner_predicted_label}\n\n"
            "Local Sentence Context:\n"
            "\"{sentence_context}\"\n\n"
            "Full Abstract Context:\n"
            "\"\"\"{abstract_text}\"\"\"\n\n"
            "Domain Guidelines:\n"
            "{guidelines}\n\n"
            "Guideline Examples:\n"
            "{guideline_examples}\n"
        )
    return path.read_text(encoding="utf-8")


def load_jev_questions(questions_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads the structured questions and category criteria from JSON file."""
    path = Path(questions_path) if questions_path else DEFAULT_QUESTIONS_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}. Using fallback criteria.")

    # Inlined fallback if JSON is not reachable
    return {
        "is_lsf_entity": {
            "type": "noul",
            "instructions": "Is this candidate span an active Lifestyle Factor (LSF) entity or background text (O)?",
            "criteria": {
                "true": "Genuinely refers to a lifestyle, behavioral, environmental, nutritional, or socioeconomic factor.",
                "false": "General non-LSF medical term, biological concept, generic word, or background text (O)."
            }
        },
        "category": {
            "type": "choice",
            "instructions": "Which of the 9 Lifestyle Factor categories or Non_LSF (O) best describes this mention?",
            "criteria": {
                "Personal_care_products_and_cosmetic_procedures": "Cosmetics, hygiene products, skincare, hair dyes, sunscreen, plastic/cosmetic surgery, grooming items.",
                "Substance_use": "Tobacco, smoking, alcohol, recreational drugs, vaping, substance dependence, narcotics.",
                "Environmental_exposures": "Air pollution, heavy metals, pesticides, secondhand smoke, radiation, occupational hazards, noise exposure.",
                "Mental_health_practices": "Psychotherapy, meditation, mindfulness, cognitive behavioral therapy, stress management, counseling, relaxation techniques.",
                "Non_physical_leisure_time_activities": "Reading, watching TV, screen time, video games, listening to music, social clubs, board games, hobbies.",
                "Nutrition": "Diet, food items, caloric intake, nutrients, vitamins, fasting, eating habits, dietary supplements, beverages.",
                "Physical_activities": "Exercise, walking, running, sports, gym workouts, manual labor, physical fitness, sedentary behavior.",
                "Sleep": "Sleep duration, sleep quality, circadian rhythms, insomnia, naps, bedtime routines, sleep hygiene.",
                "Socioeconomic_factors": "Income, employment, poverty, education level, marital status, housing conditions, wealth, health insurance.",
                "Non_LSF": "Non-lifestyle mention, background text, disease/pathology, internal biomarker, medication/clinical procedure, study name, or demographic factor."
            }
        },
        "guideline_relevance": {
            "type": "choice",
            "instructions": "Evaluate and score the relevance of domain guidelines based on the abstract text, local sentence, candidate span, and classification evidence. Select the guideline from the dynamic guidebook that is most relevant and applicable to resolving this candidate mention (or 'None' if no specialized guideline applies).",
            "criteria": {"None": "None"}
        },
        "gating_uncertainty": {
            "type": "score",
            "instructions": "Rate the classification and boundary uncertainty for this candidate mention on a scale from 0.0 to 1.0 (where 0.0 represents low uncertainty / high confidence in the result, and 1.0 represents high uncertainty / highly ambiguous).",
            "criteria": [
                "0.0: Low uncertainty (high confidence of the result, unambiguous standard entity)",
                "0.5: Moderate uncertainty (moderate ambiguity, overlapping categories, or borderline lifestyle factor)",
                "1.0: High uncertainty (highly ambiguous, compound mention requiring decomposition, or complex reasoning needed)"
            ]
        }
    }


def extract_sentence_context(full_text: str, start_char: int, end_char: int, window: int = 120) -> str:
    """Extracts the immediate sentence or local window surrounding the candidate span."""
    if not full_text or start_char < 0:
        return full_text or ""
    
    # Expand window boundaries to find surrounding sentence
    left = max(0, start_char - window)
    right = min(len(full_text), end_char + window)
    
    # Try snapping to sentence boundaries (periods or newlines)
    period_left = full_text.rfind(". ", 0, start_char)
    if period_left != -1 and period_left >= left - 40:
        left = period_left + 2
        
    period_right = full_text.find(". ", end_char)
    if period_right != -1 and period_right <= right + 40:
        right = period_right + 1

    return full_text[left:right].strip()


def format_guidelines_from_json(
    guidebook_path: Optional[Union[str, Path]] = None,
    examples_path: Optional[Union[str, Path]] = None,
    max_examples_per_rule: Optional[int] = 1
) -> str:
    """
    Formats dynamic guidelines with selective fields (id, span, span_sentence, labels, triad type, guideline)
    and integrates companion positive and negative examples (Span, GT, BERT, Gen, Triad) directly under each guideline record.
    """
    if not guidebook_path:
        return "No domain guidelines available."
    g_path = Path(guidebook_path)
    if not g_path.exists():
        return "No domain guidelines available."

    # Load examples dictionary
    ex_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if examples_path:
        e_path = Path(examples_path)
        if e_path.exists():
            try:
                e_data = json.loads(e_path.read_text(encoding="utf-8"))
                ex_map = e_data.get("examples", e_data) if isinstance(e_data, dict) else {}
            except Exception as e:
                logger.warning(f"Error loading examples from {e_path}: {e}")

    try:
        data = json.loads(g_path.read_text(encoding="utf-8"))
        items = data.get("guidebook", data) if isinstance(data, dict) else data
        if not items:
            return "No domain guidelines available."

        lines = []
        for idx, item in enumerate(items, 1):
            gid = item.get("id", f"DG-{idx:04d}")
            span = item.get("span", "").strip()
            span_sentence = item.get("span_sentence", "").strip()
            gt_lbl = item.get("ground_truth_label", "N/A")
            bert_lbl = item.get("bert_label", "N/A")
            llm_lbl = item.get("llm_label", "N/A")
            
            triad_obj = item.get("triad", {})
            triad_type = triad_obj.get("type", "N/A") if isinstance(triad_obj, dict) else str(triad_obj)
            guideline_text = item.get("guideline", "").strip()

            rule_lines = [
                f"### Guideline {gid}",
                f"- **Trigger Span**: \"{span}\"",
                f"- **Sentence**: \"{span_sentence}\"",
                f"- **Labels**: GT: {gt_lbl} | BERT: {bert_lbl} | LLM: {llm_lbl}",
                f"- **Triad**: {triad_type}",
                f"- **Guideline**: {guideline_text}"
            ]

            # Companion Positive and Negative examples for this guideline
            g_ex = ex_map.get(gid, {}) if isinstance(ex_map, dict) else {}
            pos_list = g_ex.get("positive", []) if isinstance(g_ex, dict) else []
            neg_list = g_ex.get("negative", []) if isinstance(g_ex, dict) else []
            if max_examples_per_rule is not None:
                pos_list = pos_list[:max_examples_per_rule]
                neg_list = neg_list[:max_examples_per_rule]

            if pos_list:
                rule_lines.append("- **Positive Examples**:")
                for p in pos_list:
                    p_span = str(p.get("span", "")).strip()
                    p_gt = str(p.get("ground_truth_label", p.get("gt_label", "N/A"))).strip()
                    p_bert = str(p.get("bert_label", "N/A")).strip()
                    p_gen = str(p.get("generator_label", "N/A")).strip()
                    p_tr = str(p.get("outcome_triad", "N/A")).strip()
                    rule_lines.append(f"  * [POS] Span: \"{p_span}\" | GT: {p_gt} | BERT: {p_bert} | Gen: {p_gen} | Triad: {p_tr}")
            else:
                rule_lines.append("- **Positive Examples**: None")

            if neg_list:
                rule_lines.append("- **Negative Examples**:")
                for n in neg_list:
                    n_span = str(n.get("span", "")).strip()
                    n_gt = str(n.get("ground_truth_label", n.get("gt_label", n.get("bert_label", "N/A")))).strip()
                    n_bert = str(n.get("bert_label", "N/A")).strip()
                    n_gen = str(n.get("generator_label", "N/A")).strip()
                    n_tr = str(n.get("outcome_triad", "N/A")).strip()
                    rule_lines.append(f"  * [NEG] Span: \"{n_span}\" | GT: {n_gt} | BERT: {n_bert} | Gen: {n_gen} | Triad: {n_tr}")
            else:
                rule_lines.append("- **Negative Examples**: None")

            lines.append("\n".join(rule_lines))

        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"Error loading and formatting guidelines from {g_path}: {e}")
        return "No domain guidelines available."


def format_guideline_examples_from_json(
    examples_path: Optional[Union[str, Path]] = None,
    max_examples_per_rule: int = 5
) -> str:
    """Legacy alias / fallback."""
    return ""


def extract_guideline_ids_from_json(guidebook_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Extracts unique guideline IDs from dynamicGuidebook.json."""
    if not guidebook_path:
        return []
    path = Path(guidebook_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("guidebook", data) if isinstance(data, dict) else data
        ids = []
        for idx, item in enumerate(items, 1):
            gid = item.get("id", f"DG-{idx:04d}")
            if gid not in ids:
                ids.append(gid)
        return ids
    except Exception as e:
        logger.warning(f"Error extracting guideline IDs from {path}: {e}")
        return []


def extract_guideline_ids_from_text(guidelines_text: str) -> List[str]:
    """Extracts unique guideline IDs (e.g. DG-0001) from formatted guideline strings."""
    if not guidelines_text:
        return []
    matches = re.findall(r'\b(DG-\d+)\b', guidelines_text)
    return list(dict.fromkeys(matches))


def populate_guideline_selection_criteria(questions: Dict[str, Any], guideline_ids: List[str]) -> Dict[str, Any]:
    """
    Populates the guideline choice questions (such as 'guideline_relevance') with active guideline IDs mapped to themselves.
    Schema: { "None": "None", "DG-0001": "DG-0001", "DG-0002": "DG-0002", ... }
    """
    if not guideline_ids:
        return questions

    guideline_criteria = {"None": "None"}
    for gid in guideline_ids:
        guideline_criteria[gid] = gid

    target_questions = [
        "guideline_relevance",
        "guideline_selection_abstract",
        "guideline_selection_category",
        "guideline_selection_positive",
        "guideline_selection_negative"
    ]

    for q_key in target_questions:
        if q_key in questions and isinstance(questions[q_key], dict) and questions[q_key].get("type") == "choice":
            questions[q_key]["criteria"] = guideline_criteria

    return questions


def call_single_jev_decision(
    span_data: Dict[str, Any],
    abstract_text: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    prompt_template: Optional[str] = None,
    questions_config: Optional[Dict[str, Any]] = None,
    uncertainty_escalation_threshold: float = 1.0,
    timeout: float = 45.0,
    guidelines: Optional[str] = None,
    guideline_examples: Optional[str] = None,
    guideline_ids: Optional[List[str]] = None
) -> JevDecisionResult:
    """
    Executes a single Jev System One decision call via 9Router (http://localhost:20128/v1/systemone) for a candidate span.
    """
    api_key = (
        api_key
        or os.getenv("JEV_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("NINEROUTER_OPENAI_API_KEY")
        or DEFAULT_JEV_API_KEY
    )
    base_url = (
        base_url
        or os.getenv("JEV_ENDPOINT")
        or os.getenv("JEV_URL")
        or os.getenv("JEV_BASE_URL")
        or DEFAULT_JEV_URL
    )
    model = model or os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)

    span_id = str(span_data.get("id", span_data.get("span_id", "span_0")))
    span_text = span_data.get("span_text", span_data.get("entity", span_data.get("text", "")))
    spanner_label = span_data.get("bert_predicted_label", span_data.get("predicted_label", span_data.get("spanner_label", "O")))
    
    # Offsets
    boundaries = span_data.get("boundaries", {})
    start_char = boundaries.get("start_char", span_data.get("start_char", -1))
    end_char = boundaries.get("end_char", span_data.get("end_char", -1))

    sentence_context = extract_sentence_context(abstract_text, start_char, end_char)

    # Render prompt template using safe string replacement
    template_str = prompt_template or load_jev_state_template(use_full_dg=bool(guidelines or guideline_examples))
    state_prompt = template_str
    state_prompt = state_prompt.replace("{span_text}", span_text)
    state_prompt = state_prompt.replace("{spanner_predicted_label}", spanner_label)
    state_prompt = state_prompt.replace("{sentence_context}", sentence_context)
    state_prompt = state_prompt.replace("{abstract_text}", abstract_text)
    state_prompt = state_prompt.replace("{guidelines}", guidelines or "No domain guidelines available.")
    state_prompt = state_prompt.replace("{guideline_examples}", guideline_examples or "No guideline examples available.")
    state_prompt = state_prompt.replace("{dynamicGuidebook}", guidelines or "No domain guidelines available.")
    state_prompt = state_prompt.replace("{dynamicGuidelines}", guidelines or "No domain guidelines available.")

    # Questions structure with dynamic guideline ID injection
    questions = copy.deepcopy(questions_config or load_jev_questions())
    eff_gids = guideline_ids
    if not eff_gids and guidelines:
        eff_gids = extract_guideline_ids_from_text(guidelines)
    if eff_gids:
        questions = populate_guideline_selection_criteria(questions, eff_gids)

    payload = {
        "model": model,
        "state": state_prompt,
        "questions": questions
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 5
    backoff_delay = 3.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(base_url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait_time = backoff_delay * (2 ** (attempt - 1))
                logger.warning(f"JEV Rate limit (429) on span '{span_text}' ({span_id}). Retrying in {wait_time:.1f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue

            if resp.status_code != 200:
                logger.error(f"Jev API returned status {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            
            # Parse decision results (OpenRouter Decisions API returns 'answers')
            results = data.get("answers", data.get("results", {}))

            # 1. is_lsf_entity (noul probability)
            is_lsf_info = results.get("is_lsf_entity", {})
            is_lsf_prob = float(is_lsf_info.get("noul", is_lsf_info.get("probability", 0.5)))

            # 2. category choice & full distribution
            cat_info = results.get("category", {})
            predicted_category = str(cat_info.get("choice", "Non_LSF"))
            cat_probs = cat_info.get("probabilities", {})
            if not isinstance(cat_probs, dict):
                cat_probs = {predicted_category: 1.0}

            # 3. Guideline relevance selection & scoring distribution
            selected_guideline = "None"
            guideline_scores = {}
            if "guideline_relevance" in results:
                rel_info = results["guideline_relevance"]
                selected_guideline = str(rel_info.get("choice", "None"))
                guideline_scores = rel_info.get("probabilities", {})
                if not isinstance(guideline_scores, dict):
                    guideline_scores = {selected_guideline: 1.0} if selected_guideline != "None" else {}

            # Backwards-compatible dictionary of selected guidelines
            selected_guidelines = {"relevance": selected_guideline}
            for q_key, res_key in [
                ("guideline_selection_abstract", "abstract"),
                ("guideline_selection_category", "category"),
                ("guideline_selection_positive", "positive"),
                ("guideline_selection_negative", "negative")
            ]:
                if q_key in results:
                    selected_guidelines[res_key] = str(results[q_key].get("choice", "None"))
                    if selected_guideline == "None":
                        selected_guideline = selected_guidelines[res_key]

            # 4. gating uncertainty score
            unc_info = results.get("gating_uncertainty", {})
            gating_unc = float(unc_info.get("score", 0.0))

            # 5. Escalation logic based on uncertainty or low is_lsf probability
            should_escalate = (
                gating_unc >= uncertainty_escalation_threshold or
                (0.20 <= is_lsf_prob <= 0.80) or
                (predicted_category == "Non_LSF" and spanner_label != "O")
            )

            return JevDecisionResult(
                span_id=span_id,
                span_text=span_text,
                is_lsf_entity=is_lsf_prob,
                predicted_category=predicted_category,
                category_probabilities=cat_probs,
                gating_uncertainty=gating_unc,
                should_escalate_to_llm=should_escalate,
                selected_guideline=selected_guideline,
                guideline_scores=guideline_scores,
                selected_guidelines=selected_guidelines,
                raw_response=data
            )

        except Exception as e:
            if attempt < max_retries and ("429" in str(e) or "rate_limit" in str(e).lower()):
                wait_time = backoff_delay * (2 ** (attempt - 1))
                logger.warning(f"Jev rate limit error on '{span_text}' ({span_id}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            if attempt == max_retries:
                logger.warning(f"Jev call failed for span '{span_text}' ({span_id}) after {max_retries} attempts: {e}. Falling back to prior.")
                break

    return JevDecisionResult(
        span_id=span_id,
        span_text=span_text,
        is_lsf_entity=0.5,
        predicted_category=spanner_label,
        category_probabilities={spanner_label: 1.0},
        gating_uncertainty=2.0,
        should_escalate_to_llm=True,
        selected_guideline="None",
        guideline_scores={},
        selected_guidelines={"relevance": "None", "abstract": "None", "category": "None", "positive": "None", "negative": "None"},
        raw_response={"error": "Max retries exceeded or call failed"}
    )


def call_batch_jev_decisions(
    spans_list: List[Dict[str, Any]],
    abstract_text: str,
    max_workers: int = 1,
    request_delay: float = 1.5,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    prompt_template: Optional[str] = None,
    questions_config: Optional[Dict[str, Any]] = None,
    timeout: float = 45.0,
    guidelines: Optional[str] = None,
    guidebook_path: Optional[Union[str, Path]] = None,
    guideline_examples: Optional[str] = None,
    examples_path: Optional[Union[str, Path]] = None,
    guideline_ids: Optional[List[str]] = None
) -> Dict[str, JevDecisionResult]:
    """
    Executes Jev System One decision calls across candidate spans for an abstract.
    When max_workers <= 1, executes sequentially with request_delay to strictly honor API rate limits.
    """
    eff_guidelines = guidelines or format_guidelines_from_json(guidebook_path=guidebook_path, examples_path=examples_path)
    eff_examples = guideline_examples or ""
    eff_gids = guideline_ids or extract_guideline_ids_from_json(guidebook_path) or extract_guideline_ids_from_text(eff_guidelines)
    
    use_full = bool(eff_guidelines or eff_examples)
    template_str = prompt_template or load_jev_state_template(use_full_dg=use_full)
    questions = questions_config or load_jev_questions()

    results_map: Dict[str, JevDecisionResult] = {}

    if max_workers <= 1:
        for idx, span in enumerate(spans_list):
            sid = str(span.get("id", span.get("span_id", idx)))
            dec = call_single_jev_decision(
                span_data=span,
                abstract_text=abstract_text,
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt_template=template_str,
                questions_config=questions,
                timeout=timeout,
                guidelines=eff_guidelines,
                guideline_examples=eff_examples,
                guideline_ids=eff_gids
            )
            results_map[sid] = dec
            if request_delay > 0 and idx < len(spans_list) - 1:
                time.sleep(request_delay)
        return results_map

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {
            executor.submit(
                call_single_jev_decision,
                span_data=span,
                abstract_text=abstract_text,
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt_template=template_str,
                questions_config=questions,
                timeout=timeout,
                guidelines=eff_guidelines,
                guideline_examples=eff_examples,
                guideline_ids=eff_gids
            ): str(span.get("id", span.get("span_id", idx)))
            for idx, span in enumerate(spans_list)
        }

        for future in as_completed(future_to_span):
            span_id = future_to_span[future]
            try:
                dec = future.result()
                results_map[span_id] = dec
            except Exception as e:
                logger.error(f"Error resolving Jev decision for span {span_id}: {e}")

    return results_map


def load_or_fetch_doc_jev_decisions(
    doc_id: str,
    abstract_text: str,
    spans: List[Dict[str, Any]],
    cache_dir: Optional[Union[str, Path]] = None,
    max_workers: int = 10,
    prompt_template_path: Optional[Union[str, Path]] = None,
    questions_path: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    guidelines: Optional[str] = None,
    guidebook_path: Optional[Union[str, Path]] = None,
    guideline_examples: Optional[str] = None,
    examples_path: Optional[Union[str, Path]] = None,
    guideline_ids: Optional[List[str]] = None
) -> Dict[str, JevDecisionResult]:
    """
    Loads precomputed JEV decisions from cache if present; otherwise calls Jev API and saves cache.
    """
    cache_path = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        # Check standard {doc_id}.json and fallback {doc_id}_jev.json
        candidate_paths = [
            cache_dir_path / f"{doc_id}.json",
            cache_dir_path / f"{doc_id}_jev.json"
        ]
        for cp in candidate_paths:
            if cp.exists():
                cache_path = cp
                break
        if not cache_path:
            cache_path = cache_dir_path / f"{doc_id}.json"

        if cache_path.exists():
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                decisions = {}
                for sid, d in cached_data.items():
                    decisions[str(sid)] = JevDecisionResult(
                        span_id=str(d.get("span_id", sid)),
                        span_text=d.get("span_text", ""),
                        is_lsf_entity=d.get("is_lsf_entity", 0.5),
                        predicted_category=d.get("predicted_category", "Non_LSF"),
                        category_probabilities=d.get("category_probabilities", {}),
                        gating_uncertainty=d.get("gating_uncertainty", 0.0),
                        should_escalate_to_llm=d.get("should_escalate_to_llm", False),
                        selected_guideline=d.get("selected_guideline", "None"),
                        guideline_scores=d.get("guideline_scores", {}),
                        selected_guidelines=d.get("selected_guidelines", {}),
                        raw_response=d.get("raw_response", {})
                    )
                return decisions
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}. Calling API...")

    # Load prompt template and questions dynamically if specified
    eff_guidelines = guidelines or format_guidelines_from_json(guidebook_path)
    eff_examples = guideline_examples or format_guideline_examples_from_json(examples_path)
    eff_gids = guideline_ids or extract_guideline_ids_from_json(guidebook_path) or extract_guideline_ids_from_text(eff_guidelines)
    
    use_full = bool(eff_guidelines or eff_examples or prompt_template_path)
    prompt_tpl = load_jev_state_template(prompt_template_path, use_full_dg=use_full)
    questions_cfg = load_jev_questions(questions_path)

    # Fetch fresh from Jev API
    decisions = call_batch_jev_decisions(
        spans_list=spans,
        abstract_text=abstract_text,
        max_workers=max_workers,
        prompt_template=prompt_tpl,
        questions_config=questions_cfg,
        api_key=api_key,
        guidelines=eff_guidelines,
        guideline_examples=eff_examples,
        guideline_ids=eff_gids
    )

    # Save cache
    if cache_path:
        serializable = {
            sid: {
                "span_id": dec.span_id,
                "span_text": dec.span_text,
                "is_lsf_entity": dec.is_lsf_entity,
                "predicted_category": dec.predicted_category,
                "category_probabilities": dec.category_probabilities,
                "gating_uncertainty": dec.gating_uncertainty,
                "should_escalate_to_llm": dec.should_escalate_to_llm,
                "selected_guideline": dec.selected_guideline,
                "guideline_scores": dec.guideline_scores,
                "selected_guidelines": dec.selected_guidelines
            }
            for sid, dec in decisions.items()
        }
        try:
            cache_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_path}: {e}")

    return decisions


def build_anchored_abstract_for_spans(abstract_text: str, spans: List[Dict[str, Any]]) -> str:
    """Builds anchored abstract with [id: {id}, span: {span_text}] tags."""
    if not spans or not abstract_text:
        return abstract_text or ""
    
    events = []
    for s in spans:
        sid = s.get("id", s.get("span_id", 0))
        s_text = s.get("span_text", s.get("entity", s.get("text", "")))
        boundaries = s.get("boundaries", {})
        sc = boundaries.get("start_char", s.get("start_char", 0))
        ec = boundaries.get("end_char", s.get("end_char", len(s_text)))

        if sc < 0 or ec > len(abstract_text) or abstract_text[sc:ec] != s_text:
            found = abstract_text.find(s_text)
            if found != -1:
                sc = found
                ec = found + len(s_text)
            else:
                continue

        span_len = ec - sc
        events.append((ec, 0, span_len, sid, "]"))
        events.append((sc, 1, -span_len, sid, f"[id: {sid}, span: "))

    events.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    res = abstract_text
    for pos, _, _, _, tag in events:
        res = res[:pos] + tag + res[pos:]
    return res


def call_two_stage_doc_jev(
    doc_id: str,
    abstract_text: str,
    spans: List[Dict[str, Any]],
    guidelines: str,
    guideline_ids: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    uncertainty_threshold: float = 0.6,
    timeout: float = 60.0
) -> Dict[str, JevDecisionResult]:
    """
    Executes Two-Stage JEV Document-Level Inference:
      Stage 1: In 1 API call, classifies all candidate spans for the document.
      Stage 2: In 1 API call, evaluates guideline relevance for all escalated spans.
    """
    api_key = api_key or DEFAULT_JEV_API_KEY
    base_url = base_url or DEFAULT_JEV_URL
    model = model or DEFAULT_JEV_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # --- STAGE 1: Document-Level Span Classification ---
    anchored_abstract = build_anchored_abstract_for_spans(abstract_text, spans)
    spans_list_str = "\n".join([
        f"{s.get('id', idx+1)}. \"{s.get('span_text', '')}\" (BERT Prior: {s.get('bert_predicted_label', s.get('predicted_label', 'O'))})"
        for idx, s in enumerate(spans)
    ])

    stage1_template_str = STAGE1_PROMPT_PATH.read_text(encoding="utf-8") if STAGE1_PROMPT_PATH.exists() else (
        "Biomedical Abstract (with anchored candidate spans proposed by BERT):\n\"\"\"\n{anchored_abstract}\n\"\"\"\n\n"
        "Proposed Candidate Spans to Classify:\n{candidate_spans_list}\n\n"
        "Domain Guidelines & Curated Examples:\n{guidelines}\n"
    )

    stage1_state = stage1_template_str.replace("{anchored_abstract}", anchored_abstract)\
                                      .replace("{candidate_spans_list}", spans_list_str)\
                                      .replace("{guidelines}", guidelines)

    q1_template_data = json.loads(STAGE1_QUESTIONS_PATH.read_text(encoding="utf-8")) if STAGE1_QUESTIONS_PATH.exists() else {}
    q1_base = q1_template_data.get("category_template", load_jev_questions().get("category", {}))

    stage1_questions = {}
    for idx, s in enumerate(spans, 1):
        sid = s.get("id", s.get("span_id", idx))
        span_text = s.get("span_text", "")
        bert_prior = s.get("bert_predicted_label", s.get("predicted_label", "O"))
        q = copy.deepcopy(q1_base)
        q["instructions"] = q.get("instructions", "").replace("{id}", str(sid))\
                                                    .replace("{span_text}", span_text)\
                                                    .replace("{bert_prior}", bert_prior)
        stage1_questions[f"span_{sid}"] = q

    payload1 = {
        "model": model,
        "state": stage1_state,
        "questions": stage1_questions
    }

    results_map: Dict[str, JevDecisionResult] = {}
    escalated_spans = []

    max_retries = 5
    backoff_delay = 3.0

    # --- STAGE 1 Execution with Retries ---
    for attempt in range(1, max_retries + 1):
        try:
            resp1 = requests.post(base_url, headers=headers, json=payload1, timeout=timeout)
            if resp1.status_code in (429, 502, 503, 504):
                wait_time = backoff_delay * (2 ** (attempt - 1))
                logger.warning(f"Stage 1 JEV API returned status {resp1.status_code} for Doc {doc_id}. Retrying in {wait_time:.1f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue

            resp1.raise_for_status()
            data1 = resp1.json()
            answers1 = data1.get("answers", data1.get("results", {}))

            for idx, s in enumerate(spans, 1):
                sid = str(s.get("id", s.get("span_id", idx)))
                span_text = s.get("span_text", "")
                bert_label = str(s.get("bert_predicted_label", s.get("predicted_label", "O")))

                ans = answers1.get(f"span_{sid}", {})
                pred_cat = str(ans.get("choice", "Non_LSF"))
                cat_probs = ans.get("probabilities", {})
                if not isinstance(cat_probs, dict):
                    cat_probs = {pred_cat: 1.0}

                conf = float(ans.get("confidence", max(cat_probs.values()) if cat_probs else 1.0))
                gating_unc = round(1.0 - conf, 4)
                is_lsf_prob = round(1.0 - float(cat_probs.get("Non_LSF", 0.0)), 4)

                should_escalate = (bert_label != pred_cat) or (gating_unc > uncertainty_threshold)

                dec = JevDecisionResult(
                    span_id=sid,
                    span_text=span_text,
                    is_lsf_entity=is_lsf_prob,
                    predicted_category=pred_cat,
                    category_probabilities=cat_probs,
                    gating_uncertainty=gating_unc,
                    should_escalate_to_llm=should_escalate,
                    selected_guideline="None",
                    guideline_scores={},
                    selected_guidelines={"relevance": "None"},
                    raw_response=ans
                )
                results_map[sid] = dec
                if should_escalate:
                    escalated_spans.append(s)

            break  # Successfully finished Stage 1

        except Exception as e:
            if attempt < max_retries:
                wait_time = backoff_delay * (2 ** (attempt - 1))
                logger.warning(f"Stage 1 JEV call failed for Doc {doc_id} on attempt {attempt}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Stage 1 JEV call permanently failed for Doc {doc_id} after {max_retries} attempts: {e}. Falling back to priors.")
                for idx, s in enumerate(spans, 1):
                    sid = str(s.get("id", s.get("span_id", idx)))
                    span_text = s.get("span_text", "")
                    bert_label = str(s.get("bert_predicted_label", s.get("predicted_label", "O")))
                    results_map[sid] = JevDecisionResult(
                        span_id=sid,
                        span_text=span_text,
                        is_lsf_entity=0.5,
                        predicted_category=bert_label,
                        category_probabilities={bert_label: 1.0},
                        gating_uncertainty=2.0,
                        should_escalate_to_llm=True,
                        selected_guideline="None",
                        guideline_scores={},
                        selected_guidelines={"relevance": "None"},
                        raw_response={"error": str(e)}
                    )
                    escalated_spans.append(s)

    # --- STAGE 2: Guideline Relevance Scoring for Escalated Spans ---
    if escalated_spans and guideline_ids:
        esc_spans_list_str = "\n".join([
            f"{s.get('id', idx+1)}. \"{s.get('span_text', '')}\" (BERT: {s.get('bert_predicted_label', s.get('predicted_label', 'O'))}, JEV: {results_map[str(s.get('id', s.get('span_id', idx+1)))].predicted_category}, Unc: {results_map[str(s.get('id', s.get('span_id', idx+1)))].gating_uncertainty})"
            for idx, s in enumerate(escalated_spans)
        ])

        stage2_template_str = STAGE2_PROMPT_PATH.read_text(encoding="utf-8") if STAGE2_PROMPT_PATH.exists() else (
            "Biomedical Abstract (with escalated candidate spans requiring guideline matching):\n\"\"\"\n{anchored_abstract}\n\"\"\"\n\n"
            "Escalated Candidate Spans (requiring domain guideline relevance scoring):\n{escalated_spans_list}\n\n"
            "Domain Guidelines & Curated Examples:\n{guidelines}\n"
        )

        stage2_state = stage2_template_str.replace("{anchored_abstract}", anchored_abstract)\
                                          .replace("{escalated_spans_list}", esc_spans_list_str)\
                                          .replace("{guidelines}", guidelines)

        g_criteria = {"None": "None"}
        for gid in guideline_ids:
            g_criteria[gid] = gid

        q2_template_data = json.loads(STAGE2_QUESTIONS_PATH.read_text(encoding="utf-8")) if STAGE2_QUESTIONS_PATH.exists() else {}
        q2_base = q2_template_data.get("guideline_relevance_template", {})

        stage2_questions = {}
        for idx, s in enumerate(escalated_spans, 1):
            sid = str(s.get("id", s.get("span_id", idx)))
            span_text = s.get("span_text", "")
            q2 = copy.deepcopy(q2_base)
            q2["instructions"] = q2.get("instructions", "Select the most relevant guideline ID for candidate span #{id}: \"{span_text}\" (or 'None').").replace("{id}", sid).replace("{span_text}", span_text)
            q2["criteria"] = g_criteria
            stage2_questions[f"guideline_relevance_span_{sid}"] = q2

        payload2 = {
            "model": model,
            "state": stage2_state,
            "questions": stage2_questions
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp2 = requests.post(base_url, headers=headers, json=payload2, timeout=timeout)
                if resp2.status_code in (429, 502, 503, 504):
                    wait_time = backoff_delay * (2 ** (attempt - 1))
                    logger.warning(f"Stage 2 JEV API returned status {resp2.status_code} for Doc {doc_id}. Retrying in {wait_time:.1f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    continue

                resp2.raise_for_status()
                data2 = resp2.json()
                answers2 = data2.get("answers", data2.get("results", {}))

                for s in escalated_spans:
                    sid = str(s.get("id", s.get("span_id", 0)))
                    if sid in results_map:
                        ans2 = answers2.get(f"guideline_relevance_span_{sid}", {})
                        top_g = str(ans2.get("choice", "None"))
                        g_probs = ans2.get("probabilities", {})
                        if not isinstance(g_probs, dict):
                            g_probs = {top_g: 1.0} if top_g != "None" else {}
                        results_map[sid].selected_guideline = top_g
                        results_map[sid].guideline_scores = g_probs
                        results_map[sid].selected_guidelines = {"relevance": top_g}

                break  # Successfully finished Stage 2

            except Exception as e:
                if attempt < max_retries:
                    wait_time = backoff_delay * (2 ** (attempt - 1))
                    logger.warning(f"Stage 2 JEV call failed for Doc {doc_id} on attempt {attempt}: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Stage 2 JEV call permanently failed for Doc {doc_id}: {e}.")

    return results_map
