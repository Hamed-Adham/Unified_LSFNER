"""
TypeSafe System One (Jev) Drop-In ACE Generator.
Implements the exact Generator interface expected by BertizedACEPipeline:
  run(abstract, candidate_spans, dynamicGuidelines="", save_output_path=None, anchored_abstract=None)
without modifying any underlying pipeline scripts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.JeBert.client import TypeSafeClient

LSF_CHOICE_CRITERIA: Dict[str, str] = {
    "Personal_care_products_and_cosmetic_procedures": (
        "Products, practices, and personal care items for body grooming, oral hygiene, "
        "dental care, cosmetics, orthodontic appliances, skin care, and aesthetic treatments."
    ),
    "Substance_use": (
        "Substances, medications, addictive chemical agents, or smoking products consumed for recreational, "
        "therapeutic, or psychoactive effects (tobacco, alcohol, cannabis, pharmaceuticals, nicotine)."
    ),
    "Physical_activities": (
        "Muscular bodily movements resulting in energy expenditure: exercise routines, sports, active transport, "
        "walking exertion, physical activity levels, or sedentary sitting time."
    ),
    "Nutrition": (
        "Foods, beverages, macronutrients, micronutrients, dietary patterns, caloric intake, vitamins, minerals, "
        "and dietary supplements."
    ),
    "Socioeconomic_factors": (
        "Social, economic, educational, occupational, demographic, and relationship factors: income, education, "
        "employment, job title, social support, or interpersonal exposure."
    ),
    "Environmental_exposures": (
        "External physical, chemical, or radiological agents present in ambient, indoor, or workplace environments: "
        "airborne pollutants, toxins, radiation, noise, water contaminants."
    ),
    "Sleep": (
        "Physiological sleep patterns, sleep duration, quality, disturbances, bedtime routines, and biological sleep-wake rhythms."
    ),
    "Mental_health_practices": (
        "Psychological interventions, counseling, mindfulness, meditation, stress management techniques, "
        "and therapies aimed at mental wellness."
    ),
    "Non_physical_leisure_time_activities": (
        "Sedentary recreational pastimes: screen time, TV viewing, reading, video gaming, gambling, betting, "
        "and non-exertional hobbies."
    ),
    "O": (
        "Not a lifestyle factor entity; background text, grammatical token, non-lifestyle medical condition, "
        "anatomical word, or broad non-specific umbrella phrases (e.g., 'lifestyle change', 'lifestyle modification', 'lifestyle behaviors')."
    ),
}

LABEL_NORMALIZATION: Dict[str, str] = {
    "Beauty_and_Cleaning": "Personal_care_products_and_cosmetic_procedures",
    "Drugs": "Substance_use",
    "Physical_activity": "Physical_activities",
    "Personal_care_products_and_cosmetic_procedures": "Personal_care_products_and_cosmetic_procedures",
    "Substance_use": "Substance_use",
    "Physical_activities": "Physical_activities",
    "Nutrition": "Nutrition",
    "Socioeconomic_factors": "Socioeconomic_factors",
    "Environmental_exposures": "Environmental_exposures",
    "Sleep": "Sleep",
    "Mental_health_practices": "Mental_health_practices",
    "Non_physical_leisure_time_activities": "Non_physical_leisure_time_activities",
    "Lifestyle_factor": "Lifestyle_factor",
    "O": "O",
}


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return "O"
    return LABEL_NORMALIZATION.get(label, label)


def find_latest_guidebook(runs_dir: Path) -> Path:
    """Find the dynamicGuidebook.json from the most recent run folder."""
    run_folders = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_folders:
        raise FileNotFoundError(f"No run folders found in {runs_dir}")
    latest_run = sorted(run_folders, key=lambda d: d.name)[-1]
    gb_file = latest_run / "dynamicGuidebook.json"
    if not gb_file.exists():
        raise FileNotFoundError(f"dynamicGuidebook.json not found in {latest_run}")
    return gb_file


def load_guidebook_rules(guidebook_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Loads guidelines and separates into:
      1. Helpful guidelines (helpful > 0)
      2. Full guidelines (all)
    """
    with open(guidebook_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_list = data.get("guidebook", [])
    full_rules: List[Dict[str, str]] = []
    helpful_rules: List[Dict[str, str]] = []

    for item in raw_list:
        gid = item.get("id", "")
        rule_text = item.get("guideline", "").strip()
        metrics = item.get("usage_metrics", {})
        helpful_count = metrics.get("helpful", 0)

        entry = {
            "id": gid,
            "rule": rule_text,
            "target_category": item.get("ground_truth_label", ""),
        }
        full_rules.append(entry)
        if helpful_count > 0:
            helpful_rules.append(entry)

    return helpful_rules, full_rules


def extract_sentence_for_span(
    abstract: str,
    span_text: str,
    start_char: Optional[int] = None,
    end_char: Optional[int] = None,
) -> str:
    """
    Extracts the immediate containing sentence for a candidate span from an abstract.
    Uses regex sentence boundaries (.!? followed by whitespace/newline).
    """
    if not abstract or not span_text:
        return ""
    if start_char is None or end_char is None or start_char < 0:
        idx = abstract.find(span_text)
        if idx != -1:
            start_char = idx
            end_char = idx + len(span_text)
        else:
            return span_text

    pre_text = abstract[:start_char]
    post_text = abstract[end_char:]

    # Search backwards for the nearest sentence start
    left_bound = 0
    for match in re.finditer(r'(?:[\.\?!]\s+|\n+)', pre_text):
        left_bound = match.end()

    # Search forwards for the nearest sentence end
    match = re.search(r'[\.\?!](?:\s+|$|\n)|\n+', post_text)
    if match:
        right_bound = end_char + match.end()
    else:
        right_bound = len(abstract)

    sentence = abstract[left_bound:right_bound].strip()
    return sentence if sentence else span_text


def resolve_multi_rule_assignments(
    ans: Any,
    max_rules: int = 4,
    prob_floor: float = 0.10,
    relative_ratio: float = 0.35,
) -> List[str]:
    """
    Dynamically assigns up to `max_rules` based on relative probability ratio against top choice.
    Condition: prob >= max(prob_floor, top_prob * relative_ratio)
    """
    if not ans:
        return []

    # If Jev is highly confident that none of the candidate guidelines are relevant
    if ans.choice in ("NONE_APPLY", "NONE_RELEVANT") and ans.confidence >= 0.70:
        return []

    assigned: List[str] = []
    top_choice = ans.choice
    top_prob = ans.probabilities.get(top_choice, ans.confidence)

    if top_choice not in ("NONE_APPLY", "NONE_RELEVANT") and ans.confidence >= 0.12:
        assigned.append(top_choice)

    # Dynamic threshold relative to the top choice probability
    threshold = max(prob_floor, top_prob * relative_ratio)

    # Sort options by probability descending
    for opt, prob in sorted(ans.probabilities.items(), key=lambda x: x[1], reverse=True):
        if opt not in ("NONE_APPLY", "NONE_RELEVANT") and opt not in assigned and prob >= threshold:
            assigned.append(opt)
            if len(assigned) >= max_rules:
                break

    return assigned


class JevGenerator:
    """
    Drop-in replacement for BertizedGenerator using TypeSafe System One (Jev).
    Can be assigned to `ace_pipeline.generator` without changing underlying pipeline scripts.
    """

    def __init__(
        self,
        mode: str = "mode2",
        guidebook_path: Optional[str] = None,
        model_name: str = "jev-latest",
        batch_size: int = 15,
        max_assigned_rules: int = 4,
        client: Optional[TypeSafeClient] = None,
    ):
        """
        Args:
            mode: "mode1" (no guidelines), "mode2" (helpful guidelines), or "mode3" (full guidelines).
            guidebook_path: Path to dynamicGuidebook.json. If None, auto-resolves latest run.
            model_name: Model ID (default "jev-latest").
            batch_size: Number of candidate spans to evaluate per System One call.
            max_assigned_rules: Max number of guidelines assigned per candidate span in two-stage mode (default 4).
            client: Optional pre-configured TypeSafeClient.
        """
        self.mode = mode
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_assigned_rules = max_assigned_rules
        self.client = client or TypeSafeClient(model=model_name)

        runs_dir = PROJECT_ROOT / "output" / "bertized_ace" / "runs"
        gb_file = Path(guidebook_path) if guidebook_path else find_latest_guidebook(runs_dir)
        self.guidebook_path = gb_file

        helpful_rules, full_rules = load_guidebook_rules(gb_file)
        self.helpful_rules = helpful_rules
        self.full_rules = full_rules

        if self.mode == "mode1":
            self.active_guidelines = []
        elif self.mode == "mode2":
            self.active_guidelines = self.helpful_rules
        elif self.mode in ("two_stage", "mode4", "mode4_two_stage"):
            self.active_guidelines = self.helpful_rules
            # Safe default batch size for two-stage token constraints
            if self.batch_size == 15:
                self.batch_size = 4
        else:
            self.active_guidelines = self.full_rules

        # Calibrated guideline criteria for Pass 1 (Rule Assignment)
        # Use helpful rules to prevent unvalidated or harmful rules from polluting Pass 1
        target_rules = self.helpful_rules if self.mode in ("two_stage", "mode4", "mode4_two_stage") else self.full_rules
        self.two_stage_rules = target_rules
        self.rules_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.two_stage_rules:
            cat = r.get("target_category", "O")
            self.rules_by_category[cat].append(r)

    def get_scoped_rules_for_span(
        self,
        p_lbl: str,
        s_lbl: str,
        max_scoped: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate guideline rules scoped to the span's BERT top-1 and top-2 categories,
        plus relevant boundary/exclusion rules from category 'O'.
        Keeps the choice set compact (5-8 rules) to eliminate softmax dilution.
        """
        scoped: List[Dict[str, Any]] = []
        if p_lbl in self.rules_by_category and p_lbl != "O":
            scoped.extend(self.rules_by_category[p_lbl])
        if s_lbl in self.rules_by_category and s_lbl != "O" and s_lbl != p_lbl:
            scoped.extend(self.rules_by_category[s_lbl])

        p_key = p_lbl.lower().replace("_", " ")
        s_key = s_lbl.lower().replace("_", " ")

        o_rules = self.rules_by_category.get("O", [])
        matched_o: List[Dict[str, Any]] = []
        general_o: List[Dict[str, Any]] = []
        for r in o_rules:
            text = r.get("rule", "").lower()
            if (p_key != "o" and p_key in text) or (s_key != "o" and s_key in text):
                matched_o.append(r)
            elif r.get("id") in ("DG-0026", "DG-0038", "DG-0029", "DG-0020", "DG-0013", "DG-0016", "DG-0024"):
                general_o.append(r)

        scoped.extend(matched_o)
        for r in general_o:
            if r not in scoped:
                scoped.append(r)

        seen = set()
        final: List[Dict[str, Any]] = []
        for r in scoped:
            if r["id"] not in seen:
                seen.add(r["id"])
                final.append(r)
            if len(final) >= max_scoped:
                break

        if len(final) < 3:
            for r in o_rules:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    final.append(r)
                if len(final) >= max_scoped:
                    break

        return final

    def run(
        self,
        abstract: str,
        candidate_spans: Optional[List[Dict[str, Any]]] = None,
        dynamicGuidelines: str = "",
        save_output_path: Optional[str] = None,
        anchored_abstract: Optional[str] = None,
        hints: Optional[Any] = None,
        uncertain_entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes Jev System One arbitration on candidate spans.
        Compatible with both:
          - BertizedACEPipeline: expects llm_answer / final_answer with bullet_ids
          - DinasorPipeline / DinGenPipe: expects labeled_entities with hints_applied
        """
        target_spans = candidate_spans if candidate_spans is not None else (uncertain_entities or [])
        if not target_spans:
            return {
                "reasoning": "No candidate spans provided.",
                "bullet_ids": [],
                "hints_applied": [],
                "llm_answer": [],
                "final_answer": [],
                "labeled_entities": [],
                "raw_response": "{}",
                "prompt": "",
            }

        # Chunk candidate spans into batches
        batches = [
            target_spans[i : i + self.batch_size]
            for i in range(0, len(target_spans), self.batch_size)
        ]

        llm_answers: List[Dict[str, Any]] = []

        if self.mode in ("two_stage", "mode4", "mode4_two_stage"):
            for b_idx, batch in enumerate(batches):
                batch_answers = self._run_two_stage_batch(batch, abstract, hints=hints)
                llm_answers.extend(batch_answers)
        else:
            for b_idx, batch in enumerate(batches):
                state_spans = {}
                questions = {}

                for idx, c in enumerate(batch):
                    sid = f"s{idx}"
                    s_text = c.get("span_text", c.get("entity", "")).strip()
                    p_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                    s_lbl = normalize_label(c.get("second_best_label", "O"))
                    unc = round(float(c.get("uncertainty", c.get("u_score", 0.0))), 4)
                    nov = round(float(c.get("novelty_score", c.get("novelty", 0.0))), 4)
                    margin = round(float(c.get("margin", 0.0)), 4)

                    state_spans[sid] = {
                        "span_text": s_text,
                        "bert_predicted_label": p_lbl,
                        "second_best_label": s_lbl,
                        "uncertainty": unc,
                        "novelty_score": nov,
                        "margin": margin,
                    }

                    if self.active_guidelines:
                        instr = (
                            f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                            f"in `abstract`, taking into account BERT evidence in `candidate_spans.{sid}` and strictly "
                            f"adhering to domain rules in `dynamic_guidelines`."
                        )
                    else:
                        instr = (
                            f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                            f"in `abstract`, evaluating BERT evidence in `candidate_spans.{sid}` against canonical category criteria."
                        )

                    questions[sid] = {
                        "type": "choice",
                        "instructions": instr,
                        "criteria": LSF_CHOICE_CRITERIA,
                    }

                state = {
                    "abstract": abstract,
                    "dynamic_guidelines": self.active_guidelines,
                    "candidate_spans": state_spans,
                }
                if hints:
                    state["dinasor_hints"] = hints

                try:
                    resp = self.client.system_one(state=state, questions=questions)
                    for idx, c in enumerate(batch):
                        sid = f"s{idx}"
                        cid = c.get("id", len(llm_answers) + 1)
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        choice_ans = resp.choices.get(sid)

                        if choice_ans:
                            chosen_label = normalize_label(choice_ans.choice)
                            conf = choice_ans.confidence
                            probs = choice_ans.probabilities
                        else:
                            chosen_label = "O"
                            conf = 0.0
                            probs = {}

                        llm_answers.append({
                            "id": cid,
                            "entity": s_text,
                            "llm_predicted_label": chosen_label,
                            "final_label": chosen_label,
                            "label": chosen_label,
                            "confidence": conf,
                            "probabilities": probs,
                            "source": f"TypeSafe-Jev ({self.mode})",
                            "rationale": f"Jev System One choice (confidence={conf:.2f})",
                        })

                except Exception as exc:
                    for idx, c in enumerate(batch):
                        cid = c.get("id", len(llm_answers) + 1)
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        fallback_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                        llm_answers.append({
                            "id": cid,
                            "entity": s_text,
                            "llm_predicted_label": fallback_lbl,
                            "final_label": fallback_lbl,
                            "label": fallback_lbl,
                            "confidence": 0.0,
                            "probabilities": {},
                            "source": "TypeSafe-Jev-Fallback",
                            "rationale": f"Fallback to BERT due to API error: {exc}",
                        })

        labeled_entities = [
            {
                "entity": a["entity"],
                "label": a["final_label"],
                "confidence": a.get("confidence", 0.0),
                "probabilities": a.get("probabilities", {}),
                "rationale": a.get("rationale", ""),
            }
            for a in llm_answers
        ]
        assigned_bids = set()
        for a in llm_answers:
            for rid in a.get("assigned_guidelines", []):
                assigned_bids.add(rid)
        bids = sorted(list(assigned_bids)) if assigned_bids else [g.get("id") for g in self.active_guidelines[:5] if "id" in g]

        result = {
            "reasoning": f"TypeSafe Jev System One arbitration under {self.mode}",
            "bullet_ids": bids,
            "hints_applied": bids,
            "llm_answer": llm_answers,
            "final_answer": llm_answers,
            "labeled_entities": labeled_entities,
            "raw_response": json.dumps(llm_answers),
            "prompt": "TypeSafe System One State + Questions Schema",
        }

        # Save to save_output_path if requested (matches Generator logging format)
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result

    async def run_async(
        self,
        abstract: str,
        candidate_spans: Optional[List[Dict[str, Any]]] = None,
        dynamicGuidelines: str = "",
        save_output_path: Optional[str] = None,
        anchored_abstract: Optional[str] = None,
        http_client: Optional[Any] = None,
        hints: Optional[Any] = None,
        uncertain_entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Asynchronously executes Jev System One arbitration on candidate spans.
        Compatible with both BertizedACEPipeline and DinasorPipeline.
        """
        target_spans = candidate_spans if candidate_spans is not None else (uncertain_entities or [])
        if not target_spans:
            return {
                "reasoning": "No candidate spans provided.",
                "bullet_ids": [],
                "hints_applied": [],
                "llm_answer": [],
                "final_answer": [],
                "labeled_entities": [],
                "raw_response": "{}",
                "prompt": "",
            }

        batches = [
            target_spans[i : i + self.batch_size]
            for i in range(0, len(target_spans), self.batch_size)
        ]

        llm_answers: List[Dict[str, Any]] = []

        if self.mode in ("two_stage", "mode4", "mode4_two_stage"):
            for b_idx, batch in enumerate(batches):
                batch_answers = await self._run_two_stage_batch_async(
                    batch, abstract, http_client=http_client, hints=hints
                )
                llm_answers.extend(batch_answers)
        else:
            for b_idx, batch in enumerate(batches):
                state_spans = {}
                questions = {}

                for idx, c in enumerate(batch):
                    sid = f"s{idx}"
                    s_text = c.get("span_text", c.get("entity", "")).strip()
                    p_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                    s_lbl = normalize_label(c.get("second_best_label", "O"))
                    unc = round(float(c.get("uncertainty", c.get("u_score", 0.0))), 4)
                    nov = round(float(c.get("novelty_score", c.get("novelty", 0.0))), 4)
                    margin = round(float(c.get("margin", 0.0)), 4)

                    state_spans[sid] = {
                        "span_text": s_text,
                        "bert_predicted_label": p_lbl,
                        "second_best_label": s_lbl,
                        "uncertainty": unc,
                        "novelty_score": nov,
                        "margin": margin,
                    }

                    if self.active_guidelines:
                        instr = (
                            f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                            f"in `abstract`, taking into account BERT evidence in `candidate_spans.{sid}` and strictly "
                            f"adhering to domain rules in `dynamic_guidelines`."
                        )
                    else:
                        instr = (
                            f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                            f"in `abstract`, evaluating BERT evidence in `candidate_spans.{sid}` against canonical category criteria."
                        )

                    questions[sid] = {
                        "type": "choice",
                        "instructions": instr,
                        "criteria": LSF_CHOICE_CRITERIA,
                    }

                state = {
                    "abstract": abstract,
                    "dynamic_guidelines": self.active_guidelines,
                    "candidate_spans": state_spans,
                }
                if hints:
                    state["dinasor_hints"] = hints

                try:
                    resp = await self.client.system_one_async(
                        state=state,
                        questions=questions,
                        http_client=http_client,
                    )
                    for idx, c in enumerate(batch):
                        sid = f"s{idx}"
                        cid = c.get("id", len(llm_answers) + 1)
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        choice_ans = resp.choices.get(sid)

                        if choice_ans:
                            chosen_label = normalize_label(choice_ans.choice)
                            conf = choice_ans.confidence
                            probs = choice_ans.probabilities
                        else:
                            chosen_label = "O"
                            conf = 0.0
                            probs = {}

                        llm_answers.append({
                            "id": cid,
                            "entity": s_text,
                            "llm_predicted_label": chosen_label,
                            "final_label": chosen_label,
                            "label": chosen_label,
                            "confidence": conf,
                            "probabilities": probs,
                            "source": f"TypeSafe-Jev ({self.mode})",
                            "rationale": f"Jev System One choice (confidence={conf:.2f})",
                        })

                except Exception as exc:
                    for idx, c in enumerate(batch):
                        cid = c.get("id", len(llm_answers) + 1)
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        fallback_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                        llm_answers.append({
                            "id": cid,
                            "entity": s_text,
                            "llm_predicted_label": fallback_lbl,
                            "final_label": fallback_lbl,
                            "label": fallback_lbl,
                            "confidence": 0.0,
                            "probabilities": {},
                            "source": "TypeSafe-Jev-Fallback",
                            "rationale": f"Fallback to BERT due to API error: {exc}",
                        })

        labeled_entities = [
            {
                "entity": a["entity"],
                "label": a["final_label"],
                "confidence": a.get("confidence", 0.0),
                "probabilities": a.get("probabilities", {}),
                "rationale": a.get("rationale", ""),
            }
            for a in llm_answers
        ]
        assigned_bids = set()
        for a in llm_answers:
            for rid in a.get("assigned_guidelines", []):
                assigned_bids.add(rid)
        bids = sorted(list(assigned_bids)) if assigned_bids else [g.get("id") for g in self.active_guidelines[:5] if "id" in g]

        result = {
            "reasoning": f"TypeSafe Jev System One async arbitration under {self.mode}",
            "bullet_ids": bids,
            "hints_applied": bids,
            "llm_answer": llm_answers,
            "final_answer": llm_answers,
            "labeled_entities": labeled_entities,
            "raw_response": json.dumps(llm_answers),
            "prompt": "TypeSafe System One State + Questions Schema",
        }

        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result

    def _run_two_stage_batch(
        self,
        batch: List[Dict[str, Any]],
        abstract: str,
        hints: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes synchronous two-stage evaluation:
          Pass 1: Assigns applicable guideline rules from full_rules to each candidate span.
          Pass 2: Evaluates each candidate span using ONLY its assigned rules against canonical LSF categories.
        """
        batch_answers: List[Dict[str, Any]] = []
        state_spans = {}
        pass1_questions = {}

        for idx, c in enumerate(batch):
            sid = f"s{idx}"
            s_text = c.get("span_text", c.get("entity", "")).strip()
            start_c = c.get("start_char", c.get("start", -1))
            end_c = c.get("end_char", c.get("end", -1))
            sent_text = extract_sentence_for_span(abstract, s_text, start_c, end_c)
            p_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
            s_lbl = normalize_label(c.get("second_best_label", "O"))
            unc = round(float(c.get("uncertainty", c.get("u_score", 0.0))), 4)
            nov = round(float(c.get("novelty_score", c.get("novelty", 0.0))), 4)
            margin = round(float(c.get("margin", 0.0)), 4)

            state_spans[sid] = {
                "span_text": s_text,
                "sentence_context": sent_text,
                "bert_predicted_label": p_lbl,
                "second_best_label": s_lbl,
                "uncertainty": unc,
                "novelty_score": nov,
                "margin": margin,
            }
            scoped_rules = self.get_scoped_rules_for_span(p_lbl, s_lbl, max_scoped=7)
            scoped_criteria = {
                r["id"]: r.get("rule", "").strip()
                for r in scoped_rules
            }
            scoped_criteria["NONE_RELEVANT"] = (
                "None of the candidate guidelines are relevant to this entity; standard category definitions govern it directly."
            )

            pass1_questions[sid] = {
                "type": "choice",
                "instructions": (
                    f"You are the ACE Guideline Assigner. Evaluate candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context` (BERT hypothesis: {p_lbl}, secondary: {s_lbl}). "
                    f"Identify and select the domain guideline that best clarifies, validates, or governs "
                    f"the classification, inclusion criteria, or boundaries for this entity. "
                    f"If none of the candidate guidelines are relevant to this entity, choose NONE_RELEVANT."
                ),
                "criteria": scoped_criteria,
            }

        # --- PASS 1: Rule Assignment (Chunked into sub-batches of max 4 spans to avoid max_tokens_exceeded) ---
        assignments: Dict[str, List[str]] = {f"s{i}": [] for i in range(len(batch))}
        chunk_size = 4
        for start_idx in range(0, len(batch), chunk_size):
            sub_indices = list(range(start_idx, min(start_idx + chunk_size, len(batch))))
            sub_state_spans = {f"s{i}": state_spans[f"s{i}"] for i in sub_indices}
            sub_questions = {f"s{i}": pass1_questions[f"s{i}"] for i in sub_indices}
            try:
                resp1 = self.client.system_one(
                    state={"abstract": abstract, "candidate_spans": sub_state_spans},
                    questions=sub_questions,
                )
                for i in sub_indices:
                    sid = f"s{i}"
                    ans = resp1.choices.get(sid)
                    assignments[sid] = resolve_multi_rule_assignments(
                        ans,
                        max_rules=self.max_assigned_rules,
                        prob_floor=0.10,
                        relative_ratio=0.35,
                    )
            except Exception as exc:
                print(f"[WARN] Pass 1 guideline assignment error on sub-batch {sub_indices}: {exc}")

        # --- PASS 2: Scoped Classification ---
        pass2_state_spans = {}
        pass2_questions = {}
        for idx, c in enumerate(batch):
            sid = f"s{idx}"
            assigned_ids = assignments.get(sid, [])
            assigned_rules = [
                {"id": r["id"], "rule": r["rule"]}
                for r in self.two_stage_rules
                if r["id"] in assigned_ids
            ]
            s_dict = dict(state_spans[sid])
            s_dict["assigned_guidelines"] = assigned_rules
            pass2_state_spans[sid] = s_dict

            if assigned_rules:
                instr = (
                    f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context` into one of the 9 Lifestyle Factor categories or Non-LSF (O). "
                    f"Evaluate BERT evidence against canonical category criteria, using the assigned domain rule(s) in "
                    f"`candidate_spans.{sid}.assigned_guidelines` as clarifying boundary guidance. "
                    f"Do not classify as O unless the entity explicitly meets an exclusion boundary rather than standard category definitions."
                )
            else:
                instr = (
                    f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context`, evaluating BERT evidence against canonical category criteria."
                )
            pass2_questions[sid] = {
                "type": "choice",
                "instructions": instr,
                "criteria": LSF_CHOICE_CRITERIA,
            }

        state2 = {
            "abstract": abstract,
            "candidate_spans": pass2_state_spans,
        }
        if hints:
            state2["dinasor_hints"] = hints

        try:
            resp2 = self.client.system_one(state=state2, questions=pass2_questions)
            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                cid = c.get("id", idx + 1)
                s_text = c.get("span_text", c.get("entity", "")).strip()
                choice_ans = resp2.choices.get(sid)
                assigned_ids = assignments.get(sid, [])

                if choice_ans:
                    chosen_label = normalize_label(choice_ans.choice)
                    conf = choice_ans.confidence
                    probs = choice_ans.probabilities
                else:
                    chosen_label = "O"
                    conf = 0.0
                    probs = {}

                batch_answers.append({
                    "id": cid,
                    "entity": s_text,
                    "llm_predicted_label": chosen_label,
                    "final_label": chosen_label,
                    "label": chosen_label,
                    "confidence": conf,
                    "probabilities": probs,
                    "assigned_guidelines": assigned_ids,
                    "bullet_ids": assigned_ids,
                    "guidelines_ids": assigned_ids,
                    "source": f"TypeSafe-Jev ({self.mode})",
                    "rationale": f"Jev Two-Stage choice (assigned={assigned_ids}, conf={conf:.2f})",
                })
        except Exception as exc:
            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                cid = c.get("id", idx + 1)
                s_text = c.get("span_text", c.get("entity", "")).strip()
                fallback_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                batch_answers.append({
                    "id": cid,
                    "entity": s_text,
                    "llm_predicted_label": fallback_lbl,
                    "final_label": fallback_lbl,
                    "label": fallback_lbl,
                    "confidence": 0.0,
                    "probabilities": {},
                    "assigned_guidelines": assignments.get(sid, []),
                    "source": "TypeSafe-Jev-Fallback",
                    "rationale": f"Fallback to BERT due to API error: {exc}",
                })

        return batch_answers

    async def _run_two_stage_batch_async(
        self,
        batch: List[Dict[str, Any]],
        abstract: str,
        http_client: Optional[Any] = None,
        hints: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes asynchronous two-stage evaluation:
          Pass 1: Assigns applicable guideline rules from full_rules to each candidate span.
          Pass 2: Evaluates each candidate span using ONLY its assigned rules against canonical LSF categories.
        """
        batch_answers: List[Dict[str, Any]] = []
        state_spans = {}
        pass1_questions = {}

        for idx, c in enumerate(batch):
            sid = f"s{idx}"
            s_text = c.get("span_text", c.get("entity", "")).strip()
            start_c = c.get("start_char", c.get("start", -1))
            end_c = c.get("end_char", c.get("end", -1))
            sent_text = extract_sentence_for_span(abstract, s_text, start_c, end_c)
            p_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
            s_lbl = normalize_label(c.get("second_best_label", "O"))
            unc = round(float(c.get("uncertainty", c.get("u_score", 0.0))), 4)
            nov = round(float(c.get("novelty_score", c.get("novelty", 0.0))), 4)
            margin = round(float(c.get("margin", 0.0)), 4)

            state_spans[sid] = {
                "span_text": s_text,
                "sentence_context": sent_text,
                "bert_predicted_label": p_lbl,
                "second_best_label": s_lbl,
                "uncertainty": unc,
                "novelty_score": nov,
                "margin": margin,
            }
            scoped_rules = self.get_scoped_rules_for_span(p_lbl, s_lbl, max_scoped=7)
            scoped_criteria = {
                r["id"]: r.get("rule", "").strip()
                for r in scoped_rules
            }
            scoped_criteria["NONE_RELEVANT"] = (
                "None of the candidate guidelines are relevant to this entity; standard category definitions govern it directly."
            )

            pass1_questions[sid] = {
                "type": "choice",
                "instructions": (
                    f"You are the ACE Guideline Assigner. Evaluate candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context` (BERT hypothesis: {p_lbl}, secondary: {s_lbl}). "
                    f"Identify and select the domain guideline that best clarifies, validates, or governs "
                    f"the classification, inclusion criteria, or boundaries for this entity. "
                    f"If none of the candidate guidelines are relevant to this entity, choose NONE_RELEVANT."
                ),
                "criteria": scoped_criteria,
            }

        # --- PASS 1: Rule Assignment (Chunked into sub-batches of max 4 spans evaluated concurrently) ---
        assignments: Dict[str, List[str]] = {f"s{i}": [] for i in range(len(batch))}
        chunk_size = 4
        pass1_tasks = []
        sub_indices_list = []
        for start_idx in range(0, len(batch), chunk_size):
            sub_indices = list(range(start_idx, min(start_idx + chunk_size, len(batch))))
            sub_indices_list.append(sub_indices)
            sub_state_spans = {f"s{i}": state_spans[f"s{i}"] for i in sub_indices}
            sub_questions = {f"s{i}": pass1_questions[f"s{i}"] for i in sub_indices}
            pass1_tasks.append(
                self.client.system_one_async(
                    state={"abstract": abstract, "candidate_spans": sub_state_spans},
                    questions=sub_questions,
                    http_client=http_client,
                )
            )

        if pass1_tasks:
            sub_responses = await asyncio.gather(*pass1_tasks, return_exceptions=True)
            for sub_indices, resp1 in zip(sub_indices_list, sub_responses):
                if isinstance(resp1, Exception):
                    print(f"[WARN] Pass 1 async assignment error on sub-batch {sub_indices}: {resp1}")
                    continue
                for i in sub_indices:
                    sid = f"s{i}"
                    ans = resp1.choices.get(sid)
                    assignments[sid] = resolve_multi_rule_assignments(
                        ans,
                        max_rules=self.max_assigned_rules,
                        prob_floor=0.10,
                        relative_ratio=0.35,
                    )

        # --- PASS 2: Scoped Classification ---
        pass2_state_spans = {}
        pass2_questions = {}
        for idx, c in enumerate(batch):
            sid = f"s{idx}"
            assigned_ids = assignments.get(sid, [])
            assigned_rules = [
                {"id": r["id"], "rule": r["rule"]}
                for r in self.two_stage_rules
                if r["id"] in assigned_ids
            ]
            s_dict = dict(state_spans[sid])
            s_dict["assigned_guidelines"] = assigned_rules
            pass2_state_spans[sid] = s_dict

            if assigned_rules:
                instr = (
                    f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context` into one of the 9 Lifestyle Factor categories or Non-LSF (O). "
                    f"Evaluate BERT evidence against canonical category criteria, using the assigned domain rule(s) in "
                    f"`candidate_spans.{sid}.assigned_guidelines` as clarifying boundary guidance. "
                    f"Do not classify as O unless the entity explicitly meets an exclusion boundary rather than standard category definitions."
                )
            else:
                instr = (
                    f"You are the ACE Generator. Classify candidate span `candidate_spans.{sid}.span_text` "
                    f"in sentence `candidate_spans.{sid}.sentence_context`, evaluating BERT evidence against canonical category criteria."
                )
            pass2_questions[sid] = {
                "type": "choice",
                "instructions": instr,
                "criteria": LSF_CHOICE_CRITERIA,
            }

        state2 = {
            "abstract": abstract,
            "candidate_spans": pass2_state_spans,
        }
        if hints:
            state2["dinasor_hints"] = hints

        try:
            resp2 = await self.client.system_one_async(
                state=state2,
                questions=pass2_questions,
                http_client=http_client,
            )
            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                cid = c.get("id", idx + 1)
                s_text = c.get("span_text", c.get("entity", "")).strip()
                choice_ans = resp2.choices.get(sid)
                assigned_ids = assignments.get(sid, [])

                if choice_ans:
                    chosen_label = normalize_label(choice_ans.choice)
                    conf = choice_ans.confidence
                    probs = choice_ans.probabilities
                else:
                    chosen_label = "O"
                    conf = 0.0
                    probs = {}

                batch_answers.append({
                    "id": cid,
                    "entity": s_text,
                    "llm_predicted_label": chosen_label,
                    "final_label": chosen_label,
                    "label": chosen_label,
                    "confidence": conf,
                    "probabilities": probs,
                    "assigned_guidelines": assigned_ids,
                    "bullet_ids": assigned_ids,
                    "guidelines_ids": assigned_ids,
                    "source": f"TypeSafe-Jev ({self.mode})",
                    "rationale": f"Jev Two-Stage choice (assigned={assigned_ids}, conf={conf:.2f})",
                })
        except Exception as exc:
            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                cid = c.get("id", idx + 1)
                s_text = c.get("span_text", c.get("entity", "")).strip()
                fallback_lbl = normalize_label(c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "O"))))
                batch_answers.append({
                    "id": cid,
                    "entity": s_text,
                    "llm_predicted_label": fallback_lbl,
                    "final_label": fallback_lbl,
                    "label": fallback_lbl,
                    "confidence": 0.0,
                    "probabilities": {},
                    "assigned_guidelines": assignments.get(sid, []),
                    "source": "TypeSafe-Jev-Fallback",
                    "rationale": f"Fallback to BERT due to API error: {exc}",
                })

        return batch_answers


def attach_jev_generator_to_pipeline(
    pipeline: Any,
    mode: str = "mode2",
    guidebook_path: Optional[str] = None,
    model_name: str = "jev-latest",
) -> Any:
    """
    Attaches JevGenerator to an existing pipeline instance (BertizedACEPipeline or DinasorPipeline)
    WITHOUT modifying any underlying pipeline code.
    """
    gen = JevGenerator(
        mode=mode,
        guidebook_path=guidebook_path,
        model_name=model_name,
    )
    if hasattr(pipeline, "generator"):
        pipeline.generator = gen
        print(f"✅ Attached JevGenerator ({mode}) to pipeline.generator (BertizedACE).")
    if hasattr(pipeline, "dingenerator_agent"):
        pipeline.dingenerator_agent = gen
        print(f"✅ Attached JevGenerator ({mode}) to pipeline.dingenerator_agent (DinGenPipe).")
    return pipeline
