"""
Escalation Arbitrator for JeBert Architecture.

Provides heavy LLM arbitration for candidate spans escalated by JeBertGating.
Supports 3 operational modes:
  1. 'ace_generator' (Default): Single-pass dual-hypothesis arbitration prompt.
     Arbitrates between SpanNER hypothesis and Jev hypothesis using domain guidelines.
  2. 'discovery': Arbitrates escalated spans AND scans the abstract for missed entities
     omitted by SpanNER proposal.
  3. 'dinasor_dingen': Two-call protocol:
     - Call 1: Dinasor extracts contextual hints from SpanNER vs Jev conflict.
     - Call 2: DinGen classifies the span conditioned on the extracted hints.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.llm_client import generate_llm_response
from src.common.label_mapping import canonicalize_label


class JeBertArbitrator:
    """
    Arbitrates escalated candidate spans using an LLM.
    """

    def __init__(
        self,
        mode: str = "ace_generator",
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Args:
            mode: 'ace_generator' (default), 'discovery', or 'dinasor_dingen'.
            model_name: LLM model identifier (e.g. gpt-4o-mini, gemini-cli).
            backend: LLM backend provider ('api', 'openai', 'lmstudio', 'dummy').
            temperature: Sampling temperature (default 0.0 for deterministic arbitration).
        """
        self.mode = mode.lower()
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature

    def arbitrate(
        self,
        abstract: str,
        escalated_spans: List[Dict[str, Any]],
        guidelines: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Arbitrates escalated candidate spans.
        Returns:
          (arbitrated_spans, discovered_spans)
        """
        if not escalated_spans and self.mode != "discovery":
            return [], []

        if self.mode == "discovery":
            return self._arbitrate_with_discovery(abstract, escalated_spans, guidelines)
        elif self.mode == "dinasor_dingen":
            return self._arbitrate_dinasor_dingen(abstract, escalated_spans, guidelines)
        else:  # default: 'ace_generator'
            return self._arbitrate_ace_generator(abstract, escalated_spans, guidelines)

    def _format_candidates_block(self, escalated_spans: List[Dict[str, Any]]) -> str:
        lines = []
        for s in escalated_spans:
            sid = s.get("id", "")
            text = s.get("span_text", "")
            s_lbl = s.get("spanner_label", s.get("bert_predicted_label", "O"))
            margin = s.get("margin", 0.0)
            u_score = s.get("uncertainty", 0.0)
            j_lbl = s.get("jev_label", "O")
            j_conf = s.get("jev_confidence", 0.0)
            reason = s.get("gating_reason", "Escalated for review")
            assigned = s.get("assigned_guidelines", [])

            lines.append(
                f"- [ID: {sid}] \"{text}\"\n"
                f"    • SpanNER hypothesis: {s_lbl} (margin={margin:.2f}, uncertainty={u_score:.2f})\n"
                f"    • Jev hypothesis:     {j_lbl} (confidence={j_conf:.2f}, rules={assigned})\n"
                f"    • Conflict reason:    {reason}"
            )
        return "\n".join(lines)

    def _parse_llm_json(self, raw_resp: str) -> Dict[str, Any]:
        """Safely parses JSON response from LLM, stripping markdown fences."""
        cleaned = raw_resp.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of json object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"arbitrated_entities": []}

    def _arbitrate_ace_generator(
        self,
        abstract: str,
        escalated_spans: List[Dict[str, Any]],
        guidelines: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Single-pass arbitration prompt."""
        candidates_str = self._format_candidates_block(escalated_spans)

        prompt = (
            f"You are the JeBert Senior Arbitrator for Lifestyle Factor (LSF) Named Entity Recognition.\n"
            f"Your job is to resolve difficult, ambiguous, or conflicting entity candidate spans in a biomedical abstract.\n\n"
            f"=== BIOMEDICAL ABSTRACT ===\n"
            f"{abstract}\n\n"
            f"=== CANDIDATE SPANS REQUIRING ARBITRATION ===\n"
            f"{candidates_str}\n\n"
            f"=== DOMAIN GUIDELINES & BOUNDARIES ===\n"
            f"{guidelines if guidelines else 'Standard 9 Lifestyle Factor canonical criteria apply.'}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"1. For each candidate span, evaluate its contextual meaning in the abstract.\n"
            f"2. Consider both competing model predictions (SpanNER vs Jev) and arbitrate the definitive label.\n"
            f"   Valid categories: Physical_activities, Nutrition, Substance_use, Sleep, Mental_health_practices,\n"
            f"   Non_physical_leisure_time_activities, Personal_care_products_and_cosmetic_procedures, Socioeconomic_factors,\n"
            f"   Environmental_exposures, or 'O' (Non-LSF / background).\n"
            f"3. Return ONLY a valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "arbitrated_entities": [\n'
            f'    {{\n'
            f'      "id": <span_id>,\n'
            f'      "span_text": "<text>",\n'
            f'      "final_label": "<Category or O>",\n'
            f'      "rationale": "<brief explanation>"\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
        )

        raw_resp = generate_llm_response(
            prompt=prompt,
            model_name=self.model_name,
            backend=self.backend,
            temperature=self.temperature,
        )
        parsed = self._parse_llm_json(raw_resp)
        arb_list = parsed.get("arbitrated_entities", [])
        arb_map = {item.get("id"): item for item in arb_list}

        arbitrated_spans = []
        for s in escalated_spans:
            sid = s.get("id")
            res_item = arb_map.get(sid, {})
            final_lbl = canonicalize_label(
                res_item.get("final_label", s.get("jev_label", s.get("spanner_label", "O"))),
                use_new_labels=True,
            )
            item_copy = dict(s)
            item_copy["label"] = final_lbl
            item_copy["ace_label"] = final_lbl
            item_copy["final_label"] = final_lbl
            item_copy["arbitration_rationale"] = res_item.get("rationale", "Arbitrated by JeBert ACE Generator")
            item_copy["source"] = f"JeBert-Arbitrator ({self.mode})"
            arbitrated_spans.append(item_copy)

        return arbitrated_spans, []

    def _arbitrate_with_discovery(
        self,
        abstract: str,
        escalated_spans: List[Dict[str, Any]],
        guidelines: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Arbitration with simultaneous missed entity discovery pass."""
        candidates_str = self._format_candidates_block(escalated_spans)

        prompt = (
            f"You are the JeBert Senior Arbitrator & Entity Discovery Specialist.\n"
            f"Your job is two-fold:\n"
            f"1. Resolve difficult, ambiguous, or conflicting entity candidate spans.\n"
            f"2. Scan the abstract for any additional Lifestyle Factor entities that were completely missed\n"
            f"   by the candidate proposers, identifying their exact text and character offsets.\n\n"
            f"=== BIOMEDICAL ABSTRACT ===\n"
            f"{abstract}\n\n"
            f"=== CANDIDATE SPANS REQUIRING ARBITRATION ===\n"
            f"{candidates_str if candidates_str else 'No ambiguous spans escalated.'}\n\n"
            f"=== DOMAIN GUIDELINES & BOUNDARIES ===\n"
            f"{guidelines if guidelines else 'Standard 9 Lifestyle Factor canonical criteria apply.'}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Return ONLY a valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "arbitrated_entities": [\n'
            f'    {{\n'
            f'      "id": <span_id>,\n'
            f'      "span_text": "<text>",\n'
            f'      "final_label": "<Category or O>",\n'
            f'      "rationale": "<brief explanation>"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "discovered_entities": [\n'
            f'    {{\n'
            f'      "span_text": "<missed entity exact text>",\n'
            f'      "final_label": "<Category>",\n'
            f'      "rationale": "<why this was missed and qualifies as LSF>"\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
        )

        raw_resp = generate_llm_response(
            prompt=prompt,
            model_name=self.model_name,
            backend=self.backend,
            temperature=self.temperature,
        )
        parsed = self._parse_llm_json(raw_resp)
        arb_list = parsed.get("arbitrated_entities", [])
        arb_map = {item.get("id"): item for item in arb_list}

        arbitrated_spans = []
        for s in escalated_spans:
            sid = s.get("id")
            res_item = arb_map.get(sid, {})
            final_lbl = canonicalize_label(
                res_item.get("final_label", s.get("jev_label", s.get("spanner_label", "O"))),
                use_new_labels=True,
            )
            item_copy = dict(s)
            item_copy["label"] = final_lbl
            item_copy["ace_label"] = final_lbl
            item_copy["final_label"] = final_lbl
            item_copy["arbitration_rationale"] = res_item.get("rationale", "Arbitrated by JeBert Discovery")
            item_copy["source"] = f"JeBert-Arbitrator ({self.mode})"
            arbitrated_spans.append(item_copy)

        discovered_spans = []
        raw_discovered = parsed.get("discovered_entities", [])
        for d in raw_discovered:
            text = d.get("span_text", "").strip()
            cat = canonicalize_label(d.get("final_label", "O"), use_new_labels=True)
            if not text or cat == "O":
                continue
            # Locate exact character offsets in abstract
            idx = abstract.find(text)
            if idx != -1:
                discovered_spans.append({
                    "id": f"disc_{len(discovered_spans)+1}",
                    "span_text": text,
                    "start_char": idx,
                    "end_char": idx + len(text),
                    "label": cat,
                    "spanner_label": "O",
                    "jev_label": "O",
                    "ace_label": cat,
                    "final_label": cat,
                    "confidence": 1.0,
                    "uncertainty": 0.0,
                    "novelty": 0.0,
                    "margin": 0.0,
                    "arbitration_rationale": d.get("rationale", "Discovered by LLM abstract scan"),
                    "source": "JeBert-Discovery-Pass",
                    "escalated_to_llm": True,
                })

        return arbitrated_spans, discovered_spans

    def _arbitrate_dinasor_dingen(
        self,
        abstract: str,
        escalated_spans: List[Dict[str, Any]],
        guidelines: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Two-call protocol: Dinasor hints + DinGen resolution."""
        candidates_str = self._format_candidates_block(escalated_spans)

        # Call 1: Dinasor Reflector / Hint Extractor
        dinasor_prompt = (
            f"You are Dinasor, the contextual hint generator for conflicting NER predictions.\n"
            f"Analyze the discrepancy between SpanNER and Jev for each candidate span.\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Conflicting Candidates:\n{candidates_str}\n\n"
            f"For each candidate, provide 1-2 sentence evidence hints clarifying whether the entity\n"
            f"is a true Lifestyle Factor or non-LSF ('O').\n"
            f"Output JSON: {{\"hints\": [{{\"id\": <id>, \"hint\": \"...\"}}]}}"
        )
        hint_resp = generate_llm_response(
            prompt=dinasor_prompt,
            model_name=self.model_name,
            backend=self.backend,
            temperature=self.temperature,
        )
        hints_data = self._parse_llm_json(hint_resp).get("hints", [])
        hint_map = {h.get("id"): h.get("hint", "") for h in hints_data}

        # Call 2: DinGen Classifier conditioned on Dinasor hints
        lines = []
        for s in escalated_spans:
            sid = s.get("id")
            lines.append(
                f"- [ID: {sid}] \"{s.get('span_text')}\" (SpanNER: {s.get('spanner_label')}, Jev: {s.get('jev_label')})\n"
                f"  Dinasor Hint: {hint_map.get(sid, 'No hint provided')}"
            )
        cand_with_hints = "\n".join(lines)

        dingen_prompt = (
            f"You are DinGen. Classify the candidate spans into one of the 9 LSF categories or 'O',\n"
            f"conditioned on the contextual hints from Dinasor.\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Candidates with Hints:\n{cand_with_hints}\n\n"
            f"Output JSON: {{\"arbitrated_entities\": [{{\"id\": <id>, \"final_label\": \"<Category or O>\", \"rationale\": \"...\"}}]}}"
        )
        dingen_resp = generate_llm_response(
            prompt=dingen_prompt,
            model_name=self.model_name,
            backend=self.backend,
            temperature=self.temperature,
        )
        parsed = self._parse_llm_json(dingen_resp)
        arb_list = parsed.get("arbitrated_entities", [])
        arb_map = {item.get("id"): item for item in arb_list}

        arbitrated_spans = []
        for s in escalated_spans:
            sid = s.get("id")
            res_item = arb_map.get(sid, {})
            final_lbl = canonicalize_label(
                res_item.get("final_label", s.get("jev_label", s.get("spanner_label", "O"))),
                use_new_labels=True,
            )
            item_copy = dict(s)
            item_copy["label"] = final_lbl
            item_copy["ace_label"] = final_lbl
            item_copy["final_label"] = final_lbl
            item_copy["dinasor_hint"] = hint_map.get(sid, "")
            item_copy["arbitration_rationale"] = res_item.get("rationale", "Arbitrated by Dinasor+DinGen")
            item_copy["source"] = "JeBert-Dinasor-DinGen"
            arbitrated_spans.append(item_copy)

        return arbitrated_spans, []
