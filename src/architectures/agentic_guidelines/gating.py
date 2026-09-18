import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class ACEGating:
    """
    Selective Gating & Escalation Engine for BERTized ACE.
    
    Evaluates upstream SpanNER candidate spans based on:
      - Uncertainty score (u_score)
      - Novelty score (from LOF/Qwen embeddings)
      - Prediction margin & P(O) background probability
      
    Routes high-confidence spans directly to final output (0 LLM cost)
    and selectively escalates ambiguous spans to the Generator LLM.
    Provides optional Margin Safeguard Veto to protect high-margin SpanNER entities.
    """
    def __init__(
        self,
        gating_mode: str = "prob_or",
        unreliability_threshold: float = 0.35,
        novelty_threshold: float = 0.55,
        w_uncertainty: float = 0.70,
        w_novelty: float = 0.30,
        enable_margin_safeguard_veto: bool = True,
        margin_threshold: float = 0.50,
        p_o_threshold: float = 0.20,
        include_confident_as_readonly: bool = False,
    ):
        self.gating_mode = gating_mode.lower()
        self.unreliability_threshold = float(unreliability_threshold)
        self.novelty_threshold = float(novelty_threshold)
        self.w_uncertainty = float(w_uncertainty)
        self.w_novelty = float(w_novelty)
        self.enable_margin_safeguard_veto = bool(enable_margin_safeguard_veto)
        self.margin_threshold = float(margin_threshold)
        self.p_o_threshold = float(p_o_threshold)
        self.include_confident_as_readonly = bool(include_confident_as_readonly)

    def compute_unreliability(self, u_score: float, novelty: float) -> float:
        """Computes unreliability score according to the configured gating mode."""
        u_score = max(0.0, min(1.0, float(u_score)))
        novelty = max(0.0, min(1.0, float(novelty)))

        if self.gating_mode == "prob_or":
            return float(1.0 - (1.0 - novelty) * (1.0 - u_score))
        elif self.gating_mode == "or":
            return float(max(u_score, novelty))
        elif self.gating_mode == "copula":
            return float(np.clip(u_score + novelty - 1.5 * (u_score * novelty), 0.0, 1.0))
        elif self.gating_mode == "and":
            return float(min(u_score, novelty))
        else:  # "weighted" default
            return float(self.w_novelty * novelty + self.w_uncertainty * u_score)

    def should_escalate(self, u_score: float, novelty: float) -> Tuple[bool, float]:
        """
        Determines whether a candidate span warrants LLM escalation.
        Returns: (escalate: bool, unreliability: float)
        """
        unreliability = self.compute_unreliability(u_score, novelty)

        if self.gating_mode == "or":
            escalate = (u_score >= self.unreliability_threshold) or (novelty >= self.novelty_threshold)
        elif self.gating_mode == "and":
            escalate = (u_score >= self.unreliability_threshold) and (novelty >= self.novelty_threshold)
        else:
            escalate = unreliability >= self.unreliability_threshold

        return bool(escalate), unreliability

    def filter_candidates(
        self,
        candidate_spans: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits candidates into:
          1. escalated: Ambiguous spans sent to Generator LLM
          2. confident: High-confidence spans kept with SpanNER's own prediction
        """
        escalated_cands = []
        confident_cands = []

        for idx, c in enumerate(candidate_spans, 1):
            cand_copy = dict(c)
            # Ensure consistent ID
            cid = cand_copy.get("id", idx)
            cand_copy["id"] = cid

            u_score = float(cand_copy.get("uncertainty", cand_copy.get("u_score", 0.0)))
            novelty = float(cand_copy.get("novelty_score", cand_copy.get("novelty", 0.0)))
            margin = float(cand_copy.get("margin", 0.0))
            p_lbl = cand_copy.get("bert_predicted_label", cand_copy.get("predicted_label", cand_copy.get("spanner_label", "O")))

            escalate, unreliability = self.should_escalate(u_score, novelty)
            cand_copy["uncertainty"] = round(u_score, 4)
            cand_copy["novelty_score"] = round(novelty, 4)
            cand_copy["margin"] = round(margin, 4)
            cand_copy["unreliability"] = round(unreliability, 4)
            cand_copy["escalated_to_llm"] = escalate
            cand_copy["bert_predicted_label"] = p_lbl

            if escalate:
                escalated_cands.append(cand_copy)
            else:
                confident_cands.append(cand_copy)

        return escalated_cands, confident_cands

    def apply_margin_veto(
        self,
        cand: Dict[str, Any],
        arbitrated_label: str
    ) -> Tuple[str, bool, str]:
        """
        Applies Margin Safeguard Veto:
        If LLM dropped an escalated span to Non-LSF ('O'), but SpanNER had high margin
        (margin >= margin_threshold) and low P(O) (< p_o_threshold), veto the drop
        and preserve SpanNER's predicted category.
        
        Returns: (final_label, veto_triggered, veto_reason)
        """
        if not self.enable_margin_safeguard_veto:
            return arbitrated_label, False, ""

        if arbitrated_label in ("O", "Non-LSF"):
            cand_margin = float(cand.get("margin", 0.0))
            cand_p_o = float(cand.get("p_o", cand.get("p_background_o", 0.0)))
            cand_spanner = cand.get("bert_predicted_label", cand.get("predicted_label", cand.get("spanner_label", "O")))

            if cand_margin >= self.margin_threshold and cand_p_o < self.p_o_threshold and cand_spanner not in ("O", "Non-LSF"):
                reason = (
                    f"SpanNER Margin Veto (Margin {cand_margin:.4f} >= {self.margin_threshold:.2f} "
                    f"& P(O) {cand_p_o:.2f} < {self.p_o_threshold:.2f} protected from LLM drop)"
                )
                return cand_spanner, True, reason

        return arbitrated_label, False, ""


def build_gated_anchored_abstract(
    abstract: str,
    escalated_spans: List[Dict[str, Any]],
    confident_spans: Optional[List[Dict[str, Any]]] = None,
    include_confident: bool = False
) -> str:
    """
    Inserts candidate span anchors into the abstract text.
    If include_confident is False (default), only escalated spans are anchored.
    If include_confident is True, confident spans are also anchored with a read-only tag.
    """
    events = []

    # 1. Escalated spans (for LLM arbitration)
    for c in escalated_spans:
        cid = c.get("id")
        s_text = c.get("span_text", c.get("entity", ""))
        start_char = c.get("start_char")
        end_char = c.get("end_char")
        if start_char is None or end_char is None or start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue
        span_len = end_char - start_char
        events.append((end_char, 0, span_len, f"]"))
        events.append((start_char, 1, -span_len, f"[id: {cid}, span: "))

    # 2. Confident spans (if enabled, tagged as read-only background context)
    if include_confident and confident_spans:
        for c in confident_spans:
            s_text = c.get("span_text", c.get("entity", ""))
            lbl = c.get("bert_predicted_label", c.get("predicted_label", c.get("spanner_label", "LSF")))
            start_char = c.get("start_char")
            end_char = c.get("end_char")
            if start_char is None or end_char is None or start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
                found = abstract.find(s_text)
                if found != -1:
                    start_char = found
                    end_char = found + len(s_text)
                else:
                    continue
            span_len = end_char - start_char
            events.append((end_char, 0, span_len, f" (CONFIRMED_SPANNER: {lbl})]"))
            events.append((start_char, 1, -span_len, f"[READONLY_CONFIRMED: "))

    if not events:
        return abstract

    events.sort(key=lambda x: (x[0], x[1], x[2]))

    out = []
    curr_pos = 0
    for pos, etype, prio, tag in events:
        if pos > curr_pos:
            out.append(abstract[curr_pos:pos])
            curr_pos = pos
        out.append(tag)
    if curr_pos < len(abstract):
        out.append(abstract[curr_pos:])

    return "".join(out)
