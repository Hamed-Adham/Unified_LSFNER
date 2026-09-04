from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from src.common.llm_client import generate_llm_response as call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_dingen_prompt(version: Optional[str] = None, custom_path: Optional[str] = None) -> Path:
    """
    Dynamically resolves the DinGenerator prompt path.
    Supports versions: 'v1', 'v2', 'v3' (or '1', '2', '3').
    Defaults to 'v3' or env var DINGEN_PROMPT_VERSION / PROMPT_VERSION.
    """
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    ver = (version or os.getenv("DINGEN_PROMPT_VERSION") or os.getenv("PROMPT_VERSION") or "v3").lower().strip()
    if not ver.startswith("v"):
        ver = f"v{ver}"

    base_dir = PROJECT_ROOT / "src" / "prompts" / "dinasor_dingen"
    versioned_file = base_dir / f"DinGenerator_{ver}.txt"
    if versioned_file.exists():
        return versioned_file

    default_file = base_dir / "DinGenerator.txt"
    if default_file.exists():
        return default_file

    for fallback in [
        PROJECT_ROOT / "src" / "prompts" / "DinGenerator.txt",
        PROJECT_ROOT / "prompts" / "DinGenerator.txt",
        PROJECT_ROOT / "data" / "guidelines" / "DinGenerator.txt"
    ]:
        if fallback.exists():
            return fallback

    return versioned_file


class DinGenerator:
    def __init__(
        self,
        model_name: Optional[str] = None,
        file_format: str = "BIO",
        backend: str = "lmstudio",
        max_tokens: Optional[int] = None,
        prompt_version: Optional[str] = None,
        prompt_path: Optional[str] = None
    ):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-cli")
        self.backend = backend
        self.file_format = file_format
        self.max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.prompt_version = (prompt_version or os.getenv("DINGEN_PROMPT_VERSION") or os.getenv("PROMPT_VERSION") or "v3").lower().strip()
        self.prompt_path = resolve_dingen_prompt(version=self.prompt_version, custom_path=prompt_path)

    def run(
        self,
        abstract: str,
        hints: list | dict | str = None,
        uncertain_entities: list[dict] | list | dict | str = None,
        save_output_path: str = None
    ) -> dict:
        """
        Labels given uncertain entity mentions based on the biomedical abstract
        and targeted disambiguation hints from Dinasor.

        Args:
            abstract: Input biomedical abstract text or context
            hints: Targeted hints list, string, or dict from Dinasor
            uncertain_entities: List of candidate entities with constraints and evidence
            save_output_path: Optional path to save raw LLM response

        Returns:
            Dictionary containing:
                - reasoning: Explanation of resolution
                - hints_applied: List of applied hints
                - labeled_entities: List of labeled entity dicts
                - raw_response: Raw response string
        """
        # Format hints
        if isinstance(hints, list):
            formatted_hints = []
            for i, h in enumerate(hints):
                if isinstance(h, dict):
                    ent = h.get("target_entity", h.get("entity", ""))
                    rh = h.get("resolution_hint", h.get("hint", str(h)))
                    comp = h.get("competing_categories", [])
                    comp_str = f" [Competing: {', '.join(comp)}]" if comp else ""
                    formatted_hints.append(f"- Hint {i+1} for '{ent}'{comp_str}: {rh}")
                else:
                    h_str = str(h).strip()
                    if not any(h_str.startswith(p) for p in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "Hint", "-")):
                        formatted_hints.append(f"- Hint {i+1}: {h_str}")
                    else:
                        formatted_hints.append(f"- {h_str}" if not h_str.startswith("-") else h_str)
            hints_str = "\n".join(formatted_hints) if formatted_hints else "No specific hints provided."
        elif isinstance(hints, dict):
            formatted_hints = []
            if "hints" in hints and isinstance(hints["hints"], list):
                for i, h in enumerate(hints["hints"]):
                    if isinstance(h, dict):
                        ent = h.get("target_entity", "")
                        rh = h.get("resolution_hint", "")
                        comp = h.get("competing_categories", [])
                        comp_str = f" [Competing: {', '.join(comp)}]" if comp else ""
                        formatted_hints.append(f"- Hint {i+1} for '{ent}'{comp_str}: {rh}")
                    else:
                        formatted_hints.append(f"- Hint {i+1}: {h}")
                hints_str = "\n".join(formatted_hints)
            else:
                hints_str = json.dumps(hints, indent=2)
        elif hints:
            hints_str = str(hints)
        else:
            hints_str = "No specific hints provided."

        # Keep both variable names available for prompt interpolation
        hints = hints_str

        # Format uncertain_entities
        if isinstance(uncertain_entities, list):
            formatted_entities = []
            for item in uncertain_entities:
                if isinstance(item, dict):
                    entity = item.get("entity", item.get("span_text", ""))
                    tag = item.get("predicted_tag", item.get("spanner_label", ""))
                    ch_type = item.get("challenge_type", "")
                    allowed = item.get("allowed_categories", [])
                    p_o = item.get("p_background_o", item.get("spanner_evidence", {}).get("p_background_o", ""))
                    
                    parts = [f"Entity: '{entity}'"]
                    if ch_type:
                        parts.append(f"Mode: {ch_type}")
                    if allowed:
                        parts.append(f"Allowed Categories: {allowed}")
                    elif tag:
                        parts.append(f"Predicted Tag: {tag}")
                    if p_o != "":
                        parts.append(f"P(Background O): {p_o}")
                    
                    formatted_entities.append("- " + " | ".join(parts))
                else:
                    formatted_entities.append(f"- {item}")
            uncertain_entities_str = "\n".join(formatted_entities) if formatted_entities else "None provided."
        elif isinstance(uncertain_entities, dict):
            uncertain_entities_str = json.dumps(uncertain_entities, indent=2)
        elif uncertain_entities:
            uncertain_entities_str = str(uncertain_entities)
        else:
            uncertain_entities_str = "None provided."

        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        prompt = eval(f"f'''{prompt}'''")
        response = call_llm(
            prompt,
            model_name=self.model_name,
            backend=self.backend,
            system_prompt="DinGenerator",
            max_tokens=self.max_tokens
        )

        # Save prompt and raw response if path is provided
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("DINGENERATOR INPUT PROMPT (FINAL FILLED)\n")
                f.write("=" * 80 + "\n\n")
                f.write(str(prompt or ""))
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("DINGENERATOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(str(response or ""))
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF DINGENERATOR RESPONSE\n")
                f.write("=" * 80 + "\n")

        # Parse structured JSON matching the DinGenerator.txt schema
        try:
            parsed = json.loads(response) if response else {}
            reasoning = parsed.get("reasoning", "")
            hints_applied = parsed.get("hints_applied", [])
            labeled_entities = parsed.get("labeled_entities", [])
        except (json.JSONDecodeError, TypeError):
            reasoning = str(response or "").strip()
            hints_applied = []
            labeled_entities = []

        return {
            "reasoning": reasoning,
            "hints_applied": hints_applied,
            "labeled_entities": labeled_entities,
            "raw_response": str(response or ""),
            "prompt": prompt
        }
