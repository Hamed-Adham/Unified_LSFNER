from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from src.common.llm_client import generate_llm_response as call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_dinasor_prompt(version: Optional[str] = None, custom_path: Optional[str] = None) -> Path:
    """
    Dynamically resolves the Dinasor prompt path.
    Supports versions: 'v1', 'v2', 'v3' (or '1', '2', '3').
    Defaults to 'v3' or env var DINASOR_PROMPT_VERSION / PROMPT_VERSION.
    """
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    ver = (version or os.getenv("DINASOR_PROMPT_VERSION") or os.getenv("PROMPT_VERSION") or "v3").lower().strip()
    if not ver.startswith("v"):
        ver = f"v{ver}"

    base_dir = PROJECT_ROOT / "src" / "prompts" / "dinasor_dingen"
    versioned_file = base_dir / f"Dinasor_{ver}.txt"
    if versioned_file.exists():
        return versioned_file

    default_file = base_dir / "Dinasor.txt"
    if default_file.exists():
        return default_file

    for fallback in [
        PROJECT_ROOT / "src" / "prompts" / "Dinasor.txt",
        PROJECT_ROOT / "prompts" / "Dinasor.txt",
        PROJECT_ROOT / "data" / "guidelines" / "Dinasor.txt"
    ]:
        if fallback.exists():
            return fallback

    return versioned_file


class Dinasor:
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
        self.prompt_version = (prompt_version or os.getenv("DINASOR_PROMPT_VERSION") or os.getenv("PROMPT_VERSION") or "v3").lower().strip()
        self.prompt_path = resolve_dinasor_prompt(version=self.prompt_version, custom_path=prompt_path)

    def run(
        self,
        abstract: str,
        uncertain_entities: list[dict] | list | dict | str,
        dynamicGuidelines: dict = None,
        save_output_path: str = None,
        return_prompt: bool = False,
        return_raw: bool = False
    ) -> list | dict | str | tuple:
        """
        Analyzes neural model's uncertain entities, contrastive exemplars, and anchored context
        to generate targeted disambiguation hints.

        Args:
            abstract: Raw text of the article / input abstract
            uncertain_entities: List of dicts, rich payload dict, or string with uncertainty metrics,
                                contrastive exemplars, and syntactic context
            dynamicGuidelines: Optional dynamicGuidelines dictionary
            save_output_path: Optional path to save raw LLM response
            return_prompt: Whether to return filled prompt string
            return_raw: Whether to return raw LLM response string

        Returns:
            The generated new hints (parsed list or dict from LLM response),
            or a tuple (hints, prompt, raw_response) depending on flags.
        """
        if dynamicGuidelines is None:
            dynamicGuidelines = {}

        # Format uncertain_entities
        if isinstance(uncertain_entities, list):
            # Check if elements are rich structured dictionaries
            if uncertain_entities and isinstance(uncertain_entities[0], dict) and ("spanner_evidence" in uncertain_entities[0] or "challenge_type" in uncertain_entities[0] or "contrastive_exemplars" in uncertain_entities[0]):
                uncertain_entities_str = json.dumps(uncertain_entities, indent=2)
            else:
                formatted_entities = []
                for item in uncertain_entities:
                    if isinstance(item, dict):
                        entity = item.get("entity", item.get("span_text", ""))
                        tag = item.get("predicted_tag", item.get("spanner_label", ""))
                        unc = item.get("uncertainity_score", item.get("uncertainty_score", item.get("u_score", "")))
                        nov = item.get("novelty_score", item.get("novelty", ""))
                        ch_type = item.get("challenge_type", "")
                        parts = [f"Entity: '{entity}'"]
                        if ch_type:
                            parts.append(f"Challenge Type: {ch_type}")
                        if tag:
                            parts.append(f"Predicted Tag: {tag}")
                        if unc != "":
                            parts.append(f"Uncertainty: {unc}")
                        if nov != "":
                            parts.append(f"Novelty: {nov}")
                        if "allowed_categories" in item:
                            parts.append(f"Allowed Categories: {item['allowed_categories']}")
                        formatted_entities.append("- " + " | ".join(parts))
                    else:
                        formatted_entities.append(f"- {item}")
                uncertain_entities_str = "\n".join(formatted_entities)
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
            system_prompt="Dinasor",
            max_tokens=self.max_tokens
        )
        raw_response_str = str(response or "")

        # Save prompt and raw response if path is provided
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("DINASOR INPUT PROMPT (FINAL FILLED)\n")
                f.write("=" * 80 + "\n\n")
                f.write(str(prompt or ""))
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("DINASOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(raw_response_str)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF DINASOR RESPONSE\n")
                f.write("=" * 80 + "\n")

        if not response:
            hints = []
            if return_prompt and return_raw:
                return (hints, prompt, raw_response_str)
            elif return_prompt:
                return (hints, prompt)
            elif return_raw:
                return (hints, raw_response_str)
            return hints

        # Return generated hints
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                if "hints" in parsed:
                    hints = parsed["hints"]
                elif "new_hints" in parsed:
                    hints = parsed["new_hints"]
                else:
                    hints = parsed
            else:
                hints = parsed
        except (json.JSONDecodeError, TypeError):
            hints = response or []

        if return_prompt and return_raw:
            return (hints, prompt, raw_response_str)
        elif return_prompt:
            return (hints, prompt)
        elif return_raw:
            return (hints, raw_response_str)
        return hints


# Alias for backward compatibility
ReflectorBERT = Dinasor
