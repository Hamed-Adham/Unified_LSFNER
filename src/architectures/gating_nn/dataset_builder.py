"""
Gating Dataset Builder for preparing training datasets for Gating Neural Networks.
Executes dual parallel LLM calls (Prompt v1 with label vs Prompt v2 blind) and exports tabular CSV & JSONL datasets.
"""

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, OUTPUT_DIR
from src.common.label_mapping import canonicalize_label, to_new_label
from src.common.llm_client import generate_llm_response
from src.architectures.gating_nn.prompt_templates import (
    load_prompt_template,
    format_inlined_abstract,
    parse_llm_json_response
)
from src.architectures.gating_nn.span_features import run_nms, match_gt_label


def classify_outcome(gt_label: str, spanner_label: str, llm_label: str) -> str:
    """Classifies the tripartite outcome between Ground Truth, SpanNER, and LLM."""
    sp_corr = (spanner_label == gt_label)
    llm_corr = (llm_label == gt_label)

    if sp_corr and llm_corr:
        return "both_correct"
    elif sp_corr and not llm_corr:
        return "degradation"
    elif not sp_corr and llm_corr:
        return "rescue"
    else:
        return "both_wrong"


class GatingDatasetBuilder:
    """
    Orchestrates candidate span extraction, dual LLM prompt calls, and dataset exports.
    """
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or (OUTPUT_DIR / "gating_nn"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.output_dir / "per_abstract_json"
        self.reports_dir = self.output_dir / "per_abstract_reports"
        self.json_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def export_datasets(self, records: List[Dict[str, Any]], dataset_prefix: str = "gating_dataset_400train") -> Dict[str, str]:
        """Saves full combined dataset, prompt v1 slice, and prompt v2 slice to CSV and JSONL."""
        df = pd.DataFrame(records)
        paths = {}

        if len(df) == 0:
            return paths

        # 1. Full Combined Dataset
        full_csv = self.output_dir / f"{dataset_prefix}_dual_prompts.csv"
        full_jsonl = self.output_dir / f"{dataset_prefix}_dual_prompts.jsonl"
        df.to_csv(full_csv, index=False)
        with open(full_jsonl, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths["full_csv"] = str(full_csv)
        paths["full_jsonl"] = str(full_jsonl)

        # 2. Slice v1
        cols_v1 = [
            'doc_id', 'span_text', 'start_char', 'end_char', 'gt_label',
            'spanner_label', 'spanner_confidence', 'spanner_top2_prob', 'second_best_label',
            'margin', 'uncertainty', 'uncertainty_method', 'novelty_score', 'p_background_o',
            'llm_v1_label', 'llm_v1_predicted_tag', 'llm_v1_rationale', 'llm_v1_correct',
            'v1_outcome', 'spanner_correct', 'v1_escalation_benefit'
        ]
        df_v1 = df[[c for c in cols_v1 if c in df.columns]].rename(columns={
            'llm_v1_label': 'llm_label',
            'llm_v1_predicted_tag': 'llm_predicted_tag',
            'llm_v1_rationale': 'llm_rationale',
            'llm_v1_correct': 'llm_correct',
            'v1_escalation_benefit': 'escalation_benefit',
            'v1_outcome': 'outcome'
        })
        v1_csv = self.output_dir / f"{dataset_prefix}_v1.csv"
        df_v1.to_csv(v1_csv, index=False)
        paths["v1_csv"] = str(v1_csv)

        # 3. Slice v2
        cols_v2 = [
            'doc_id', 'span_text', 'start_char', 'end_char', 'gt_label',
            'spanner_label', 'spanner_confidence', 'spanner_top2_prob', 'second_best_label',
            'margin', 'uncertainty', 'uncertainty_method', 'novelty_score', 'p_background_o',
            'llm_v2_label', 'llm_v2_rationale', 'llm_v2_correct',
            'v2_outcome', 'spanner_correct', 'v2_escalation_benefit'
        ]
        df_v2 = df[[c for c in cols_v2 if c in df.columns]].rename(columns={
            'llm_v2_label': 'llm_label',
            'llm_v2_rationale': 'llm_rationale',
            'llm_v2_correct': 'llm_correct',
            'v2_escalation_benefit': 'escalation_benefit',
            'v2_outcome': 'outcome'
        })
        v2_csv = self.output_dir / f"{dataset_prefix}_v2.csv"
        df_v2.to_csv(v2_csv, index=False)
        paths["v2_csv"] = str(v2_csv)

        return paths
