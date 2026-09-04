"""
Span Feature Extraction Utilities for Gating Neural Network Dataset Preparation.
Extracts candidate entity spans from SpanNER with Monte Carlo Dropout uncertainty and LOF novelty scores.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

from src.common.config import SPANNER_160_MODEL_PATH
from src.common.label_mapping import canonicalize_label, to_new_label, ALL_KNOWN_LABELS
from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
from src.architectures.hybrid_linkner.novelty_detector import LOFNoveltyDetector, get_default_embedder


def run_nms(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Longest-Span Greedy Non-Maximum Suppression."""
    sorted_spans = sorted(spans, key=lambda s: (s['end_char'] - s['start_char'], s['confidence']), reverse=True)
    kept = []
    for s in sorted_spans:
        overlap = False
        for k in kept:
            if not (s['end_char'] <= k['start_char'] or s['start_char'] >= k['end_char']):
                overlap = True
                break
        if not overlap:
            kept.append(s)
    return sorted(kept, key=lambda s: s['start_char'])


def match_gt_label(span_text: str, start_char: int, end_char: int, gt_ents: List[Dict[str, Any]]) -> str:
    """Matches candidate span coordinates against Ground Truth annotations."""
    st = span_text.strip().lower()
    for g in gt_ents:
        gt_text = g.get('text', '').strip().lower()
        if (g['start_char'] == start_char and g['end_char'] == end_char) or (st == gt_text):
            return canonicalize_label(g['label'], use_new_labels=True)
        if not (end_char <= g['start_char'] or start_char >= g['end_char']):
            if st in gt_text or gt_text in st:
                return canonicalize_label(g['label'], use_new_labels=True)
    return 'O'
