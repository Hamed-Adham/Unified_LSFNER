#!/usr/bin/env python3
"""
Benchmark Evaluation Script:
Evaluates and compares:
  1. Baseline 10-Class SpanNER (best_spanner_160train.pt)
  2. Joint Two-Stage SpanNER (two_stage_spanner_joint.pt)
  3. Sequential Two-Stage SpanNER (two_stage_spanner_sequential.pt)

Evaluated on the 80-abstract balanced no-disease validation set:
  data/new/400Abstracts/no_disease/balanced/val
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
from src.architectures.hybrid_linkner.two_stage_spanner_model import (
    TwoStageSpanNERModel,
    STAGE2_9CLASS_ID2LABEL
)
from src.common.label_mapping import (
    CANONICAL_10_ID2LABEL,
    canonicalize_label
)
from src.data.dataset_converter import parse_ann_file


def get_device(requested_device: str = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_val_abstracts(val_dir: str) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    """Loads paired (.txt, .ann) files from the balanced validation set."""
    ann_files = sorted(glob.glob(os.path.join(val_dir, "*.ann")))
    items = []
    for ann_path in ann_files:
        txt_path = ann_path.replace(".ann", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            gt_ents = parse_ann_file(ann_path)
            # Canonicalize gold labels
            canonical_gt = []
            for g in gt_ents:
                c_lbl = canonicalize_label(g["label"])
                if c_lbl not in ("O", "Non-LSF", "Lifestyle_factor"):
                    canonical_gt.append({
                        "start_char": g["start_char"],
                        "end_char": g["end_char"],
                        "label": c_lbl,
                        "text": g["text"]
                    })
            items.append((txt_path, text, canonical_gt))
    return items


def predict_baseline_10class(
    model: SpanNERModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device
) -> List[Dict[str, Any]]:
    """Predict candidate spans using the 10-class baseline model."""
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().tolist()
    
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)
        logits = out["logits"][0]  # [Num_Spans, 10]
        candidate_spans = out["candidate_spans"]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        
    extracted = []
    for s_idx, label_id in enumerate(preds):
        if label_id > 0:  # Class > 0 is an LSF entity
            start_tok, end_tok, length = candidate_spans[s_idx]
            if start_tok < len(offset_mapping) and end_tok < len(offset_mapping):
                start_char = offset_mapping[start_tok][0]
                end_char = offset_mapping[end_tok][1]
                if start_char < end_char:
                    lbl_name = CANONICAL_10_ID2LABEL.get(label_id, "Unknown")
                    extracted.append({
                        "start_char": start_char,
                        "end_char": end_char,
                        "label": lbl_name,
                        "text": text[start_char:end_char],
                        "prob": probs[s_idx, label_id].item()
                    })
    return extracted


def predict_two_stage(
    model: TwoStageSpanNERModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    tau: float = 0.5
) -> List[Dict[str, Any]]:
    """Predict candidate spans using a Two-Stage model (Head 1 binary + Head 2 9-class)."""
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().tolist()
    
    with torch.no_grad():
        doc_preds = model.predict(input_ids, attention_mask=attention_mask, tau=tau)[0]
        
    extracted = []
    for p in doc_preds:
        start_tok = p["start"]
        end_tok = p["end"]
        if start_tok < len(offset_mapping) and end_tok < len(offset_mapping):
            start_char = offset_mapping[start_tok][0]
            end_char = offset_mapping[end_tok][1]
            if start_char < end_char:
                extracted.append({
                    "start_char": start_char,
                    "end_char": end_char,
                    "label": p["label"],
                    "text": text[start_char:end_char],
                    "p_lsf": p["p_lsf"],
                    "prob": p["cat_prob"]
                })
    return extracted


def evaluate_predictions(
    all_preds: List[List[Dict[str, Any]]],
    all_gts: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Computes Proposal Recall, Exact PRF, and Per-Class breakdown."""
    total_gt = sum(len(gt) for gt in all_gts)
    total_pred = sum(len(p) for p in all_preds)
    
    covered_gt_boundaries = 0
    exact_tp = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_gt = defaultdict(int)
    
    for preds, gts in zip(all_preds, all_gts):
        matched_gt_exact = set()
        matched_gt_boundary = set()
        
        for g_idx, g in enumerate(gts):
            per_class_gt[g["label"]] += 1
            for p in preds:
                if p["start_char"] == g["start_char"] and p["end_char"] == g["end_char"]:
                    matched_gt_boundary.add(g_idx)
                    if p["label"] == g["label"]:
                        matched_gt_exact.add(g_idx)
                        break
                        
        covered_gt_boundaries += len(matched_gt_boundary)
        exact_tp += len(matched_gt_exact)
        
        # Per-class counts
        pred_matched = set()
        for p_idx, p in enumerate(preds):
            matched = False
            for g in gts:
                if p["start_char"] == g["start_char"] and p["end_char"] == g["end_char"] and p["label"] == g["label"]:
                    matched = True
                    break
            if matched:
                per_class_tp[p["label"]] += 1
                pred_matched.add(p_idx)
            else:
                per_class_fp[p["label"]] += 1
                
        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt_exact:
                per_class_fn[g["label"]] += 1
                
    exact_fp = total_pred - exact_tp
    exact_fn = total_gt - exact_tp
    
    # Proposal Recall: Fraction of GT entities whose exact boundaries were proposed
    proposal_recall = (covered_gt_boundaries / total_gt * 100) if total_gt > 0 else 0.0
    proposal_precision = (covered_gt_boundaries / total_pred * 100) if total_pred > 0 else 0.0
    
    micro_p = (exact_tp / (exact_tp + exact_fp) * 100) if (exact_tp + exact_fp) > 0 else 0.0
    micro_r = (exact_tp / (exact_tp + exact_fn) * 100) if (exact_tp + exact_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    
    # Macro F1
    per_class_metrics = {}
    f1_list = []
    for c in sorted(list(set(STAGE2_9CLASS_ID2LABEL.values()))):
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        c_p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        c_r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        per_class_metrics[c] = {
            "P": c_p,
            "R": c_r,
            "F1": c_f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "GT": per_class_gt[c]
        }
        f1_list.append(c_f1)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    
    return {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "covered_gt_boundaries": covered_gt_boundaries,
        "proposal_recall": proposal_recall,
        "proposal_precision": proposal_precision,
        "exact_tp": exact_tp,
        "exact_fp": exact_fp,
        "exact_fn": exact_fn,
        "exact_micro_p": micro_p,
        "exact_micro_r": micro_r,
        "exact_micro_f1": micro_f1,
        "exact_macro_f1": macro_f1,
        "per_class": per_class_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SpanNER Models on Balanced Val Set")
    parser.add_argument("--tau", type=float, default=0.5, help="Two-stage proposal threshold")
    parser.add_argument("--device", type=str, default="", help="Device: cuda, mps, cpu")
    parser.add_argument("--val_dir", type=str,
                        default="data/new/400Abstracts/no_disease/balanced/val",
                        help="Path to balanced val set directory")
    parser.add_argument("--output_dir", type=str,
                        default="output/two_stage_benchmarks",
                        help="Directory to save benchmark reports")
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using compute device: {device}")
    
    val_dir = os.path.join(PROJECT_ROOT, args.val_dir)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading balanced validation abstracts from: {val_dir}")
    val_items = load_val_abstracts(val_dir)
    print(f"Successfully loaded {len(val_items)} abstracts with annotations.")
    
    all_gts = [item[2] for item in val_items]
    total_gold = sum(len(g) for g in all_gts)
    print(f"Total Gold LSF Entities: {total_gold}")
    
    encoder_path = os.path.join(PROJECT_ROOT, "models/NER_Model/trained_NER_model")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    
    results = {}
    
    # --------------------------------------------------------------------------
    # 1. Evaluate Baseline 10-Class Model
    # --------------------------------------------------------------------------
    baseline_ckpt = os.path.join(PROJECT_ROOT, "models/SpanNER_LSF/best_spanner_160train.pt")
    if os.path.exists(baseline_ckpt):
        print("\n" + "=" * 75)
        print("🔍 EVALUATING BASELINE: 10-Class SpanNER (best_spanner_160train.pt)")
        print("=" * 75)
        base_model = SpanNERModel(encoder_path=encoder_path, num_classes=10, max_span_width=6)
        base_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
        base_model.to(device)
        base_model.eval()
        
        base_preds = [predict_baseline_10class(base_model, tokenizer, item[1], device) for item in val_items]
        results["Baseline_10Class"] = evaluate_predictions(base_preds, all_gts)
        del base_model
        
    # --------------------------------------------------------------------------
    # 2. Evaluate Joint Two-Stage Model
    # --------------------------------------------------------------------------
    joint_ckpt = os.path.join(PROJECT_ROOT, "models/two_stage_spanner/two_stage_spanner_joint.pt")
    if os.path.exists(joint_ckpt):
        print("\n" + "=" * 75)
        print(f"🔍 EVALUATING TWO-STAGE: Joint Multi-Task (tau = {args.tau:.2f})")
        print("=" * 75)
        joint_model = TwoStageSpanNERModel(encoder_path=encoder_path, max_span_width=6)
        joint_model.load_state_dict(torch.load(joint_ckpt, map_location=device))
        joint_model.to(device)
        joint_model.eval()
        
        joint_preds = [predict_two_stage(joint_model, tokenizer, item[1], device, tau=args.tau) for item in val_items]
        results["TwoStage_Joint"] = evaluate_predictions(joint_preds, all_gts)
        del joint_model
        
    # --------------------------------------------------------------------------
    # 3. Evaluate Sequential Two-Stage Model
    # --------------------------------------------------------------------------
    seq_ckpt = os.path.join(PROJECT_ROOT, "models/two_stage_spanner/two_stage_spanner_sequential.pt")
    if os.path.exists(seq_ckpt):
        print("\n" + "=" * 75)
        print(f"🔍 EVALUATING TWO-STAGE: Sequential Two-Phase (tau = {args.tau:.2f})")
        print("=" * 75)
        seq_model = TwoStageSpanNERModel(encoder_path=encoder_path, max_span_width=6)
        seq_model.load_state_dict(torch.load(seq_ckpt, map_location=device))
        seq_model.to(device)
        seq_model.eval()
        
        seq_preds = [predict_two_stage(seq_model, tokenizer, item[1], device, tau=args.tau) for item in val_items]
        results["TwoStage_Sequential"] = evaluate_predictions(seq_preds, all_gts)
        del seq_model

    # --------------------------------------------------------------------------
    # Summary Output Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("📊 BENCHMARK SUMMARY: 80 ABSTRACT BALANCED NO-DISEASE VAL SET")
    print("=" * 90)
    print(f"{'Model Configuration':<28} | {'Prop Recall':<12} | {'Exact P':<10} | {'Exact R':<10} | {'Exact F1':<10} | {'Macro F1':<10}")
    print("-" * 90)
    for model_name, m in results.items():
        print(f"{model_name:<28} | {m['proposal_recall']:>10.2f}% | {m['exact_micro_p']:>8.2f}% | {m['exact_micro_r']:>8.2f}% | {m['exact_micro_f1']:>8.2f}% | {m['exact_macro_f1']:>8.2f}%")
    print("=" * 90)
    
    # Save Report JSON
    json_path = os.path.join(output_dir, "benchmark_balanced_val_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Metrics saved to: {json_path}")
    
    # Save Markdown Report
    md_path = os.path.join(output_dir, "benchmark_balanced_val_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Report: Baseline vs Two-Stage SpanNER\n\n")
        f.write(f"**Dataset**: Balanced No-Disease Validation Set (`80 Abstracts`, `{total_gold} Gold Entities`)\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Model Configuration | Proposal Recall | Exact Precision | Exact Recall | Exact F1 | Macro F1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for model_name, m in results.items():
            f.write(f"| **{model_name}** | {m['proposal_recall']:.2f}% | {m['exact_micro_p']:.2f}% | {m['exact_micro_r']:.2f}% | **{m['exact_micro_f1']:.2f}%** | {m['exact_macro_f1']:.2f}% |\n")
            
        f.write("\n## Per-Class Exact F1 Comparison\n\n")
        f.write("| Category | Gold Count | Baseline 10-Class F1 | Joint Two-Stage F1 | Sequential Two-Stage F1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        categories = sorted(list(STAGE2_9CLASS_ID2LABEL.values()))
        for c in categories:
            b_f1 = results.get("Baseline_10Class", {}).get("per_class", {}).get(c, {}).get("F1", 0.0)
            j_f1 = results.get("TwoStage_Joint", {}).get("per_class", {}).get(c, {}).get("F1", 0.0)
            s_f1 = results.get("TwoStage_Sequential", {}).get("per_class", {}).get(c, {}).get("F1", 0.0)
            gt_c = results.get("Baseline_10Class", {}).get("per_class", {}).get(c, {}).get("GT", 0)
            f.write(f"| {c} | {gt_c} | {b_f1:.2f}% | {j_f1:.2f}% | {s_f1:.2f}% |\n")
            
    print(f"💾 Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
