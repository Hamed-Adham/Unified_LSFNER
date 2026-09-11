#!/usr/bin/env python3
"""
SpanNER Candidate Extraction and Multi-Signal Caching Script.

Runs the 10-class SpanNER model (best_spanner_160train.pt) across target datasets
and saves comprehensive span-level neural evidence, novelty metrics, and token geometry
to data/cached/spanner_160_10_class/:
  - 400/train (240 abstracts from data/new/400Abstracts/train)
  - 400/val   (80 abstracts from data/new/400Abstracts/val)
  - 200/train (160 abstracts: 120 train + 40 val from data/new/200Abstracts)

Retention Rule:
  Stores all candidate spans whose top-1 predicted class is NOT background 'O'
  (i.e., predicted as one of the 9 canonical LSF categories).

Packaging:
  Dual format: saves individual <doc_id>.json files AND an aggregated all_candidates.json
  inside each split folder.
"""

import os
import sys
import glob
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
from src.architectures.hybrid_linkner.novelty_detector import build_lof_novelty_detector
from src.data.dataset_converter import parse_ann_file
from src.common.audit_logger import match_gt_label
from src.common.label_mapping import CANONICAL_10_ID2LABEL, canonicalize_label


def parse_args():
    parser = argparse.ArgumentParser(description="Cache SpanNER candidates and multi-signal evidence.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/cached/spanner_160_10_class"),
        help="Base output directory for cached candidate files."
    )
    parser.add_argument(
        "--spanner_checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models/SpanNER_LSF/best_spanner_160train.pt"),
        help="Path to trained 10-class SpanNER model checkpoint."
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=str(PROJECT_ROOT / "models/NER_Model/trained_NER_model"),
        help="Path to trained NER encoder / tokenizer."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for SpanNER neural forward passes."
    )
    parser.add_argument(
        "--max_span_width",
        type=int,
        default=6,
        help="Maximum span width in tokens (must match SpanNER training config: 6)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device ('mps', 'cuda', 'cpu'). Auto-detected if empty."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["400/val", "400/train", "200/train"],
        help="Target splits to cache (e.g. '400/val', '400/train', '200/train')."
    )
    parser.add_argument(
        "--recompute_novelty_only",
        action="store_true",
        help="Fast in-place update: recomputes novelty scores using active vector embedder (e.g. Qwen-8B via 9Router) on existing cached spans without re-running SpanNER inference."
    )
    return parser.parse_args()


def get_split_sources() -> Dict[str, List[str]]:
    """Defines the source directory mapping for each target split."""
    return {
        "200/train": [
            str(PROJECT_ROOT / "data/new/200Abstracts/train"),
            str(PROJECT_ROOT / "data/new/200Abstracts/val"),
        ],
        "400/train": [
            str(PROJECT_ROOT / "data/new/400Abstracts/no_disease/train"),
        ],
        "400/val": [
            str(PROJECT_ROOT / "data/new/400Abstracts/no_disease/val"),
        ],
    }


def load_abstract_records(folder_paths: List[str]) -> List[Dict[str, Any]]:
    """Loads text and ground-truth entities for all paired abstracts in folder_paths."""
    records = []
    seen_ids = set()
    for fld in folder_paths:
        ann_files = sorted(glob.glob(os.path.join(fld, "*.ann")))
        for ann_p in ann_files:
            doc_id = Path(ann_p).stem
            if doc_id in seen_ids:
                continue
            txt_p = str(Path(ann_p).with_suffix(".txt"))
            if not os.path.exists(txt_p):
                continue
            with open(txt_p, "r", encoding="utf-8") as f:
                text = f.read()
            gt_ents = parse_ann_file(ann_p)
            seen_ids.add(doc_id)
            records.append({
                "doc_id": doc_id,
                "txt_path": txt_p,
                "ann_path": ann_p,
                "text": text,
                "gt_ents": gt_ents,
            })
    return records


def recompute_novelty_for_splits(args, novelty_detector):
    """Fast in-place recalculation of novelty metrics across pre-extracted cached candidates."""
    print("\n" + "=" * 88)
    print("⚡ FAST IN-PLACE NOVELTY RECALCULATION")
    print(f"  • Embedder Backend : {getattr(novelty_detector.embedder, 'active_backend_name', type(novelty_detector.embedder))}")
    print(f"  • Exemplars Fitted : {len(novelty_detector.records)} (dim: {novelty_detector.matrix.shape[1]})")
    print(f"  • Target Splits    : {args.splits}")
    print("=" * 88)

    base_out = Path(args.output_dir)
    total_abstracts = 0
    total_spans = 0

    for split_key in args.splits:
        split_dir = base_out / split_key
        if not split_dir.exists():
            print(f"⚠️ Warning: Split directory not found: {split_dir}, skipping.")
            continue

        doc_files = sorted([p for p in split_dir.glob("*.json") if p.name != "all_candidates.json"])
        print(f"\n📂 Recalculating split [{split_key}]: {len(doc_files)} abstract files found...")

        split_aggregated = {}
        split_span_count = 0

        for idx, doc_p in enumerate(doc_files, 1):
            with open(doc_p, "r", encoding="utf-8") as f:
                doc_entry = json.load(f)

            spans = doc_entry.get("spans", [])
            if spans:
                cand_texts = [s["span_text"] for s in spans]
                cand_vecs = novelty_detector.embed(cand_texts)
                knn_scores = novelty_detector.score_knn_distance(cand_vecs)
                lof_scores = novelty_detector.score_lof(cand_vecs)
                nov_scores = novelty_detector.novelty(cand_vecs)

                for c_idx, s in enumerate(spans):
                    nov_c = round(float(nov_scores[c_idx]), 4)
                    lof_c = round(float(lof_scores[c_idx]), 4)
                    knn_c = round(float(knn_scores[c_idx]), 4)

                    s.setdefault("spanner_evidence", {})["novelty_score"] = nov_c
                    s.setdefault("novelty_breakdown", {})["novelty_combined"] = nov_c
                    s["novelty_breakdown"]["novelty_lof"] = lof_c
                    s["novelty_breakdown"]["novelty_knn_dist"] = knn_c

            with open(doc_p, "w", encoding="utf-8") as f:
                json.dump(doc_entry, f, indent=2, ensure_ascii=False)

            doc_id = doc_entry.get("doc_id", doc_p.stem)
            split_aggregated[doc_id] = doc_entry
            split_span_count += len(spans)

            if idx % 40 == 0 or idx == len(doc_files):
                print(f"  Processed {idx}/{len(doc_files)} abstracts ({split_span_count} spans updated)...")

        agg_path = split_dir / "all_candidates.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(split_aggregated, f, indent=2, ensure_ascii=False)

        print(f"✅ Completed [{split_key}]: {len(doc_files)} abstracts, {split_span_count} candidate spans rewritten.")
        total_abstracts += len(doc_files)
        total_spans += split_span_count

    print("\n" + "=" * 88)
    print(f"🎉 ALL REQUESTED SPLITS SUCCESSFULLY UPDATED!")
    print(f"  • Total Abstracts Updated: {total_abstracts}")
    print(f"  • Total Candidate Spans Updated: {total_spans}")
    print(f"  • Target Directory: {args.output_dir}")
    print("=" * 88)


def run_caching():
    args = parse_args()

    # 1. Resolve compute device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Fast In-Place Recalculation Path
    if args.recompute_novelty_only:
        print("🔍 Initializing LOF Novelty Detector (Zero Leakage on 160 train/val reference)...")
        novelty_detector = build_lof_novelty_detector(k=10, alpha=0.5, device=str(device))
        print(f"✅ Novelty Detector fitted with {len(novelty_detector.matrix)} exemplars (dim: {novelty_detector.matrix.shape[1]}).")
        recompute_novelty_for_splits(args, novelty_detector)
        return

    print("=" * 88)
    print("🚀 SPANNER CANDIDATE EXTRACTION & MULTI-SIGNAL CACHING")
    print("=" * 88)
    print(f"  • Model Checkpoint: {args.spanner_checkpoint}")
    print(f"  • Tokenizer/Encoder: {args.encoder_path}")
    print(f"  • Compute Device   : {device}")
    print(f"  • Batch Size       : {args.batch_size}")
    print(f"  • Max Span Width   : {args.max_span_width}")
    print(f"  • Output Base Dir  : {args.output_dir}")
    print("=" * 88)

    # 2. Load Tokenizer & SpanNER Model
    print("⚡ Loading SpanNER model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)
    spanner_model = SpanNERModel(
        encoder_path=args.encoder_path,
        num_classes=10,
        max_span_width=args.max_span_width
    )
    checkpoint_state = torch.load(args.spanner_checkpoint, map_location=device)
    spanner_model.load_state_dict(checkpoint_state)
    spanner_model.to(device)
    spanner_model.eval()
    print("✅ SpanNER model successfully loaded.")

    # 3. Initialize & Fit LOF Novelty Detector on 160 fine-tuning abstracts
    print("🔍 Initializing LOF Novelty Detector (Zero Leakage on 160 train/val reference)...")
    novelty_detector = build_lof_novelty_detector(k=10, alpha=0.5, device=str(device))
    print(f"✅ Novelty Detector fitted with {len(novelty_detector.matrix)} exemplars (dim: {novelty_detector.matrix.shape[1]}).")

    all_split_sources = get_split_sources()
    split_sources = {k: v for k, v in all_split_sources.items() if k in args.splits}
    total_processed_abstracts = 0
    total_extracted_candidates = 0

    # 4. Process Each Split
    for split_key, folders in split_sources.items():
        print("\n" + "-" * 88)
        print(f"📂 Processing Split: [{split_key}] from sources: {folders}")
        print("-" * 88)

        split_out_dir = Path(args.output_dir) / split_key
        split_out_dir.mkdir(parents=True, exist_ok=True)

        abstract_records = load_abstract_records(folders)
        print(f"  Found {len(abstract_records)} paired abstracts (.txt + .ann).")

        split_aggregated = {}
        split_span_count = 0

        # Process in batches
        for b_start in range(0, len(abstract_records), args.batch_size):
            b_records = abstract_records[b_start : b_start + args.batch_size]
            b_texts = [r["text"] for r in b_records]

            # Batch Tokenization
            encoding = tokenizer(
                b_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            offsets_all = encoding["offset_mapping"].cpu().numpy()

            with torch.no_grad():
                out = spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                batch_logits = out["logits"]  # [B, Num_Spans, 10]
                candidate_spans = out["candidate_spans"]  # list of (s_tok, e_tok, len)

            batch_probs = F.softmax(batch_logits, dim=-1)
            b_sorted_probs, b_sorted_indices = torch.sort(batch_probs, dim=-1, descending=True)

            # Collect candidate spans per document
            for item_idx, record in enumerate(b_records):
                doc_id = record["doc_id"]
                raw_text = record["text"]
                gt_ents = record["gt_ents"]
                offsets = offsets_all[item_idx]

                doc_candidates = []
                span_counter = 0

                for s_idx, (s_tok, e_tok, length) in enumerate(candidate_spans):
                    top1_id = b_sorted_indices[item_idx, s_idx, 0].item()

                    # RETENTION RULE: top-1 must not be background 'O' (0)
                    if top1_id == 0:
                        continue

                    top1_label = CANONICAL_10_ID2LABEL[top1_id]
                    st_c = int(offsets[s_tok][0])
                    end_c = int(offsets[e_tok][1])
                    span_txt = raw_text[st_c:end_c].strip()
                    if not span_txt:
                        continue

                    top1_p = float(b_sorted_probs[item_idx, s_idx, 0].item())
                    top2_id = b_sorted_indices[item_idx, s_idx, 1].item()
                    top2_label = CANONICAL_10_ID2LABEL[top2_id]
                    top2_p = float(b_sorted_probs[item_idx, s_idx, 1].item())
                    margin = float(top1_p - top2_p)
                    p_o = float(batch_probs[item_idx, s_idx, 0].item())

                    # Uncertainty: Normalized Prediction Entropy in [0, 1]
                    p_vec = batch_probs[item_idx, s_idx]
                    entropy_norm = float((-torch.sum(p_vec * torch.log(p_vec + 1e-9)) / math.log(10)).item())
                    least_conf = float(1.0 - top1_p)
                    margin_unc = float(1.0 - margin)

                    # Ground truth matching (with canonical new label)
                    gt_raw = match_gt_label(span_txt, st_c, end_c, gt_ents)
                    gt_clean = canonicalize_label(gt_raw, use_new_labels=True)
                    is_gt_entity = (gt_clean != "O")

                    # Probabilities and logits across all 10 canonical classes
                    prob_map = {
                        CANONICAL_10_ID2LABEL[c_id]: round(float(batch_probs[item_idx, s_idx, c_id].item()), 4)
                        for c_id in range(10)
                    }
                    logit_map = {
                        CANONICAL_10_ID2LABEL[c_id]: round(float(batch_logits[item_idx, s_idx, c_id].item()), 4)
                        for c_id in range(10)
                    }

                    span_record = {
                        "span_id": f"{doc_id}_s{span_counter:03d}",
                        "abstract_id": doc_id,
                        "span_text": span_txt,
                        "boundaries": {
                            "start_char": st_c,
                            "end_char": end_c,
                            "token_start": int(s_tok),
                            "token_end": int(e_tok),
                            "token_length": int(length),
                        },
                        "ground_truth": {
                            "gt_label": gt_clean,
                            "is_entity": is_gt_entity,
                        },
                        "spanner_evidence": {
                            "predicted_label": top1_label,
                            "second_best_label": top2_label,
                            "uncertainty": round(entropy_norm, 4),
                            "novelty_score": 0.0,  # populated below via vector novelty detector
                            "margin": round(margin, 4),
                            "top1_prob": round(top1_p, 4),
                            "top2_prob": round(top2_p, 4),
                            "p_background_o": round(p_o, 4),
                        },
                        "novelty_breakdown": {
                            "novelty_combined": 0.0,
                            "novelty_lof": 0.0,
                            "novelty_knn_dist": 0.0,
                        },
                        "uncertainty_metrics": {
                            "entropy_pe_norm": round(entropy_norm, 4),
                            "least_confidence": round(least_conf, 4),
                            "margin_uncertainty": round(margin_unc, 4),
                        },
                        "probabilities": prob_map,
                        "logits": logit_map,
                    }
                    doc_candidates.append(span_record)
                    span_counter += 1

                # Batch novelty scoring for all candidate spans in this document
                if doc_candidates:
                    cand_texts = [s["span_text"] for s in doc_candidates]
                    cand_vecs = novelty_detector.embed(cand_texts)
                    knn_scores = novelty_detector.score_knn_distance(cand_vecs)
                    lof_scores = novelty_detector.score_lof(cand_vecs)
                    nov_scores = novelty_detector.novelty(cand_vecs)

                    for c_idx, s in enumerate(doc_candidates):
                        nov_c = round(float(nov_scores[c_idx]), 4)
                        lof_c = round(float(lof_scores[c_idx]), 4)
                        knn_c = round(float(knn_scores[c_idx]), 4)

                        s["spanner_evidence"]["novelty_score"] = nov_c
                        s["novelty_breakdown"]["novelty_combined"] = nov_c
                        s["novelty_breakdown"]["novelty_lof"] = lof_c
                        s["novelty_breakdown"]["novelty_knn_dist"] = knn_c

                doc_entry = {
                    "doc_id": doc_id,
                    "dataset_split": split_key,
                    "text": raw_text,
                    "total_candidates": len(doc_candidates),
                    "spans": doc_candidates,
                }

                # 1. Save individual per-abstract JSON file
                doc_json_path = split_out_dir / f"{doc_id}.json"
                with open(doc_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_entry, f, indent=2, ensure_ascii=False)

                split_aggregated[doc_id] = doc_entry
                split_span_count += len(doc_candidates)

            print(f"  Processed {min(b_start + args.batch_size, len(abstract_records))}/{len(abstract_records)} abstracts ({split_span_count} candidate spans so far)...")

        # 2. Save aggregated all_candidates.json for the split
        manifest_path = split_out_dir / "all_candidates.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_aggregated, f, indent=2, ensure_ascii=False)

        print(f"✅ Completed [{split_key}]: {len(split_aggregated)} abstracts, {split_span_count} candidate spans saved to {split_out_dir}.")
        total_processed_abstracts += len(split_aggregated)
        total_extracted_candidates += split_span_count

    print("\n" + "=" * 88)
    print("🎉 ALL SPLITS SUCCESSFULLY CACHED!")
    print(f"  • Total Abstracts Processed: {total_processed_abstracts}")
    print(f"  • Total Candidate Spans Extracted: {total_extracted_candidates}")
    print(f"  • Target Directory: {args.output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    run_caching()
