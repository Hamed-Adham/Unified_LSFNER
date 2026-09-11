#!/usr/bin/env python3
"""
Convert the 160-abstract SpanNER dataset (10-class) into two dedicated datasets
for two-stage architecture experimentation:

Stage 1 (Binary Proposal / Detection):
  - Label schema: {"Non-LSF": 0, "LSF": 1}
  - All true entity mentions mapped to class 1 ("LSF").
  - Background non-entity spans remain class 0 ("Non-LSF").
  - Output: data/processed/spanner_dataset_160_stage1_binary.json

Stage 2 (9-Class LSF Categorization):
  - Label schema: 9 LSF categories indexed 0..8 (no 'O' class).
  - Only positive entity spans remapped from 1..9 to 0..8.
  - Output: data/processed/spanner_dataset_160_stage2_9class.json
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

STAGE2_9CLASS_LABEL2ID = {
    "Personal_care_products_and_cosmetic_procedures": 0,
    "Substance_use": 1,
    "Environmental_exposures": 2,
    "Mental_health_practices": 3,
    "Non_physical_leisure_time_activities": 4,
    "Nutrition": 5,
    "Physical_activities": 6,
    "Sleep": 7,
    "Socioeconomic_factors": 8
}

STAGE2_9CLASS_ID2LABEL = {v: k for k, v in STAGE2_9CLASS_LABEL2ID.items()}

STAGE1_BINARY_LABEL2ID = {
    "Non-LSF": 0,
    "LSF": 1
}

STAGE1_BINARY_ID2LABEL = {
    0: "Non-LSF",
    1: "LSF"
}


def convert_stage1_binary(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dataset to binary proposal format (0: Non-LSF, 1: LSF)."""
    converted = {
        "label2id": STAGE1_BINARY_LABEL2ID,
        "id2label": STAGE1_BINARY_ID2LABEL,
        "train": [],
        "test": []
    }
    
    total_spans = 0
    for split in ["train", "test"]:
        for sample in raw_data.get(split, []):
            new_spans = []
            for s in sample.get("spans", []):
                new_spans.append({
                    "start": s["start"],
                    "end": s["end"],
                    "label": "LSF",
                    "label_id": 1,
                    "text": s["text"],
                    "original_label": s.get("label", "")
                })
                total_spans += 1
            converted[split].append({
                "doc_id": sample["doc_id"],
                "text": sample["text"],
                "input_ids": sample["input_ids"],
                "tokens": sample["tokens"],
                "spans": new_spans
            })
            
    print(f"[Stage 1 Binary] Converted {len(converted['train'])} train, {len(converted['test'])} test abstracts.")
    print(f"[Stage 1 Binary] Total positive LSF entity spans: {total_spans}")
    return converted


def convert_stage2_9class(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dataset to 9-class entity categorization format (0..8)."""
    converted = {
        "label2id": STAGE2_9CLASS_LABEL2ID,
        "id2label": STAGE2_9CLASS_ID2LABEL,
        "train": [],
        "test": []
    }
    
    total_spans = 0
    class_counts = {k: 0 for k in STAGE2_9CLASS_LABEL2ID}
    
    for split in ["train", "test"]:
        for sample in raw_data.get(split, []):
            new_spans = []
            for s in sample.get("spans", []):
                orig_label = s["label"]
                if orig_label not in STAGE2_9CLASS_LABEL2ID:
                    print(f"Warning: Skipping unexpected label '{orig_label}' in doc {sample['doc_id']}")
                    continue
                new_label_id = STAGE2_9CLASS_LABEL2ID[orig_label]
                new_spans.append({
                    "start": s["start"],
                    "end": s["end"],
                    "label": orig_label,
                    "label_id": new_label_id,
                    "text": s["text"]
                })
                class_counts[orig_label] += 1
                total_spans += 1
                
            converted[split].append({
                "doc_id": sample["doc_id"],
                "text": sample["text"],
                "input_ids": sample["input_ids"],
                "tokens": sample["tokens"],
                "spans": new_spans
            })
            
    print(f"\n[Stage 2 9-Class] Converted {len(converted['train'])} train, {len(converted['test'])} test abstracts.")
    print(f"[Stage 2 9-Class] Total entity spans: {total_spans}")
    print("[Stage 2 9-Class] Class distribution:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cls_name:<48}: {count}")
    return converted


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_file = project_root / "data" / "processed" / "spanner_dataset_160_10class.json"
    
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")
        
    print(f"Loading source dataset: {src_file}")
    with open(src_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    stage1_file = project_root / "data" / "processed" / "spanner_dataset_160_stage1_binary.json"
    stage2_file = project_root / "data" / "processed" / "spanner_dataset_160_stage2_9class.json"
    
    # 1. Stage 1 Binary
    stage1_data = convert_stage1_binary(raw_data)
    with open(stage1_file, "w", encoding="utf-8") as f:
        json.dump(stage1_data, f, indent=2)
    print(f"Saved Stage 1 Binary dataset to: {stage1_file}")
    
    # 2. Stage 2 9-Class
    stage2_data = convert_stage2_9class(raw_data)
    with open(stage2_file, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, indent=2)
    print(f"Saved Stage 2 9-Class dataset to: {stage2_file}")


if __name__ == "__main__":
    main()
