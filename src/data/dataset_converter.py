"""
Dataset Converter and BRAT .ann Annotation Parser.
Supports both legacy (20-class) and relabeled (10-class) LSF schemas.
"""

import os
import glob
import re
import json
import random
from typing import List, Dict, Any, Optional, Set

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from src.common.label_mapping import (
    ALL_KNOWN_LABELS,
    CANONICAL_10_LABELS,
    CANONICAL_10_LABEL2ID,
    CANONICAL_10_ID2LABEL,
    LEGACY_LABEL_LIST,
    RELABELED_LABEL_LIST,
    to_new_label,
    to_old_label,
    canonicalize_label,
    map_label,
)

LABEL_LIST = CANONICAL_10_LABELS
LABEL2ID = CANONICAL_10_LABEL2ID


def parse_ann_file(ann_path: str, valid_labels: Optional[Set[str]] = None, target_schema: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse BRAT .ann file to extract character offset spans and entity types.
    Supports both legacy and relabeled schemas.
    
    Args:
        ann_path: Path to the .ann file.
        valid_labels: Optional set/list of allowed labels. If None, accepts all known LSF labels.
        target_schema: Optional target schema ('new' or 'old') to map parsed labels onto.
    """
    entities = []
    if not os.path.exists(ann_path):
        return entities
        
    allowed_labels = set(valid_labels) if valid_labels is not None else ALL_KNOWN_LABELS
        
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
                
            type_offsets = parts[1]
            text = parts[2]
            
            type_parts = type_offsets.split()
            etype = type_parts[0]
            if allowed_labels and etype not in allowed_labels:
                continue
                
            raw_offsets = type_offsets[len(etype):].strip().replace(';', ' ').split()
            if len(raw_offsets) >= 2:
                start_char = int(raw_offsets[0])
                end_char = int(raw_offsets[-1])
                final_label = map_label(etype, target_schema) if target_schema else etype
                entities.append({
                    "start_char": start_char,
                    "end_char": end_char,
                    "label": final_label,
                    "text": text
                })
    return entities


def convert_dataset(data_dir: str, model_path: str, output_json: str, max_len: int = 512):
    """Converts a directory of .txt and .ann files into tokenized span dataset for SpanNER."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for convert_dataset. Install transformers.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    samples = []

    for txt_path in txt_files:
        doc_id = os.path.basename(txt_path).replace('.txt', '')
        ann_path = txt_path.replace('.txt', '.ann')
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        entities = parse_ann_file(ann_path)
        
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True
        )
        
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        target_spans = []
        for ent in entities:
            e_start = ent["start_char"]
            e_end = ent["end_char"]
            canonical = to_new_label(ent["label"])
            if canonical not in LABEL2ID or canonical == "O" or canonical == "Lifestyle_factor":
                continue
            label_id = LABEL2ID[canonical]
            
            start_tok_idx = None
            end_tok_idx = None
            
            for idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start == t_end:
                    continue
                if t_start <= e_start < t_end or (start_tok_idx is None and t_start >= e_start):
                    if start_tok_idx is None:
                        start_tok_idx = idx
                if t_start < e_end <= t_end or (start_tok_idx is not None and t_end <= e_end):
                    end_tok_idx = idx
                    
            if start_tok_idx is not None and end_tok_idx is not None and start_tok_idx <= end_tok_idx:
                target_spans.append({
                    "start": start_tok_idx,
                    "end": end_tok_idx,
                    "label": canonical,
                    "label_id": label_id,
                    "text": ent["text"]
                })
                
        samples.append({
            "doc_id": doc_id,
            "text": text,
            "input_ids": input_ids,
            "tokens": tokens,
            "spans": target_spans
        })
        
    random.seed(42)
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    dataset = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train+n_val],
        "test": samples[n_train+n_val:],
        "label2id": LABEL2ID,
        "id2label": {v: k for k, v in LABEL2ID.items()}
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset converted successfully! Total: {n_total} (Train: {len(dataset['train'])}, Val: {len(dataset['val'])}, Test: {len(dataset['test'])})")
    print(f"Saved to {output_json}")


def convert_split_dataset(train_dir: str, val_dir: str, model_path: str, output_json: str, max_len: int = 512):
    """Converts predefined train/val directory splits into SpanNER dataset JSON."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for convert_split_dataset.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def process_dir(directory):
        txt_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
        samples = []
        for txt_path in txt_files:
            doc_id = os.path.basename(txt_path).replace('.txt', '')
            ann_path = txt_path.replace('.txt', '.ann')
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            entities = parse_ann_file(ann_path)
            encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len, add_special_tokens=True)
            input_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            target_spans = []
            for ent in entities:
                e_start = ent["start_char"]
                e_end = ent["end_char"]
                canonical = to_new_label(ent["label"])
                if canonical not in LABEL2ID or canonical == "O" or canonical == "Lifestyle_factor":
                    continue
                label_id = LABEL2ID[canonical]
                start_tok_idx = None
                end_tok_idx = None
                for idx, (t_start, t_end) in enumerate(offset_mapping):
                    if t_start == t_end: continue
                    if t_start <= e_start < t_end or (start_tok_idx is None and t_start >= e_start):
                        if start_tok_idx is None: start_tok_idx = idx
                    if t_start < e_end <= t_end or (start_tok_idx is not None and t_end <= e_end):
                        end_tok_idx = idx
                if start_tok_idx is not None and end_tok_idx is not None and start_tok_idx <= end_tok_idx:
                    target_spans.append({"start": start_tok_idx, "end": end_tok_idx, "label": canonical, "label_id": label_id, "text": ent["text"]})
            samples.append({"doc_id": doc_id, "text": text, "input_ids": input_ids, "tokens": tokens, "spans": target_spans})
        return samples

    train_samples = process_dir(train_dir)
    val_samples = process_dir(val_dir)
    
    dataset = {
        "train": train_samples,
        "val": val_samples,
        "test": val_samples,
        "label2id": LABEL2ID,
        "id2label": {v: k for k, v in LABEL2ID.items()}
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Split dataset converted! Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Saved to {output_json}")
