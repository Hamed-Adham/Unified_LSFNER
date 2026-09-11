#!/usr/bin/env python3
"""
Training Script for Two-Stage SpanNER Model:
Supports both Joint Multi-Task Training and Sequential Two-Phase Training.
Frozen RoBERTa backbone ensures fast convergence and stability.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.hybrid_linkner.two_stage_spanner_model import (
    TwoStageSpanNERModel,
    STAGE1_BINARY_LABEL2ID,
    STAGE2_9CLASS_LABEL2ID
)


class SpanNERDataset(Dataset):
    def __init__(self, binary_data_path: str, category_data_path: str, split: str = "train"):
        with open(binary_data_path, "r", encoding="utf-8") as f:
            bin_data = json.load(f)
        with open(category_data_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
            
        self.bin_samples = bin_data.get(split, [])
        self.cat_samples = cat_data.get(split, [])
        
        assert len(self.bin_samples) == len(self.cat_samples), (
            f"Mismatch between binary ({len(self.bin_samples)}) and category ({len(self.cat_samples)}) samples in {split}!"
        )

    def __len__(self):
        return len(self.bin_samples)

    def __getitem__(self, idx):
        bin_s = self.bin_samples[idx]
        cat_s = self.cat_samples[idx]
        return {
            "doc_id": bin_s["doc_id"],
            "input_ids": bin_s["input_ids"],
            "binary_spans": bin_s["spans"],
            "category_spans": cat_s["spans"]
        }


def pad_collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_binary_spans = []
    batch_category_spans = []
    batch_doc_ids = []
    
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        padded_ids = ids + [1] * pad_len  # 1 is RoBERTa pad token id
        mask = [1] * len(ids) + [0] * pad_len
        
        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(mask)
        batch_binary_spans.append(item["binary_spans"])
        batch_category_spans.append(item["category_spans"])
        batch_doc_ids.append(item["doc_id"])
        
    return {
        "doc_ids": batch_doc_ids,
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "binary_spans": batch_binary_spans,
        "category_spans": batch_category_spans
    }


def get_device(requested_device: str = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_joint(
    model: TwoStageSpanNERModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 1.0,
    save_path: str = None
):
    print("\n" + "=" * 75)
    print(f"🚀 STARTING JOINT MULTI-TASK TRAINING (Epochs: {epochs}, LR: {lr}, Alpha: {alpha})")
    print("=" * 75)
    
    # Train only parameters that require gradients (heads + length embeddings)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_bin_loss = 0.0
        total_cat_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_spans_binary=batch["binary_spans"],
                target_spans_category=batch["category_spans"],
                mode="joint",
                alpha=alpha
            )
            
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bin_loss += outputs["loss_binary"].item()
            total_cat_loss += outputs["loss_category"].item()
            
        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        avg_bin = total_bin_loss / len(train_loader)
        avg_cat = total_cat_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                v_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_spans_binary=batch["binary_spans"],
                    target_spans_category=batch["category_spans"],
                    mode="joint",
                    alpha=alpha
                )
                val_loss += v_out["loss"].item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {avg_loss:.4f} (Bin: {avg_bin:.4f}, Cat: {avg_cat:.4f}) | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                
    print(f"✅ Joint training complete! Best Val Loss: {best_val_loss:.4f}")
    if save_path:
        print(f"💾 Checkpoint saved to: {save_path}")


def train_sequential(
    model: TwoStageSpanNERModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-3,
    save_path: str = None
):
    print("\n" + "=" * 75)
    print(f"🚀 STARTING SEQUENTIAL TWO-PHASE TRAINING (Epochs/Phase: {epochs}, LR: {lr})")
    print("=" * 75)
    
    # --------------------------------------------------------------------------
    # Phase 1: Train Binary Proposal Head (Non-LSF vs LSF)
    # --------------------------------------------------------------------------
    print("\n[PHASE 1] Training Binary Proposal Head (0: Non-LSF, 1: LSF)...")
    p1_params = list(model.head_binary.parameters()) + list(model.span_len_embedding.parameters())
    opt_p1 = torch.optim.AdamW(p1_params, lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_bin_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            opt_p1.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_spans_binary=batch["binary_spans"],
                mode="binary"
            )
            loss = outputs["loss_binary"]
            loss.backward()
            opt_p1.step()
            total_bin_loss += loss.item()
            
        elapsed = time.time() - start_time
        print(f"  Phase 1 Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Binary Train Loss: {total_bin_loss / len(train_loader):.4f}")
        
    # --------------------------------------------------------------------------
    # Phase 2: Train 9-Class LSF Categorizer Head (0..8)
    # --------------------------------------------------------------------------
    print("\n[PHASE 2] Training 9-Class Categorization Head (0..8 LSF categories)...")
    p2_params = list(model.head_category.parameters())
    opt_p2 = torch.optim.AdamW(p2_params, lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_cat_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            opt_p2.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_spans_category=batch["category_spans"],
                mode="category"
            )
            loss = outputs["loss_category"]
            loss.backward()
            opt_p2.step()
            total_cat_loss += loss.item()
            
        elapsed = time.time() - start_time
        print(f"  Phase 2 Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | 9-Class Train Loss: {total_cat_loss / len(train_loader):.4f}")
        
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\n💾 Sequential two-phase checkpoint saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Two-Stage SpanNER Model")
    parser.add_argument("--mode", type=str, choices=["joint", "sequential", "all"], default="all",
                        help="Training regime: 'joint', 'sequential', or 'all'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for heads")
    parser.add_argument("--alpha", type=float, default=1.0, help="Category loss weight in joint mode")
    parser.add_argument("--device", type=str, default="", help="Device: cuda, mps, or cpu")
    parser.add_argument("--output_dir", type=str, default="models/two_stage_spanner",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using compute device: {device}")
    
    # Paths
    encoder_path = os.path.join(PROJECT_ROOT, "models/NER_Model/trained_NER_model")
    baseline_checkpoint = os.path.join(PROJECT_ROOT, "models/SpanNER_LSF/best_spanner_160train.pt")
    stage1_json = os.path.join(PROJECT_ROOT, "data/processed/spanner_dataset_160_stage1_binary.json")
    stage2_json = os.path.join(PROJECT_ROOT, "data/processed/spanner_dataset_160_stage2_9class.json")
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare Dataloaders
    print("Loading 160-abstract datasets...")
    train_dataset = SpanNERDataset(stage1_json, stage2_json, split="train")
    test_dataset = SpanNERDataset(stage1_json, stage2_json, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    print(f"Loaded {len(train_dataset)} train abstracts, {len(test_dataset)} test abstracts.")
    
    # 1. Execute Joint Training
    if args.mode in ("joint", "all"):
        model_joint = TwoStageSpanNERModel(encoder_path=encoder_path, max_span_width=6, freeze_encoder=True)
        if os.path.exists(baseline_checkpoint):
            model_joint.load_from_pretrained_spanner(baseline_checkpoint)
        model_joint.to(device)
        
        joint_save_path = os.path.join(output_dir, "two_stage_spanner_joint.pt")
        train_joint(
            model=model_joint,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            save_path=joint_save_path
        )
        
    # 2. Execute Sequential Training
    if args.mode in ("sequential", "all"):
        model_seq = TwoStageSpanNERModel(encoder_path=encoder_path, max_span_width=6, freeze_encoder=True)
        if os.path.exists(baseline_checkpoint):
            model_seq.load_from_pretrained_spanner(baseline_checkpoint)
        model_seq.to(device)
        
        seq_save_path = os.path.join(output_dir, "two_stage_spanner_sequential.pt")
        train_sequential(
            model=model_seq,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=seq_save_path
        )


if __name__ == "__main__":
    main()
