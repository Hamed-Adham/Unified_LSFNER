"""
Two-Stage SpanNER Model:
Decoupled Neural Architecture for Span Proposal and LSF Categorization.

Stage 1: Binary Proposal Head (Non-LSF: 0, LSF: 1)
Stage 2: 9-Class LSF Categorizer Head (0..8 across the 9 lifestyle categories)

Designed with a frozen RoBERTa encoder backbone to eliminate language model
catastrophic forgetting and support fast, stable head training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, logging

logging.set_verbosity_error()

STAGE2_9CLASS_ID2LABEL = {
    0: "Personal_care_products_and_cosmetic_procedures",
    1: "Substance_use",
    2: "Environmental_exposures",
    3: "Mental_health_practices",
    4: "Non_physical_leisure_time_activities",
    5: "Nutrition",
    6: "Physical_activities",
    7: "Sleep",
    8: "Socioeconomic_factors"
}

STAGE2_9CLASS_LABEL2ID = {v: k for k, v in STAGE2_9CLASS_ID2LABEL.items()}

STAGE1_BINARY_ID2LABEL = {0: "Non-LSF", 1: "LSF"}
STAGE1_BINARY_LABEL2ID = {"Non-LSF": 0, "LSF": 1}


class TwoStageSpanNERModel(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        max_span_width: int = 6,
        hidden_dim: int = 1024,
        len_emb_dim: int = 64,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        self.max_span_width = max_span_width
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Learned span length embeddings
        self.span_len_embedding = nn.Embedding(max_span_width + 1, len_emb_dim)
        span_dim = hidden_dim * 2 + len_emb_dim
        
        # Head 1: Binary Proposal Head (0: Non-LSF, 1: LSF)
        self.head_binary = nn.Sequential(
            nn.Linear(span_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
        
        # Head 2: 9-Class Categorization Head (0..8: 9 LSF Categories)
        self.head_category = nn.Sequential(
            nn.Linear(span_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 9)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def load_from_pretrained_spanner(self, checkpoint_path: str):
        """
        Warm-starts the two-stage model from a 10-class SpanNER checkpoint.
        Transfers span length embeddings and MLP projection weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        sd = torch.load(checkpoint_path, map_location="cpu")
        
        # 1. Span length embedding
        if "span_len_embedding.weight" in sd:
            self.span_len_embedding.weight.data.copy_(sd["span_len_embedding.weight"])
            
        # 2. Shared first linear projection (classifier.0) into both heads
        if "classifier.0.weight" in sd:
            self.head_binary[0].weight.data.copy_(sd["classifier.0.weight"])
            self.head_binary[0].bias.data.copy_(sd["classifier.0.bias"])
            self.head_category[0].weight.data.copy_(sd["classifier.0.weight"])
            self.head_category[0].bias.data.copy_(sd["classifier.0.bias"])
            
        # 3. Output layer initialization (classifier.3)
        if "classifier.3.weight" in sd:
            c3_weight = sd["classifier.3.weight"]  # [10, 512]
            c3_bias = sd["classifier.3.bias"]      # [10]
            
            # Head 1 (Binary): Class 0 is Non-LSF (index 0), Class 1 is pooled LSF (indices 1..9)
            self.head_binary[3].weight.data[0].copy_(c3_weight[0])
            self.head_binary[3].bias.data[0].copy_(c3_bias[0])
            self.head_binary[3].weight.data[1].copy_(c3_weight[1:10].mean(dim=0))
            self.head_binary[3].bias.data[1].copy_(c3_bias[1:10].mean())
            
            # Head 2 (9-Class): Classes 1..9 from 10-class model map to 0..8
            self.head_category[3].weight.data.copy_(c3_weight[1:10])
            self.head_category[3].bias.data.copy_(c3_bias[1:10])
            
        print(f"✅ Successfully warm-started TwoStageSpanNERModel from {checkpoint_path}")

    def get_candidate_spans(self, seq_len: int):
        """Enumerate candidate spans (start, end, length) up to max_span_width."""
        spans = []
        for length in range(1, min(self.max_span_width + 1, seq_len + 1)):
            for start in range(0, seq_len - length + 1):
                end = start + length - 1
                spans.append((start, end, length))
        return spans

    def get_candidate_span_tensors(self, seq_len: int, device: torch.device):
        """Caches candidate span index tensors for fast vectorized gathering."""
        if not hasattr(self, "_span_cache"):
            self._span_cache = {}
        cache_key = (seq_len, str(device))
        if cache_key not in self._span_cache:
            spans = self.get_candidate_spans(seq_len)
            starts = torch.tensor([s[0] for s in spans], dtype=torch.long, device=device)
            ends = torch.tensor([s[1] for s in spans], dtype=torch.long, device=device)
            lengths = torch.tensor([s[2] for s in spans], dtype=torch.long, device=device)
            self._span_cache[cache_key] = (spans, starts, ends, lengths)
        return self._span_cache[cache_key]

    def extract_span_representations(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Runs RoBERTa and gathers vectorized span representations."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
        candidate_spans, starts_t, ends_t, lengths_t = self.get_candidate_span_tensors(seq_len, device)
        
        h_start = sequence_output[:, starts_t, :]
        h_end = sequence_output[:, ends_t, :]
        len_emb = self.span_len_embedding(lengths_t).unsqueeze(0).expand(batch_size, -1, -1)
        
        span_reps = torch.cat([h_start, h_end, len_emb], dim=-1)  # [Batch, Num_Spans, Span_Dim]
        return span_reps, candidate_spans

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        target_spans_binary: list = None,
        target_spans_category: list = None,
        mode: str = "joint",
        alpha: float = 1.0,
        binary_class_weights: torch.Tensor = None
    ):
        """
        Forward pass supporting 'joint', 'binary', and 'category' training modes.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        span_reps, candidate_spans = self.extract_span_representations(input_ids, attention_mask)
        num_spans = len(candidate_spans)
        
        binary_logits = None
        category_logits = None
        loss = None
        loss_binary = None
        loss_category = None
        
        # 1. Binary Head Pass (for 'joint' or 'binary' mode)
        if mode in ("joint", "binary"):
            binary_logits = self.head_binary(span_reps)  # [Batch, Num_Spans, 2]
            
            if target_spans_binary is not None:
                targets_bin = torch.zeros((batch_size, num_spans), dtype=torch.long, device=device)
                for b in range(batch_size):
                    t_map = {(t["start"], t["end"]): t["label_id"] for t in target_spans_binary[b]}
                    for s_idx, (start, end, _) in enumerate(candidate_spans):
                        if (start, end) in t_map:
                            targets_bin[b, s_idx] = t_map[(start, end)]
                            
                flat_bin_logits = binary_logits.view(-1, 2)
                flat_bin_targets = targets_bin.view(-1)
                
                if binary_class_weights is not None:
                    loss_binary = F.cross_entropy(flat_bin_logits, flat_bin_targets, weight=binary_class_weights.to(device))
                else:
                    loss_binary = F.cross_entropy(flat_bin_logits, flat_bin_targets)
                    
        # 2. 9-Class Categorizer Pass (for 'joint' or 'category' mode)
        if mode in ("joint", "category"):
            category_logits = self.head_category(span_reps)  # [Batch, Num_Spans, 9]
            
            if target_spans_category is not None:
                # Target spans contain ground-truth entity labels (0..8)
                # We mask loss to ONLY positive entity candidate spans!
                cat_targets = torch.full((batch_size, num_spans), -100, dtype=torch.long, device=device)
                num_pos = 0
                for b in range(batch_size):
                    t_map = {(t["start"], t["end"]): t["label_id"] for t in target_spans_category[b]}
                    for s_idx, (start, end, _) in enumerate(candidate_spans):
                        if (start, end) in t_map:
                            cat_targets[b, s_idx] = t_map[(start, end)]
                            num_pos += 1
                            
                flat_cat_logits = category_logits.view(-1, 9)
                flat_cat_targets = cat_targets.view(-1)
                
                if num_pos > 0:
                    loss_category = F.cross_entropy(flat_cat_logits, flat_cat_targets, ignore_index=-100)
                else:
                    loss_category = torch.tensor(0.0, device=device, requires_grad=True)
                    
        # 3. Combine losses based on mode
        if mode == "joint":
            loss = loss_binary + alpha * (loss_category if loss_category is not None else 0.0)
        elif mode == "binary":
            loss = loss_binary
        elif mode == "category":
            loss = loss_category
            
        return {
            "binary_logits": binary_logits,
            "category_logits": category_logits,
            "loss": loss,
            "loss_binary": loss_binary,
            "loss_category": loss_category,
            "candidate_spans": candidate_spans
        }

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        tau: float = 0.5
    ):
        """
        Two-stage inference:
        1. Run Head 1 -> filter spans with P(LSF) >= tau
        2. Run Head 2 -> assign predicted 9-class category to surviving spans.
        """
        batch_size, seq_len = input_ids.shape
        span_reps, candidate_spans = self.extract_span_representations(input_ids, attention_mask)
        
        # Step 1: Head 1 Binary Probabilities
        bin_logits = self.head_binary(span_reps)  # [Batch, Num_Spans, 2]
        bin_probs = F.softmax(bin_logits, dim=-1)
        p_lsf = bin_probs[:, :, 1]  # [Batch, Num_Spans]
        
        # Step 2: Head 2 9-Class Logits
        cat_logits = self.head_category(span_reps)  # [Batch, Num_Spans, 9]
        cat_probs = F.softmax(cat_logits, dim=-1)
        
        batch_predictions = []
        for b in range(batch_size):
            doc_preds = []
            for s_idx, (start, end, length) in enumerate(candidate_spans):
                prob_entity = p_lsf[b, s_idx].item()
                if prob_entity >= tau:
                    cat_id = torch.argmax(cat_logits[b, s_idx]).item()
                    cat_prob = cat_probs[b, s_idx, cat_id].item()
                    cat_label = STAGE2_9CLASS_ID2LABEL.get(cat_id, "Unknown")
                    doc_preds.append({
                        "start": start,
                        "end": end,
                        "length": length,
                        "p_lsf": prob_entity,
                        "label_id": cat_id,
                        "label": cat_label,
                        "cat_prob": cat_prob
                    })
            batch_predictions.append(doc_preds)
            
        return batch_predictions
