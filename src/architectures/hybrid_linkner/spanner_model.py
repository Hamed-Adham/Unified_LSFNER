import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, logging

# Suppress harmless HuggingFace pooler weight initialization warning
logging.set_verbosity_error()

class SpanNERModel(nn.Module):
    def __init__(self, encoder_path, num_classes=11, max_span_width=8, hidden_dim=1024, len_emb_dim=64):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        self.max_span_width = max_span_width
        self.num_classes = num_classes
        
        # Learned span length embeddings
        self.span_len_embedding = nn.Embedding(max_span_width + 1, len_emb_dim)
        
        # Span feature dimension: start token + end token + span length embedding
        span_dim = hidden_dim * 2 + len_emb_dim
        
        # Multi-layer Perceptron (MLP) Classifier for Span Prediction
        self.classifier = nn.Sequential(
            nn.Linear(span_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def get_candidate_spans(self, seq_len, device=None):
        """Enumerate candidate spans (start, end, length) up to max_span_width."""
        spans = []
        for length in range(1, min(self.max_span_width + 1, seq_len + 1)):
            for start in range(0, seq_len - length + 1):
                end = start + length - 1
                spans.append((start, end, length))
        return spans

    def get_candidate_span_tensors(self, seq_len, device):
        """Precomputes and caches candidate spans and index tensors for fast vectorized gathering."""
        if not hasattr(self, "_span_cache"):
            self._span_cache = {}
        cache_key = (seq_len, str(device))
        if cache_key not in self._span_cache:
            spans = self.get_candidate_spans(seq_len, device)
            starts = torch.tensor([s[0] for s in spans], dtype=torch.long, device=device)
            ends = torch.tensor([s[1] for s in spans], dtype=torch.long, device=device)
            lengths = torch.tensor([s[2] for s in spans], dtype=torch.long, device=device)
            self._span_cache[cache_key] = (spans, starts, ends, lengths)
        return self._span_cache[cache_key]

    def forward(self, input_ids, attention_mask=None, target_spans_list=None, class_weights=None):
        """
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
        target_spans_list: list of dicts/lists per batch item: [{"start": int, "end": int, "label_id": int}]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Extract contextual token embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [Batch, Seq_Len, Hidden_Dim]
        
        candidate_spans, starts_t, ends_t, lengths_t = self.get_candidate_span_tensors(seq_len, device)
        num_spans = len(candidate_spans)
        
        # Vectorized Span Representation Construction (10x-50x faster than Python loop)
        h_start = sequence_output[:, starts_t, :]  # [Batch, Num_Spans, Hidden_Dim]
        h_end = sequence_output[:, ends_t, :]      # [Batch, Num_Spans, Hidden_Dim]
        len_emb = self.span_len_embedding(lengths_t).unsqueeze(0).expand(batch_size, -1, -1)  # [Batch, Num_Spans, Len_Emb_Dim]
        
        # Concatenate and classify in one batched tensor operation: [Batch, Num_Spans, Span_Dim]
        span_reps = torch.cat([h_start, h_end, len_emb], dim=-1)
        logits = self.classifier(span_reps)  # [Batch, Num_Spans, Num_Classes]
        
        loss = None
        if target_spans_list is not None:
            # Build target tensor for all candidate spans
            targets = torch.zeros((batch_size, num_spans), dtype=torch.long, device=device)
            
            for b in range(batch_size):
                item_targets = target_spans_list[b]
                # Map target spans to candidate index
                target_map = {(t["start"], t["end"]): t["label_id"] for t in item_targets}
                
                for s_idx, (start, end, _) in enumerate(candidate_spans):
                    if (start, end) in target_map:
                        targets[b, s_idx] = target_map[(start, end)]
                        
            # Reshape for Cross Entropy Loss
            flat_logits = logits.view(-1, self.num_classes)
            flat_targets = targets.view(-1)
            
            if class_weights is not None:
                loss = F.cross_entropy(flat_logits, flat_targets, weight=class_weights.to(device))
            else:
                loss = F.cross_entropy(flat_logits, flat_targets)
                
        return {
            "logits": logits,
            "candidate_spans": candidate_spans,
            "loss": loss
        }
