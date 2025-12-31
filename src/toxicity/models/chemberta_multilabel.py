"""
ChemBERTa multi-label toxicity model.

Purpose:
- Use pretrained ChemBERTa to encode SMILES
- Predict multiple toxicity assays independently
- Handle missing labels and class imbalance
"""

from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


class ChemBERTaMultiLabel(nn.Module):
    # ChemBERTa encoder + multi-label classification head
    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        n_tasks: int = 12,
        dropout: float = 0.1,
        pooling: str = "cls",
        freeze_encoder: bool = False,
        pos_weight: Optional[torch.Tensor] = None,
        use_mlp_head: bool = True,
        mlp_hidden_mult: int = 2,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        # Config
        self.n_tasks = int(n_tasks)
        self.pooling = pooling
        self.label_smoothing = float(label_smoothing)

        # Pretrained ChemBERTa encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.encoder.config.hidden_size)

        # Classification head
        if use_mlp_head:
            hidden = max(self.hidden_size * mlp_hidden_mult // 2, self.hidden_size)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.n_tasks),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.n_tasks),
            )

        # Class imbalance weights (per task)
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight.view(-1))
        else:
            self.pos_weight = None

        # Multi-label loss (sigmoid inside)
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
        )

        # Optional feature-extraction mode
        if freeze_encoder:
            self.freeze_all_encoder()

    # Freeze full encoder
    def freeze_all_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    # Unfreeze full encoder
    def unfreeze_all_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    # Unfreeze last n transformer blocks
    def unfreeze_last_n_layers(self, n: int = 2):
        self.freeze_all_encoder()
        layers = self.encoder.encoder.layer
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Convert token embeddings → molecule embedding
    def pool_output(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        raise ValueError("Invalid pooling")

    # Forward pass
    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool_output(out.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            raw_loss = self.loss_fn(logits, labels)
            if label_mask is not None:
                loss = (raw_loss * label_mask).sum() / label_mask.sum().clamp(min=1.0)
            else:
                loss = raw_loss.mean()

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
        }

    # Thresholded predictions
    @torch.no_grad()
    def predict(self, input_ids, attention_mask, threshold=0.5):
        probs = self.forward(input_ids, attention_mask)["probabilities"]
        return (probs >= threshold).long()


# Compute per-task positive class weights from training data
def compute_pos_weights(train_df, task_cols, method="sqrt", clip_max=20.0):
    weights = []
    for task in task_cols:
        values = train_df[task].dropna().values
        if len(values) == 0:
            w = 1.0
        else:
            pos = (values == 1).sum()
            neg = (values == 0).sum()
            w = np.sqrt(neg / max(pos, 1)) if method == "sqrt" else neg / max(pos, 1)
        weights.append(min(w, clip_max) if clip_max else w)
    return torch.tensor(weights, dtype=torch.float32)
