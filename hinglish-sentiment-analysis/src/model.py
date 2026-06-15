"""
Transformer model + PyTorch Dataset for Hinglish sentiment classification.

Architecture (matches the requested diagram):

    Input Text
        -> Tokenizer (HF AutoTokenizer)
        -> Transformer Encoder (IndicBERT / mBERT / DistilBERT)
        -> [CLS] pooled representation
        -> Dropout
        -> Dense (Linear) classifier
        -> Softmax  (applied at inference)
        -> Sentiment Prediction

We build a thin custom head on top of `AutoModel` rather than using
`AutoModelForSequenceClassification` so the dropout + dense layer are explicit
and easy to reason about / tweak.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import CONFIG, ID2LABEL, LABEL2ID, NUM_LABELS


# --------------------------------------------------------------------------- #
# Tokenizer helper
# --------------------------------------------------------------------------- #
def load_tokenizer(model_name: str = None):
    model_name = model_name or CONFIG.model_name
    return AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SentimentDataset(Dataset):
    """Tokenises text on the fly; returns input_ids / attention_mask / labels."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class HinglishSentimentClassifier(nn.Module):
    """Transformer encoder + dropout + linear classification head."""

    def __init__(self, model_name: str = None, num_labels: int = NUM_LABELS,
                 dropout: float = None):
        super().__init__()
        self.model_name = model_name or CONFIG.model_name
        self.num_labels = num_labels
        dropout = CONFIG.dropout if dropout is None else dropout

        self.encoder_config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        hidden = self.encoder_config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        # Keep label maps on the module for easy serialisation.
        self.id2label = dict(ID2LABEL)
        self.label2id = dict(LABEL2ID)

    def _pool(self, encoder_outputs, attention_mask) -> torch.Tensor:
        """Return a sentence vector. Prefer pooler_output, else CLS, else mean."""
        if getattr(encoder_outputs, "pooler_output", None) is not None:
            return encoder_outputs.pooler_output
        last_hidden = encoder_outputs.last_hidden_state  # (B, T, H)
        # Masked mean pooling as a robust fallback (e.g. DistilBERT has no pooler).
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled = self._pool(outputs, attention_mask)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    # ----- persistence -----
    def save(self, save_dir, tokenizer=None):
        """Save weights + config + tokenizer to `save_dir`."""
        from pathlib import Path
        import json

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        meta = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "max_length": CONFIG.max_length,
        }
        with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir, map_location=None):
        """Load a model saved with :meth:`save`. Returns (model, tokenizer, meta)."""
        from pathlib import Path
        import json

        save_dir = Path(save_dir)
        with open(save_dir / "model_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(model_name=meta["model_name"], num_labels=meta["num_labels"])
        state = torch.load(save_dir / "pytorch_model.bin", map_location=map_location or "cpu")
        model.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        return model, tokenizer, meta
