"""
src/task2/bert_data.py — BERT-compatible PyTorch dataset.

Tokenizes each document on-the-fly using a HuggingFace tokenizer.
This is shared by Task 2 (vanilla fine-tuning) and Task 3 (MLM →
classification fine-tuning).
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BertClassificationDataset(Dataset):
    """Tokenize text on-the-fly; return input_ids, attention_mask, label."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer(model_name: str):
    """Convenience wrapper around AutoTokenizer."""
    return AutoTokenizer.from_pretrained(model_name)
