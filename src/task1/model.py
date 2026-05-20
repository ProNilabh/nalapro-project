"""
src/task1/model.py — Simple two-layer neural network.

Architecture (as required by the task):
    Input → Linear → ReLU → Dropout → Linear → Logits

Same architecture is reused for Word2Vec, TF-IDF, and combined inputs;
only the input dimension changes.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SimpleNN(nn.Module):
    """Two linear layers with a ReLU in between, plus dropout."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class VectorDataset(Dataset):
    """Wraps a numpy feature matrix + integer labels for PyTorch."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
