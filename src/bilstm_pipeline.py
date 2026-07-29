"""Huấn luyện và đánh giá BiLSTM với seed cố định và vocabulary chỉ từ train."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from common import dataloader_generator, set_global_seed
from metrics import classification_metrics

TOKEN_RE = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b", flags=re.IGNORECASE)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class BiLSTMConfig:
    max_length: int = 150
    min_frequency: int = 2
    embedding_dim: int = 100
    hidden_dim: int = 64
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3


def simple_tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def build_vocabulary(texts: Sequence[str], min_frequency: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counter.items()):
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in simple_tokenize(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))
    return token_ids


def train_evaluate_bilstm(
    train_texts: Sequence[str],
    y_train: Sequence[int],
    test_texts: Sequence[str],
    y_test: Sequence[int],
    *,
    seed: int = 42,
    config: BiLSTMConfig | None = None,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    cfg = config or BiLSTMConfig()
    set_global_seed(seed)

    vocab = build_vocabulary(train_texts, cfg.min_frequency)
    train_matrix = np.asarray([encode_text(text, vocab, cfg.max_length) for text in train_texts], dtype=np.int64)
    test_matrix = np.asarray([encode_text(text, vocab, cfg.max_length) for text in test_texts], dtype=np.int64)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class TextDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.as_tensor(x, dtype=torch.long)
            self.y = torch.as_tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            return self.x[index], self.y[index]

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.fc = nn.Linear(cfg.hidden_dim * 2, 4)

        def forward(self, token_ids):
            embedded = self.embedding(token_ids)
            _, (hidden, _) = self.lstm(embedded)
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.fc(self.dropout(final_hidden))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(len(vocab)).to(device)

    classes = np.arange(4)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_arr)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader = DataLoader(
        TextDataset(train_matrix, y_train_arr),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=dataloader_generator(seed),
        num_workers=0,
    )
    test_loader = DataLoader(
        TextDataset(test_matrix, y_test_arr),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    started = time.perf_counter()
    model.train()
    for _ in range(cfg.epochs):
        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    training_time = time.perf_counter() - started

    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for token_ids, _ in test_loader:
            logits = model(token_ids.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = classification_metrics(y_test_arr, predictions)
    metrics["training_time_s"] = float(training_time)
    metrics["vocab_size"] = int(len(vocab))
    return metrics, np.asarray(predictions, dtype=int), model, vocab
