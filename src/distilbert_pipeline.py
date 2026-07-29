"""Các hàm fine-tune, đánh giá và lưu DistilBERT.

Mỗi lần đánh giá khởi tạo lại mô hình từ checkpoint tiền huấn luyện để tránh
việc vô tình tái sử dụng trọng số đã nhìn thấy tập kiểm thử ở lần chia khác.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from common import dataloader_generator, set_global_seed
from metrics import classification_metrics


@dataclass(frozen=True)
class DistilBERTConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 150
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class EncodedTextDataset:
    """Dataset nhỏ gọn, tránh phụ thuộc vào datasets của Hugging Face."""

    def __init__(self, encodings, labels):
        import torch

        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _encode(tokenizer, texts: Sequence[str], max_length: int):
    return tokenizer(
        list(map(str, texts)),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def evaluate_distilbert_model(model, tokenizer, texts, labels, *, max_length: int, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    encodings = _encode(tokenizer, texts, max_length)
    loader = DataLoader(
        EncodedTextDataset(encodings, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
            probabilities.append(probs.cpu().numpy())
    inference_time = time.perf_counter() - started

    prediction_array = np.asarray(predictions, dtype=int)
    probability_array = np.vstack(probabilities) if probabilities else np.empty((0, 4), dtype=float)
    metrics = classification_metrics(labels, prediction_array)
    metrics["inference_time_s"] = float(inference_time)
    metrics["inference_ms_per_sample"] = float(1000.0 * inference_time / max(len(prediction_array), 1))
    return metrics, prediction_array, probability_array


def train_distilbert_once(
    train_texts: Sequence[str],
    y_train: Sequence[int],
    eval_texts: Sequence[str],
    y_eval: Sequence[int],
    *,
    seed: int = 42,
    config: DistilBERTConfig | None = None,
    save_dir: Path | None = None,
):
    """Fine-tune một mô hình mới và đánh giá trên tập eval.

    Weighted Cross-Entropy được tính bên ngoài model để xử lý mất cân bằng lớp.
    """
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    cfg = config or DistilBERTConfig()
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=4,
        id2label={0: "minimum", 1: "mild", 2: "moderate", 3: "severe"},
        label2id={"minimum": 0, "mild": 1, "moderate": 2, "severe": 3},
    ).to(device)

    train_encodings = _encode(tokenizer, train_texts, cfg.max_length)
    train_dataset = EncodedTextDataset(train_encodings, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=dataloader_generator(seed),
        num_workers=0,
    )

    y_train_arr = np.asarray(y_train, dtype=int)
    classes = np.arange(4)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_arr)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": running_loss / max(len(train_loader), 1),
            }
        )

    training_time = time.perf_counter() - started
    metrics, predictions, probabilities = evaluate_distilbert_model(
        model,
        tokenizer,
        eval_texts,
        y_eval,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    metrics["training_time_s"] = float(training_time)
    metrics["seed"] = int(seed)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    return metrics, predictions, probabilities, model, tokenizer, history
