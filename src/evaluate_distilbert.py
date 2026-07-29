"""Đánh giá DistilBERT bằng repeated stratified holdout hoặc Stratified K-Fold.

Mỗi split/fold khởi tạo mô hình mới từ checkpoint tiền huấn luyện. Kết quả được
báo cáo bằng trung bình và độ lệch chuẩn thay vì một lần chia duy nhất.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import LABEL_MAP, PROCESSED_DATA_PATH, REPORT_DIR, ensure_output_dirs, load_json, save_json
from distilbert_pipeline import DistilBERTConfig, train_distilbert_once


def load_config(path: Path | None, model_name: str) -> DistilBERTConfig:
    if path is None or not path.exists():
        return DistilBERTConfig(model_name=model_name)
    raw = load_json(path)
    raw["model_name"] = raw.get("model_name", model_name)
    allowed = DistilBERTConfig.__dataclass_fields__.keys()
    return DistilBERTConfig(**{key: raw[key] for key in allowed if key in raw})


def release_model(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated/K-Fold DistilBERT evaluation")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--text-column", default="text_neural")
    parser.add_argument("--protocol", choices=["repeated_holdout", "kfold"], default="repeated_holdout")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--config", type=Path, default=REPORT_DIR / "best_distilbert_config.json")
    args = parser.parse_args()

    ensure_output_dirs()
    df = pd.read_csv(args.data)
    texts = df[args.text_column].fillna("").astype(str).to_numpy()
    labels = df["label"].str.lower().map(LABEL_MAP)
    if labels.isna().any():
        raise ValueError("Dữ liệu có nhãn không hợp lệ.")
    y = labels.to_numpy(dtype=int)
    config = load_config(args.config, args.model_name)

    splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    if args.protocol == "repeated_holdout":
        all_indices = np.arange(len(df))
        for seed in args.seeds:
            train_idx, test_idx = train_test_split(
                all_indices,
                test_size=args.test_size,
                stratify=y,
                random_state=seed,
            )
            splits.append((f"seed_{seed}", seed, train_idx, test_idx))
    else:
        seed = args.seeds[0]
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, y), start=1):
            splits.append((f"fold_{fold}", seed + fold, train_idx, test_idx))

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for split_name, seed, train_idx, test_idx in splits:
        print(f"\n=== {split_name}: train={len(train_idx)}, test={len(test_idx)} ===")
        metrics, predictions, probabilities, model, tokenizer, history = train_distilbert_once(
            texts[train_idx],
            y[train_idx],
            texts[test_idx],
            y[test_idx],
            seed=seed,
            config=config,
        )
        metric_rows.append(
            {
                "split": split_name,
                "seed": seed,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                **metrics,
            }
        )
        split_predictions = pd.DataFrame(
            {
                "protocol": args.protocol,
                "split": split_name,
                "seed": seed,
                "row_index": test_idx,
                "text": texts[test_idx],
                "true_label": y[test_idx],
                "pred_label": predictions,
                "prob_minimum": probabilities[:, 0],
                "prob_mild": probabilities[:, 1],
                "prob_moderate": probabilities[:, 2],
                "prob_severe": probabilities[:, 3],
            }
        )
        prediction_rows.append(split_predictions)
        print(
            f"Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
            f"QWK={metrics['qwk']:.4f}"
        )
        del model, tokenizer, probabilities, history
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    metrics_df = pd.DataFrame(metric_rows)
    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "qwk",
        "training_time_s",
        "inference_ms_per_sample",
    ]
    summary = pd.DataFrame(
        {
            "mean": metrics_df[metric_columns].mean(),
            "std": metrics_df[metric_columns].std(ddof=1),
            "min": metrics_df[metric_columns].min(),
            "max": metrics_df[metric_columns].max(),
        }
    )

    prefix = f"distilbert_{args.protocol}"
    metrics_df.to_csv(REPORT_DIR / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(REPORT_DIR / f"{prefix}_summary.csv")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        REPORT_DIR / f"{prefix}_predictions.csv", index=False
    )
    save_json(
        {
            "protocol": args.protocol,
            "seeds": args.seeds,
            "n_splits": args.n_splits if args.protocol == "kfold" else None,
            "test_size": args.test_size if args.protocol == "repeated_holdout" else None,
            "config": config.to_dict(),
            "note": "Mỗi split/fold khởi tạo lại DistilBERT từ checkpoint tiền huấn luyện.",
        },
        REPORT_DIR / f"{prefix}_metadata.json",
    )

    print("\n=== TÓM TẮT ===")
    print(summary.round(4))


if __name__ == "__main__":
    main()
