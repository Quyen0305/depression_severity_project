"""Tinh chỉnh tối thiểu DistilBERT trên tập development riêng biệt.

Quy trình chống leakage:
1. Tách một tập test cố định 20% và không dùng trong lựa chọn siêu tham số.
2. Tách phần còn lại thành train/validation.
3. Chọn cấu hình theo Macro-F1 validation, dùng QWK làm tiêu chí phụ.
4. Huấn luyện lại cấu hình tốt nhất trên train+validation và đánh giá đúng một lần trên test.
"""
from __future__ import annotations

import argparse
import gc
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import LABEL_MAP, PROCESSED_DATA_PATH, REPORT_DIR, ensure_output_dirs, save_json, set_global_seed
from distilbert_pipeline import DistilBERTConfig, train_distilbert_once
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_candidates(args) -> list[DistilBERTConfig]:
    combinations = itertools.product(
        parse_float_list(args.learning_rates),
        parse_int_list(args.epochs),
        parse_int_list(args.max_lengths),
    )
    candidates = [
        DistilBERTConfig(
            model_name=args.model_name,
            learning_rate=lr,
            epochs=epochs,
            max_length=max_length,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
        )
        for lr, epochs, max_length in combinations
    ]
    if args.max_configs is not None and args.max_configs < len(candidates):
        # Lấy mẫu đều trên toàn bộ lưới để tuning tối thiểu vẫn bao phủ nhiều
        # learning rate/epoch/max_length, thay vì chỉ lấy các cấu hình đầu tiên.
        selected_indices = np.linspace(0, len(candidates) - 1, args.max_configs, dtype=int)
        candidates = [candidates[index] for index in selected_indices]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal DistilBERT hyperparameter tuning")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--text-column", default="text_neural")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.15,
                        help="Tỉ lệ validation tính trên phần train_dev")
    parser.add_argument("--learning-rates", default="1e-5,2e-5,3e-5")
    parser.add_argument("--epochs", default="2,3")
    parser.add_argument("--max-lengths", default="128,150")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-configs", type=int, default=4,
                        help="Giới hạn số cấu hình để tuning tối thiểu trên CPU; đặt 0 để chạy toàn bộ")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--output-model-dir", type=Path, default=Path("models/distilbert_model_tuned"))
    args = parser.parse_args()
    if args.max_configs == 0:
        args.max_configs = None

    ensure_output_dirs()
    set_global_seed(args.seed)
    df = pd.read_csv(args.data)
    if args.text_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột {args.text_column!r} trong {args.data}")
    if df["label"].isna().any():
        raise ValueError("Cột label chứa giá trị thiếu. Hãy chạy data cleaning trước.")

    texts = df[args.text_column].fillna("").astype(str).to_numpy()
    labels = df["label"].str.lower().map(LABEL_MAP)
    if labels.isna().any():
        unknown = sorted(df.loc[labels.isna(), "label"].astype(str).unique())
        raise ValueError(f"Có nhãn không hợp lệ: {unknown}")
    y = labels.to_numpy(dtype=int)

    x_train_dev, x_test, y_train_dev, y_test = train_test_split(
        texts,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_dev,
        y_train_dev,
        test_size=args.validation_size,
        stratify=y_train_dev,
        random_state=args.seed,
    )

    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("Không có cấu hình tuning nào được tạo.")

    rows: list[dict] = []
    for index, config in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)}] {config.to_dict()}")
        metrics, _, _, model, tokenizer, history = train_distilbert_once(
            x_train,
            y_train,
            x_val,
            y_val,
            seed=args.seed,
            config=config,
        )
        row = {
            "config_id": index,
            **config.to_dict(),
            **metrics,
            "final_train_loss": history[-1]["mean_train_loss"],
        }
        rows.append(row)
        print(
            f"Validation Macro-F1={metrics['macro_f1']:.4f}, "
            f"QWK={metrics['qwk']:.4f}, Accuracy={metrics['accuracy']:.4f}"
        )
        del model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    tuning_df = pd.DataFrame(rows).sort_values(
        ["macro_f1", "qwk", "accuracy"], ascending=False
    )
    tuning_path = REPORT_DIR / "distilbert_tuning_results.csv"
    tuning_df.to_csv(tuning_path, index=False)

    best_row = tuning_df.iloc[0]
    best_config = DistilBERTConfig(
        model_name=str(best_row["model_name"]),
        max_length=int(best_row["max_length"]),
        batch_size=int(best_row["batch_size"]),
        epochs=int(best_row["epochs"]),
        learning_rate=float(best_row["learning_rate"]),
        weight_decay=float(best_row["weight_decay"]),
        warmup_ratio=float(best_row["warmup_ratio"]),
    )
    best_config_path = REPORT_DIR / "best_distilbert_config.json"
    save_json(best_config.to_dict(), best_config_path)

    print("\nHuấn luyện lại cấu hình tốt nhất trên train+validation...")
    save_dir = args.output_model_dir if args.save_model else None
    test_metrics, test_pred, test_prob, model, tokenizer, history = train_distilbert_once(
        x_train_dev,
        y_train_dev,
        x_test,
        y_test,
        seed=args.seed,
        config=best_config,
        save_dir=save_dir,
    )
    test_result = {
        "selection_metric": "macro_f1",
        "secondary_metric": "qwk",
        "seed": args.seed,
        "train_dev_size": len(x_train_dev),
        "test_size": len(x_test),
        "best_config": best_config.to_dict(),
        "test_metrics": test_metrics,
    }
    save_json(test_result, REPORT_DIR / "distilbert_tuned_holdout_result.json")

    prediction_df = pd.DataFrame(
        {
            "text": x_test,
            "true_label": y_test,
            "pred_label": test_pred,
            "prob_minimum": test_prob[:, 0],
            "prob_mild": test_prob[:, 1],
            "prob_moderate": test_prob[:, 2],
            "prob_severe": test_prob[:, 3],
        }
    )
    prediction_df.to_csv(REPORT_DIR / "distilbert_tuned_holdout_predictions.csv", index=False)

    print(f"\nĐã lưu tuning: {tuning_path}")
    print(f"Đã lưu cấu hình tốt nhất: {best_config_path}")
    print("Kết quả test chưa dùng trong tuning:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
