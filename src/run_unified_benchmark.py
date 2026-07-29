"""Benchmark thống nhất năm mô hình trên cùng các lần chia dữ liệu.

Điểm khác với notebook cũ:
- Tất cả mô hình dùng đúng cùng train/test indices của mỗi seed.
- TF-IDF và TruncatedSVD chỉ fit trên train của từng split.
- Không nhập cứng kết quả từ lần chạy khác.
- Báo cáo mean ± std qua nhiều lần chia.

Lưu ý: chạy đủ BiLSTM và DistilBERT trên CPU có thể mất nhiều giờ.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from bilstm_pipeline import BiLSTMConfig, train_evaluate_bilstm
from common import LABEL_MAP, PROCESSED_DATA_PATH, REPORT_DIR, ensure_output_dirs, load_json, save_json, set_global_seed
from distilbert_pipeline import DistilBERTConfig, train_distilbert_once
from features import TfidfConfig, build_tfidf_vectorizer, fit_transform_xgb_features
from metrics import classification_metrics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
SUPPORTED_MODELS = ["logreg", "svm", "xgboost", "bilstm", "distilbert"]
DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "bilstm": "BiLSTM",
    "distilbert": "DistilBERT",
}


def add_result(metric_rows, prediction_rows, *, model_key, split_name, seed, test_idx, texts, y_true, y_pred, metrics):
    metric_rows.append(
        {
            "model": DISPLAY_NAMES[model_key],
            "split": split_name,
            "seed": seed,
            "test_size": len(test_idx),
            **metrics,
        }
    )
    prediction_rows.append(
        pd.DataFrame(
            {
                "model": DISPLAY_NAMES[model_key],
                "split": split_name,
                "seed": seed,
                "row_index": test_idx,
                "text": texts[test_idx],
                "true_label": y_true,
                "pred_label": y_pred,
            }
        )
    )


def load_distilbert_config(path: Path, model_name: str) -> DistilBERTConfig:
    raw = load_json(path, default={})
    raw["model_name"] = raw.get("model_name", model_name)
    fields = DistilBERTConfig.__dataclass_fields__.keys()
    return DistilBERTConfig(**{key: raw[key] for key in fields if key in raw})


def release(*objects) -> None:
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
    parser = argparse.ArgumentParser(description="Unified repeated-holdout benchmark")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS)
    parser.add_argument("--distilbert-config", type=Path, default=REPORT_DIR / "best_distilbert_config.json")
    parser.add_argument("--distilbert-model-name", default="distilbert-base-uncased")
    parser.add_argument("--bilstm-epochs", type=int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: chỉ seed đầu, BiLSTM 1 epoch, DistilBERT 1 epoch")
    args = parser.parse_args()

    ensure_output_dirs()
    df = pd.read_csv(args.data)
    required = {"text", "text_classical", "text_neural", "label"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột: {sorted(missing_columns)}")

    y_series = df["label"].str.lower().map(LABEL_MAP)
    if y_series.isna().any():
        raise ValueError("Có nhãn không hợp lệ trong dữ liệu.")
    y = y_series.to_numpy(dtype=int)
    raw_text = df["text"].fillna("").astype(str).to_numpy()
    classical_text = df["text_classical"].fillna("").astype(str).to_numpy()
    neural_text = df["text_neural"].fillna("").astype(str).to_numpy()
    all_indices = np.arange(len(df))

    seeds = args.seeds[:1] if args.quick else args.seeds
    distilbert_config = load_distilbert_config(args.distilbert_config, args.distilbert_model_name)
    if args.quick:
        distilbert_config = DistilBERTConfig(
            **{**distilbert_config.to_dict(), "epochs": 1, "max_length": min(distilbert_config.max_length, 64)}
        )
    bilstm_config = BiLSTMConfig(epochs=1 if args.quick else args.bilstm_epochs)
    tfidf_config = TfidfConfig()

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_selection_rows: list[dict] = []

    for seed in seeds:
        set_global_seed(seed)
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_size,
            stratify=y,
            random_state=seed,
        )
        split_name = f"seed_{seed}"
        print(f"\n===== {split_name}: train={len(train_idx)}, test={len(test_idx)} =====")

        # Fit TF-IDF một lần trên train cho Logistic Regression và SVM.
        vectorizer = build_tfidf_vectorizer(tfidf_config)
        x_train_tfidf = vectorizer.fit_transform(classical_text[train_idx])
        x_test_tfidf = vectorizer.transform(classical_text[test_idx])
        feature_selection_rows.append(
            {
                "split": split_name,
                "seed": seed,
                "selection_method": "TF-IDF min_df/max_df/max_features",
                "selected_tfidf_features": x_train_tfidf.shape[1],
                "max_features": tfidf_config.max_features,
                "min_df": tfidf_config.min_df,
                "max_df": tfidf_config.max_df,
                "ngram_min": tfidf_config.ngram_range[0],
                "ngram_max": tfidf_config.ngram_range[1],
            }
        )

        if "logreg" in args.models:
            model = LogisticRegression(
                C=0.1,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            )
            started = time.perf_counter()
            model.fit(x_train_tfidf, y[train_idx])
            training_time = time.perf_counter() - started
            pred = model.predict(x_test_tfidf)
            metrics = classification_metrics(y[test_idx], pred)
            metrics["training_time_s"] = training_time
            add_result(metric_rows, prediction_rows, model_key="logreg", split_name=split_name,
                       seed=seed, test_idx=test_idx, texts=raw_text, y_true=y[test_idx], y_pred=pred, metrics=metrics)
            print("LogReg:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

        if "svm" in args.models:
            model = SVC(
                C=0.5,
                kernel="linear",
                class_weight="balanced",
                random_state=seed,
            )
            started = time.perf_counter()
            model.fit(x_train_tfidf, y[train_idx])
            training_time = time.perf_counter() - started
            pred = model.predict(x_test_tfidf)
            metrics = classification_metrics(y[test_idx], pred)
            metrics["training_time_s"] = training_time
            add_result(metric_rows, prediction_rows, model_key="svm", split_name=split_name,
                       seed=seed, test_idx=test_idx, texts=raw_text, y_true=y[test_idx], y_pred=pred, metrics=metrics)
            print("SVM:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

        if "xgboost" in args.models:
            x_train_xgb, x_test_xgb, _, svd, feature_names = fit_transform_xgb_features(
                classical_text[train_idx],
                classical_text[test_idx],
                raw_text[train_idx],
                raw_text[test_idx],
                tfidf_config=tfidf_config,
                n_components=300,
                seed=seed,
            )
            feature_selection_rows.append(
                {
                    "split": split_name,
                    "seed": seed,
                    "selection_method": "TruncatedSVD + handcrafted",
                    "selected_tfidf_features": x_train_tfidf.shape[1],
                    "svd_components": len(feature_names) - 7,
                    "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
                    "handcrafted_features": 7,
                }
            )
            sample_weights = compute_sample_weight(class_weight="balanced", y=y[train_idx])
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=4,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                eval_metric="mlogloss",
                n_jobs=1,
            )
            started = time.perf_counter()
            model.fit(x_train_xgb, y[train_idx], sample_weight=sample_weights)
            training_time = time.perf_counter() - started
            pred = model.predict(x_test_xgb).astype(int)
            metrics = classification_metrics(y[test_idx], pred)
            metrics["training_time_s"] = training_time
            add_result(metric_rows, prediction_rows, model_key="xgboost", split_name=split_name,
                       seed=seed, test_idx=test_idx, texts=raw_text, y_true=y[test_idx], y_pred=pred, metrics=metrics)
            print("XGBoost:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
            release(model)

        if "bilstm" in args.models:
            metrics, pred, model, vocab = train_evaluate_bilstm(
                neural_text[train_idx],
                y[train_idx],
                neural_text[test_idx],
                y[test_idx],
                seed=seed,
                config=bilstm_config,
            )
            add_result(metric_rows, prediction_rows, model_key="bilstm", split_name=split_name,
                       seed=seed, test_idx=test_idx, texts=raw_text, y_true=y[test_idx], y_pred=pred, metrics=metrics)
            print("BiLSTM:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
            del model, vocab
            gc.collect()

        if "distilbert" in args.models:
            metrics, pred, probabilities, model, tokenizer, history = train_distilbert_once(
                neural_text[train_idx],
                y[train_idx],
                neural_text[test_idx],
                y[test_idx],
                seed=seed,
                config=distilbert_config,
            )
            add_result(metric_rows, prediction_rows, model_key="distilbert", split_name=split_name,
                       seed=seed, test_idx=test_idx, texts=raw_text, y_true=y[test_idx], y_pred=pred, metrics=metrics)
            print("DistilBERT:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
            del model, tokenizer, probabilities, history
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    metrics_df = pd.DataFrame(metric_rows)
    output_prefix = "smoke_test_unified_repeated_holdout" if args.quick else "unified_repeated_holdout"
    metrics_path = REPORT_DIR / f"{output_prefix}_metrics.csv"
    predictions_path = REPORT_DIR / f"{output_prefix}_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False)
    pd.DataFrame(feature_selection_rows).to_csv(
        REPORT_DIR / f"{output_prefix}_feature_selection_by_split.csv", index=False
    )

    numeric_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "qwk",
        "training_time_s",
    ]
    summary_parts = []
    for model_name, group in metrics_df.groupby("model"):
        for metric in numeric_metrics:
            summary_parts.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                    "min": group[metric].min(),
                    "max": group[metric].max(),
                    "n_splits": len(group),
                }
            )
    summary_df = pd.DataFrame(summary_parts)
    summary_df.to_csv(REPORT_DIR / f"{output_prefix}_summary.csv", index=False)
    save_json(
        {
            "protocol": "Repeated Stratified Holdout",
            "seeds": seeds,
            "test_size": args.test_size,
            "models": args.models,
            "distilbert_config": distilbert_config.to_dict(),
            "bilstm_config": bilstm_config.__dict__,
            "data_leakage_control": "TF-IDF/SVD/vocabulary/model are fitted only on train of each split.",
        },
        REPORT_DIR / f"{output_prefix}_metadata.json",
    )

    print("\n=== TÓM TẮT MEAN ± STD ===")
    print(summary_df.pivot(index="model", columns="metric", values="mean").round(4))
    print(f"\nĐã lưu: {metrics_path}")
    print(f"Đã lưu: {predictions_path}")


if __name__ == "__main__":
    main()
