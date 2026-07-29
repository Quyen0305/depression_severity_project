"""Phân tích dữ liệu thiếu, outlier và giới hạn fairness của bộ dữ liệu.

Fairness ở đây được trình bày trung thực: dataset không có thuộc tính bảo vệ như
độ tuổi, giới tính, dân tộc hoặc khu vực nên không thể tính demographic parity,
equal opportunity hay equalized odds theo nhóm nhân khẩu học.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from common import LABEL_MAP, PROCESSED_DATA_PATH, REPORT_DIR, ensure_output_dirs, save_json


def length_bin(series: pd.Series) -> pd.Series:
    """Chia độ dài thành ba nhóm dựa trên phân vị, phục vụ slice analysis."""
    try:
        return pd.qcut(series, q=3, labels=["short", "medium", "long"], duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=3, labels=["short", "medium", "long"])


def analyze_predictions(prediction_path: Path, original_df: pd.DataFrame) -> pd.DataFrame:
    if not prediction_path.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(prediction_path)
    required = {"model", "true_label", "pred_label", "text"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    predictions["word_count"] = predictions["text"].fillna("").astype(str).str.split().str.len()
    predictions["length_slice"] = length_bin(predictions["word_count"])
    rows = []
    group_columns = ["model", "length_slice"]
    if "split" in predictions.columns:
        group_columns.insert(1, "split")

    for keys, group in predictions.groupby(group_columns, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        rows.append(
            {
                **key_map,
                "n_samples": len(group),
                "accuracy": accuracy_score(group["true_label"], group["pred_label"]),
                "macro_f1": f1_score(
                    group["true_label"], group["pred_label"], average="macro", zero_division=0
                ),
                "recall_severe": recall_score(
                    group["true_label"] == 3,
                    group["pred_label"] == 3,
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality, outlier and fairness audit")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPORT_DIR / "unified_repeated_holdout_predictions.csv",
        help="Prediction file để phân tích hiệu năng theo độ dài. Không bắt buộc.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    df = pd.read_csv(args.data)
    required_columns = ["text", "label", "text_classical", "text_neural"]

    missing_rows = []
    for column in required_columns:
        if column not in df.columns:
            missing_rows.append(
                {"column": column, "missing_count": len(df), "missing_ratio": 1.0, "status": "column_missing"}
            )
            continue
        count = int(df[column].isna().sum())
        blank_count = int(df[column].fillna("").astype(str).str.strip().eq("").sum())
        missing_rows.append(
            {
                "column": column,
                "missing_count": count,
                "blank_count": blank_count,
                "missing_ratio": count / max(len(df), 1),
                "status": "ok" if count == 0 and blank_count == 0 else "review",
            }
        )
    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(REPORT_DIR / "missing_data_report.csv", index=False)

    # Kiểm tra nhãn và trùng lặp còn tồn tại.
    normalized_labels = df["label"].fillna("").astype(str).str.lower()
    unknown_labels = sorted(set(normalized_labels) - set(LABEL_MAP))
    exact_duplicates = int(df.duplicated(subset=["text"], keep=False).sum()) if "text" in df.columns else 0

    # Outlier theo số từ: chỉ gắn cờ, không tự động xóa vì bài dài có thể hợp lệ.
    df = df.copy()
    df["word_count"] = df["text"].fillna("").astype(str).str.split().str.len()
    q1 = float(df["word_count"].quantile(0.25))
    q3 = float(df["word_count"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    df["length_outlier"] = (df["word_count"] < lower_bound) | (df["word_count"] > upper_bound)

    outlier_summary = (
        df.groupby("label", dropna=False)
        .agg(
            n_samples=("word_count", "size"),
            mean_words=("word_count", "mean"),
            median_words=("word_count", "median"),
            min_words=("word_count", "min"),
            max_words=("word_count", "max"),
            outlier_count=("length_outlier", "sum"),
        )
        .reset_index()
    )
    outlier_summary["outlier_ratio"] = outlier_summary["outlier_count"] / outlier_summary["n_samples"]
    outlier_summary.to_csv(REPORT_DIR / "text_length_outlier_summary.csv", index=False)
    df.loc[df["length_outlier"], ["text", "label", "word_count"]].to_csv(
        REPORT_DIR / "text_length_outlier_examples.csv", index=False
    )

    label_distribution = (
        normalized_labels.value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    label_distribution["ratio"] = label_distribution["count"] / len(df)
    label_distribution.to_csv(REPORT_DIR / "label_representation_report.csv", index=False)

    slice_df = analyze_predictions(args.predictions, df)
    slice_output_name = (
        "smoke_test_performance_by_text_length_slice.csv"
        if "smoke_test" in args.predictions.name
        else "performance_by_text_length_slice.csv"
    )
    if not slice_df.empty:
        slice_df.to_csv(REPORT_DIR / slice_output_name, index=False)

    audit = {
        "n_rows": len(df),
        "unknown_labels": unknown_labels,
        "exact_duplicate_rows_remaining": exact_duplicates,
        "word_count_q1": q1,
        "word_count_q3": q3,
        "word_count_iqr": iqr,
        "outlier_lower_bound": lower_bound,
        "outlier_upper_bound": upper_bound,
        "outlier_count": int(df["length_outlier"].sum()),
        "outlier_policy": (
            "Flag for review; do not automatically remove. Long/short posts may be valid. "
            "Sequence models truncate/pad according to max_length."
        ),
        "protected_attributes_available": [],
        "demographic_fairness_metrics_possible": False,
        "fairness_limitations": [
            "Không có tuổi, giới tính, dân tộc, khu vực hoặc nhóm văn hóa.",
            "Không thể tính demographic parity, equal opportunity hoặc equalized odds theo nhóm nhân khẩu học.",
            "Dữ liệu Reddit tiếng Anh có thể thiên lệch theo nền tảng, ngôn ngữ và nhóm người chủ động đăng bài.",
            "Phân tích theo độ dài văn bản chỉ là performance slicing, không thay thế fairness audit nhân khẩu học.",
        ],
    }
    save_json(audit, REPORT_DIR / "data_quality_fairness_audit.json")

    slice_note = (
        f"Đã tạo {slice_output_name} từ kết quả benchmark thống nhất."
        if not slice_df.empty
        else "Chưa có prediction file; chạy run_unified_benchmark.py rồi chạy lại script này để phân tích theo độ dài."
    )
    report = f"""# Data Quality, Outlier và Fairness Audit

## Dữ liệu thiếu

- Số dòng: **{len(df)}**.
- Báo cáo chi tiết: `missing_data_report.csv`.
- Nhãn không hợp lệ: **{unknown_labels if unknown_labels else 'không có'}**.
- Số dòng còn trùng nội dung trong file đầu vào: **{exact_duplicates}**.

## Outlier độ dài văn bản

- Q1 = **{q1:.2f}** từ; Q3 = **{q3:.2f}** từ; IQR = **{iqr:.2f}**.
- Ngưỡng IQR: [{lower_bound:.2f}, {upper_bound:.2f}] từ.
- Số mẫu được gắn cờ outlier: **{int(df['length_outlier'].sum())}**.
- Các mẫu này **không bị xóa tự động** vì bài đăng dài hoặc ngắn vẫn có thể là dữ liệu hợp lệ. Với BiLSTM/DistilBERT, chuỗi được padding hoặc truncation theo `max_length`.

## Fairness

Dataset không cung cấp thuộc tính bảo vệ như tuổi, giới tính, dân tộc, khu vực hoặc nhóm văn hóa. Vì vậy, dự án **không thể** tính các thước đo fairness nhân khẩu học như demographic parity, equal opportunity hoặc equalized odds.

Các rủi ro còn lại:

1. Dữ liệu Reddit tiếng Anh không đại diện cho toàn bộ cộng đồng hoặc người dùng Việt Nam.
2. Người chủ động đăng bài trên Reddit có thể khác với người không sử dụng mạng xã hội.
3. Mất cân bằng nhãn có thể làm giảm Recall của các lớp thiểu số.
4. Phân tích theo độ dài văn bản chỉ là phân tích lát cắt hiệu năng, không phải bằng chứng fairness theo nhân khẩu học.

{slice_note}
"""
    (REPORT_DIR / "data_quality_fairness_report.md").write_text(report, encoding="utf-8")

    print("Đã lưu báo cáo dữ liệu thiếu, outlier và fairness vào outputs/reports.")
    print(missing_df)
    print(outlier_summary.round(4))
    print(slice_note)


if __name__ == "__main__":
    main()
