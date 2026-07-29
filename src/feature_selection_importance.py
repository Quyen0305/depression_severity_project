"""Làm rõ feature selection và feature importance của nhánh mô hình cổ điển.

Kết quả được lưu vào outputs/reports và outputs/figures:
- Số đặc trưng TF-IDF thực tế sau min_df/max_df/max_features.
- Top hệ số Logistic Regression theo từng lớp.
- Feature importance của XGBoost.
- Permutation importance riêng cho 7 handcrafted features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import FIGURE_DIR, ID_TO_LABEL, LABEL_MAP, PROCESSED_DATA_PATH, REPORT_DIR, ensure_output_dirs, save_json, set_global_seed
from features import HANDCRAFTED_FEATURE_NAMES, TfidfConfig, build_tfidf_vectorizer, fit_transform_xgb_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature selection and importance analysis")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    args = parser.parse_args()

    ensure_output_dirs()
    set_global_seed(args.seed)
    df = pd.read_csv(args.data)
    y_series = df["label"].str.lower().map(LABEL_MAP)
    if y_series.isna().any():
        raise ValueError("Có nhãn không hợp lệ.")
    y = y_series.to_numpy(dtype=int)
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    classical_text = df["text_classical"].fillna("").astype(str).to_numpy()
    raw_text = df["text"].fillna("").astype(str).to_numpy()
    tfidf_config = TfidfConfig()

    # 1) Feature selection gián tiếp của TF-IDF.
    vectorizer = build_tfidf_vectorizer(tfidf_config)
    x_train = vectorizer.fit_transform(classical_text[train_idx])
    x_test = vectorizer.transform(classical_text[test_idx])
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    selection_summary = {
        "selection_type": "implicit_filtering",
        "description": (
            "TF-IDF dùng min_df/max_df để loại token quá hiếm/quá phổ biến, "
            "max_features để giới hạn đặc trưng; TruncatedSVD giảm chiều cho XGBoost."
        ),
        "selected_tfidf_features": int(x_train.shape[1]),
        "max_features": tfidf_config.max_features,
        "min_df": tfidf_config.min_df,
        "max_df": tfidf_config.max_df,
        "ngram_range": list(tfidf_config.ngram_range),
    }

    # 2) Logistic Regression coefficients theo lớp.
    logreg = LogisticRegression(
        C=0.1,
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    logreg.fit(x_train, y[train_idx])
    coefficient_rows = []
    for class_index, coefficients in enumerate(logreg.coef_):
        top_positive = np.argsort(coefficients)[-25:][::-1]
        top_negative = np.argsort(coefficients)[:25]
        for rank, feature_index in enumerate(top_positive, start=1):
            coefficient_rows.append(
                {
                    "class_id": class_index,
                    "class_name": ID_TO_LABEL[class_index],
                    "direction": "positive",
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "coefficient": coefficients[feature_index],
                }
            )
        for rank, feature_index in enumerate(top_negative, start=1):
            coefficient_rows.append(
                {
                    "class_id": class_index,
                    "class_name": ID_TO_LABEL[class_index],
                    "direction": "negative",
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "coefficient": coefficients[feature_index],
                }
            )
    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_df.to_csv(REPORT_DIR / "logreg_top_features_by_class.csv", index=False)

    # 3) XGBoost importance trên SVD + 7 handcrafted features.
    x_train_xgb, x_test_xgb, _, svd, xgb_feature_names = fit_transform_xgb_features(
        classical_text[train_idx],
        classical_text[test_idx],
        raw_text[train_idx],
        raw_text[test_idx],
        tfidf_config=tfidf_config,
        n_components=300,
        seed=args.seed,
    )
    selection_summary["svd_components"] = int(len(xgb_feature_names) - len(HANDCRAFTED_FEATURE_NAMES))
    selection_summary["svd_explained_variance_ratio"] = float(svd.explained_variance_ratio_.sum())
    selection_summary["xgboost_input_dimensions"] = int(x_train_xgb.shape[1])

    sample_weights = compute_sample_weight(class_weight="balanced", y=y[train_idx])
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        eval_metric="mlogloss",
        n_jobs=1,
    )
    xgb.fit(x_train_xgb, y[train_idx], sample_weight=sample_weights)
    importance_df = pd.DataFrame(
        {
            "feature": xgb_feature_names,
            "importance": xgb.feature_importances_,
            "feature_group": [
                "handcrafted" if name in HANDCRAFTED_FEATURE_NAMES else "svd_component"
                for name in xgb_feature_names
            ],
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(REPORT_DIR / "xgboost_feature_importance.csv", index=False)

    aggregate_rows = [
        {
            "feature": "all_svd_components",
            "importance": importance_df.loc[
                importance_df["feature_group"] == "svd_component", "importance"
            ].sum(),
        }
    ]
    for feature in HANDCRAFTED_FEATURE_NAMES:
        aggregate_rows.append(
            {
                "feature": feature,
                "importance": importance_df.loc[importance_df["feature"] == feature, "importance"].sum(),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("importance", ascending=False)
    aggregate_df.to_csv(REPORT_DIR / "xgboost_importance_aggregated.csv", index=False)

    # 4) Permutation importance cho từng handcrafted feature trên tập test.
    baseline_predictions = xgb.predict(x_test_xgb).astype(int)
    baseline_macro_f1 = f1_score(y[test_idx], baseline_predictions, average="macro", zero_division=0)
    rng = np.random.default_rng(args.seed)
    hc_start = x_test_xgb.shape[1] - len(HANDCRAFTED_FEATURE_NAMES)
    permutation_rows = []
    for offset, feature in enumerate(HANDCRAFTED_FEATURE_NAMES):
        column_index = hc_start + offset
        drops = []
        for repeat in range(args.permutation_repeats):
            permuted = x_test_xgb.copy()
            permuted[:, column_index] = rng.permutation(permuted[:, column_index])
            permuted_predictions = xgb.predict(permuted).astype(int)
            permuted_score = f1_score(
                y[test_idx], permuted_predictions, average="macro", zero_division=0
            )
            drops.append(baseline_macro_f1 - permuted_score)
        permutation_rows.append(
            {
                "feature": feature,
                "macro_f1_drop_mean": float(np.mean(drops)),
                "macro_f1_drop_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
                "repeats": args.permutation_repeats,
            }
        )
    permutation_df = pd.DataFrame(permutation_rows).sort_values(
        "macro_f1_drop_mean", ascending=False
    )
    permutation_df.to_csv(REPORT_DIR / "handcrafted_permutation_importance.csv", index=False)

    # Biểu đồ mặc định của Matplotlib, không ép màu để dễ tái sử dụng.
    top_plot = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top_plot["feature"], top_plot["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Đặc trưng")
    plt.title("Top 20 đặc trưng theo XGBoost")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_top20_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    perm_plot = permutation_df.sort_values("macro_f1_drop_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(perm_plot["feature"], perm_plot["macro_f1_drop_mean"])
    plt.xlabel("Mức giảm Macro-F1 sau hoán vị")
    plt.ylabel("Handcrafted feature")
    plt.title("Permutation importance của 7 đặc trưng thủ công")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "handcrafted_permutation_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    save_json(selection_summary, REPORT_DIR / "feature_selection_summary.json")
    markdown = f"""# Feature Selection và Feature Importance

## Feature selection

Dự án thực hiện **feature selection gián tiếp** thay vì dùng một bộ chọn đặc trưng độc lập:

- `min_df={tfidf_config.min_df}` loại n-gram xuất hiện quá hiếm.
- `max_df={tfidf_config.max_df}` loại n-gram quá phổ biến.
- `max_features={tfidf_config.max_features}` giới hạn số đặc trưng TF-IDF.
- Số đặc trưng thực tế sau lọc: **{x_train.shape[1]}**.
- TruncatedSVD giữ **{selection_summary['svd_components']}** thành phần, giải thích khoảng **{selection_summary['svd_explained_variance_ratio']:.2%}** phương sai trên tập train.
- XGBoost nhận các thành phần SVD kết hợp **7 handcrafted features**.

## Feature importance

- `logreg_top_features_by_class.csv`: hệ số Logistic Regression theo từng lớp.
- `xgboost_feature_importance.csv`: importance của từng thành phần đầu vào XGBoost.
- `xgboost_importance_aggregated.csv`: tổng importance của toàn bộ SVD so với từng handcrafted feature.
- `handcrafted_permutation_importance.csv`: mức giảm Macro-F1 khi hoán vị từng handcrafted feature trên tập test.

Feature importance phản ánh mối liên hệ mô hình học được, không chứng minh quan hệ nhân quả hoặc giá trị chẩn đoán.
"""
    (REPORT_DIR / "feature_selection_and_importance.md").write_text(markdown, encoding="utf-8")

    print("Đã lưu báo cáo feature selection/importance vào outputs/reports.")
    print("Số đặc trưng TF-IDF thực tế:", x_train.shape[1])
    print("Macro-F1 baseline XGBoost:", round(baseline_macro_f1, 4))
    print(permutation_df.round(4))


if __name__ == "__main__":
    main()
