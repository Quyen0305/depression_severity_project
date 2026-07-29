# Các thay đổi đã thực hiện

## 1. Đồng nhất điều kiện đánh giá

- Thêm `src/run_unified_benchmark.py` để năm mô hình dùng cùng train/test indices theo từng seed.
- Thêm kết quả mean, std, min và max qua nhiều lần chia.
- Loại bỏ cách nhập cứng kết quả trong pipeline mới.
- TF-IDF, TruncatedSVD và vocabulary chỉ được fit trên train của từng split.

## 2. Đánh giá DistilBERT nhiều lần

- Thêm `src/evaluate_distilbert.py`.
- Hỗ trợ `repeated_holdout` với nhiều seed.
- Hỗ trợ `kfold` với Stratified K-Fold.
- Mỗi split/fold khởi tạo lại mô hình từ checkpoint tiền huấn luyện.

## 3. Tuning DistilBERT

- Thêm `src/tune_distilbert.py`.
- Giữ riêng test set, chỉ chọn tham số trên validation.
- Tuning learning rate, epoch và max length.
- Chọn theo Macro-F1, dùng QWK làm tiêu chí phụ.

## 4. Khả năng tái lập

- Thêm `set_global_seed()` cho Python, NumPy và PyTorch.
- Bật deterministic mode khi có thể.
- DataLoader dùng generator có seed.
- Ghi seed và cấu hình vào metadata JSON.

## 5. Feature selection và importance

- Gọi rõ feature selection gián tiếp qua `min_df`, `max_df`, `max_features` và TruncatedSVD.
- Thêm top coefficients của Logistic Regression.
- Thêm XGBoost feature importance.
- Thêm permutation importance của 7 handcrafted features.

## 6. Dữ liệu thiếu, outlier và fairness

- Thêm báo cáo missing/blank values.
- Gắn cờ outlier độ dài bằng IQR nhưng không tự động xóa.
- Bổ sung phân bố nhãn và performance slicing theo độ dài nếu có predictions.
- Nêu rõ không thể tính fairness nhân khẩu học do dataset thiếu protected attributes.

## 7. Tài liệu và môi trường

- Thêm `README.md`, `requirements.txt`, `.gitignore`.
- Thêm notebook `08_distilbert_repeated_evaluation.ipynb` và `12_feature_importance_fairness.ipynb`.
- Bổ sung các cell liên quan vào notebook 04, 06, 09 và 10.
- Loại `.venv` khỏi bản đóng gói mới.
