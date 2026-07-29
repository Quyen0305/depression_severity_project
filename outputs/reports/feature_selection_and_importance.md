# Feature Selection và Feature Importance

## Feature selection

Dự án thực hiện **feature selection gián tiếp** thay vì dùng một bộ chọn đặc trưng độc lập:

- `min_df=2` loại n-gram xuất hiện quá hiếm.
- `max_df=0.9` loại n-gram quá phổ biến.
- `max_features=3000` giới hạn số đặc trưng TF-IDF.
- Số đặc trưng thực tế sau lọc: **3000**.
- TruncatedSVD giữ **300** thành phần, giải thích khoảng **45.07%** phương sai trên tập train.
- XGBoost nhận các thành phần SVD kết hợp **7 handcrafted features**.

## Feature importance

- `logreg_top_features_by_class.csv`: hệ số Logistic Regression theo từng lớp.
- `xgboost_feature_importance.csv`: importance của từng thành phần đầu vào XGBoost.
- `xgboost_importance_aggregated.csv`: tổng importance của toàn bộ SVD so với từng handcrafted feature.
- `handcrafted_permutation_importance.csv`: mức giảm Macro-F1 khi hoán vị từng handcrafted feature trên tập test.

Feature importance phản ánh mối liên hệ mô hình học được, không chứng minh quan hệ nhân quả hoặc giá trị chẩn đoán.
