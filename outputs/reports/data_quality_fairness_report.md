# Data Quality, Outlier và Fairness Audit

## Dữ liệu thiếu

- Số dòng: **3519**.
- Báo cáo chi tiết: `missing_data_report.csv`.
- Nhãn không hợp lệ: **không có**.
- Số dòng còn trùng nội dung trong file đầu vào: **0**.

## Outlier độ dài văn bản

- Q1 = **65.00** từ; Q3 = **101.00** từ; IQR = **36.00**.
- Ngưỡng IQR: [11.00, 155.00] từ.
- Số mẫu được gắn cờ outlier: **112**.
- Các mẫu này **không bị xóa tự động** vì bài đăng dài hoặc ngắn vẫn có thể là dữ liệu hợp lệ. Với BiLSTM/DistilBERT, chuỗi được padding hoặc truncation theo `max_length`.

## Fairness

Dataset không cung cấp thuộc tính bảo vệ như tuổi, giới tính, dân tộc, khu vực hoặc nhóm văn hóa. Vì vậy, dự án **không thể** tính các thước đo fairness nhân khẩu học như demographic parity, equal opportunity hoặc equalized odds.

Các rủi ro còn lại:

1. Dữ liệu Reddit tiếng Anh không đại diện cho toàn bộ cộng đồng hoặc người dùng Việt Nam.
2. Người chủ động đăng bài trên Reddit có thể khác với người không sử dụng mạng xã hội.
3. Mất cân bằng nhãn có thể làm giảm Recall của các lớp thiểu số.
4. Phân tích theo độ dài văn bản chỉ là phân tích lát cắt hiệu năng, không phải bằng chứng fairness theo nhân khẩu học.

Đã tạo performance_by_text_length_slice.csv từ kết quả benchmark thống nhất.
