# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phùng Gia Bảo |
| MSSV | 2A202602386 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/fubao1tuoi/K4-L3L4-Track2-Day21-PhungGiaBao-2A202602386-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 được chọn vì `f1_score=0.7149` cao nhất và vượt ngưỡng 0.65. Lần chạy 1 có accuracy cao nhất (0.878) nhưng F1 thấp hơn (0.7109), chứng tỏ accuracy chưa phản ánh đầy đủ khả năng nhận diện lớp thu nhập cao. Khi giảm số cây, tốc độ học và độ sâu ở lần 2, cả hai chỉ số đều giảm. Tăng từ 100 lên 200 cây và độ sâu từ 3 lên 5 chỉ cải thiện F1 nhẹ nhưng làm accuracy giảm nhẹ, thể hiện sự đánh đổi giữa nhận diện lớp dương và hiệu quả tổng thể.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Lớp thu nhập trên 50K chỉ chiếm 24,8% dữ liệu, nên mô hình luôn dự đoán “thu nhập thấp” vẫn đạt accuracy khoảng 75,2% dù không phát hiện lớp dương. Accuracy vì vậy dễ gây hiểu nhầm. F1 kết hợp precision và recall, phản ánh cả độ chính xác khi dự đoán thu nhập cao và khả năng không bỏ sót nhóm này. Lab dùng `f1_score(y_eval, preds)` cho `target=1`, không dùng weighted F1 vì lớp đa số sẽ chi phối kết quả. Macro F1 phù hợp khi cần coi hai lớp ngang nhau, còn quality gate này tập trung trực tiếp vào lớp thu nhập cao. Vì vậy ngưỡng `f1_score >= 0.65` phù hợp hơn accuracy.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi dependency | MLflow 2.13 xung đột với Setuptools 82+ và SQLAlchemy 2.1 | Pin `setuptools==80.9.0`, `SQLAlchemy==2.0.54`; dùng venv riêng. |
| Không tạo được `sa-key.json` | Organization Policy cấm Service Account key | Dùng ADC, Workload Identity Federation và Service Account gắn vào VM. |
| SCP/POST JSON lỗi trên PowerShell | Thiếu thư mục đích và cách xử lý dấu nháy khác Bash | Tạo thư mục trước; dùng `curl.exe` với JSON đã escape. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.882 |

**Nhận xét:** Sau khi bổ sung `train_batch2`, F1 tăng 0.0205 và accuracy tăng 0.008. Dữ liệu lớn hơn giúp nhận diện lớp thu nhập cao tốt hơn, nhưng mức tăng vừa phải vì hai batch cùng phân phối.
