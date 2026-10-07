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

**Lý do:** Lần 3 được chọn vì F1 0.7149 cao nhất và vượt ngưỡng 0.65. Lần 1 có accuracy cao nhất nhưng F1 thấp hơn, cho thấy accuracy chưa phản ánh tốt lớp thiểu số. Cấu hình yếu ở lần 2 làm cả hai chỉ số giảm; tăng số cây và độ sâu chỉ cải thiện F1 nhẹ nhưng giảm nhẹ accuracy.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Lớp thu nhập trên 50K chỉ chiếm 24,8%, nên mô hình luôn đoán “thu nhập thấp” vẫn đạt accuracy 75,2% dù không phát hiện lớp dương. F1 kết hợp precision và recall, đo cả độ chính xác lẫn khả năng không bỏ sót nhóm thu nhập cao. Lab tính trực tiếp cho `target=1`; weighted F1 không phù hợp vì bị lớp đa số chi phối, còn macro F1 coi hai lớp ngang nhau. Vì quality gate tập trung vào lớp dương, ngưỡng F1 0.65 có ý nghĩa hơn accuracy.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi dependency | Xung đột Setuptools/SQLAlchemy | Pin phiên bản tương thích và dùng venv riêng. |
| Không tạo được `sa-key.json` | Organization Policy cấm key | Dùng ADC, WIF và Service Account gắn VM. |
| SCP/JSON lỗi trên PowerShell | Khác biệt đường dẫn/dấu nháy | Tạo thư mục trước; escape JSON cho `curl.exe`. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.882 |

**Nhận xét:** Thêm `train_batch2` làm F1 tăng 0.0205 và accuracy tăng 0.008. Mức tăng vừa phải vì hai batch cùng phân phối.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - DagsHub: GitHub Actions ghi run `income-model` cùng tham số, metrics và model lên MLflow từ xa.
- [x] Bonus 2 - Điều chỉnh ngưỡng: quét 0.1–0.9, chọn 0.30; F1 tăng từ 0.7354 lên 0.7537.
- [x] Bonus 3 - Precision/recall: lưu confusion matrix và chỉ số từng lớp trong `detail.txt`; ưu tiên recall lớp cao để giảm bỏ sót.
- [x] Bonus 4 - Rollback: chỉ upload và restart khi F1 ứng viên không thấp hơn report production trên bucket.
- [x] Bonus 5 - Data drift: cảnh báo khi tỷ lệ lớp dương lệch quá 5% so với mốc 24,8%.
