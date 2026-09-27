# 📈 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC GIAO DỊCH KẾT HỢP EMA & OBV
> **Môn học:** Quản Lý Danh Mục Đầu Tư (Chương trình Thạc sĩ - HK3)  
> **Đơn vị thực hiện:** Nhóm 3  
> **Đối tượng phân tích:** Dữ liệu chuỗi thời gian cổ phiếu Ngân hàng TMCP Á Châu (**ACB**)  
> **Nền tảng triển khai:** [Streamlit](https://streamlit.io/) & [GitHub](https://github.com/)

---

## 📌 1. Giới Thiệu Tổng Quan

Dự án này chuyển đổi và nâng cấp toàn bộ mô hình nghiên cứu định lượng từ Jupyter Notebook (`ACB_Strategy_EMA_OBV_Nhom3_Chinhsua.ipynb`) thành một **Ứng dụng Web Tương tác (Streamlit Web App)** chuyên nghiệp.

Ứng dụng cho phép nhà đầu tư và giảng viên:
1. **Kiểm định thực nghiệm (Backtesting):** Đánh giá hiệu suất của 3 chiến lược:
   - Chiến lược theo xu hướng: **EMA Riêng lẻ** (Exponential Moving Average).
   - Chiến lược theo dòng tiền: **OBV Riêng lẻ** (On-Balance Volume Slope).
   - Chiến lược phối hợp: **EMA + OBV Kết hợp** (Giá trên EMA đồng thời Độ dốc OBV dương).
   - So sánh chuẩn với chiến lược thụ động **Mua & Nắm Giữ (Buy & Hold)**.
2. **Kiểm định Out-of-Sample (In-Sample vs Out-of-Sample):**
   - **Tập Train (2014 – 2020):** 1,743 phiên giao dịch (Giai đoạn huấn luyện & tối ưu hóa).
   - **Tập Test (2021 – 2023):** 748 phiên giao dịch (Giai đoạn kiểm định độ bền vững thực tế).
3. **Mô phỏng Quản trị Rủi ro & Chi phí giao dịch thực tế:**
   - Cơ chế cắt lỗ nghiêm ngặt (**Stop Loss 7%**).
   - Phí giao dịch (**0.2%**) và Trượt giá (**0.1%**).
   - Triệt tiêu hoàn toàn thiên lệch nhìn trước tương lai (**No Look-ahead Bias**) thông qua kỹ thuật `shift(1)`.
4. **Tối ưu hóa Tham số Định lượng:** Tích hợp thuật toán Bayesian Optimization (Hyperopt TPE) trực tiếp trên giao diện web.

---

## 📁 2. Cấu Trúc Thư Mục Dự Án

```text
├── app.py                                   # Mã nguồn chính của Web App Streamlit
├── requirements.txt                         # Danh sách thư viện Python tương thích
├── README.md                                # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv                                  # Dữ liệu giá & khối lượng lịch sử ACB (2014 - 2023)
└── ACB_Strategy_EMA_OBV_Nhom3_Chinhsua.ipynb # Notebook nghiên cứu gốc của Nhóm 3
```

---

## 🧠 3. Cơ Sở Lý Thuyết & Quy Tắc Chiến Lược

### 3.1. Chỉ báo Đường trung bình động hàm mũ (EMA)
- **Đặc điểm:** Gán trọng số giảm dần theo hàm mũ cho các phiên quá khứ, giúp phản ứng nhanh hơn với các biến động giá gần nhất so với SMA.
- **Quy tắc tín hiệu:**
  - **Mua (Buy):** Khi Giá đóng cửa vượt lên trên đường EMA: $\text{Close}_t > \text{EMA}_t$.
  - **Bán (Sell):** Khi Giá đóng cửa cắt xuống dưới đường EMA: $\text{Close}_t < \text{EMA}_t$.
  - *Áp dụng trễ 1 phiên:* $\text{Signal}_{t+1} = \text{Condition}_t$ để đảm bảo tính khả thi trong thực tế.

### 3.2. Chỉ báo Khối lượng cân bằng (OBV) & Độ dốc OBV (OBV Slope)
- **Đặc điểm:** Theo lý thuyết của Joseph Granville, *"Khối lượng luôn đi trước giá"*. OBV cộng dồn khối lượng khi giá tăng và trừ khối lượng khi giá giảm.
- **Độ dốc OBV ($\Delta \text{OBV}$):** Đo lường mức biến động của OBV trong $N$ phiên gần nhất:
  $$\Delta \text{OBV}_t = \text{OBV}_t - \text{OBV}_{t - N}$$
- **Quy tắc tín hiệu:**
  - **Mua:** $\Delta \text{OBV}_t > 0$ (Áp lực mua chủ động chiếm ưu thế).
  - **Bán:** $\Delta \text{OBV}_t < 0$ (Áp lực phân phối/xả hàng chiếm ưu thế).

### 3.3. Chiến lược Phối hợp EMA + OBV
- **Triết lý:** Sự kết hợp hoàn hảo giữa **Động lượng Xu hướng (Trend)** và **Xác nhận Dòng tiền (Volume Confirmation)**.
- **Quy tắc vào lệnh:** Mua khi và chỉ khi **CẢ HAI** điều kiện đồng thời thỏa mãn:
  $$\text{Entry}: (\text{Close}_t > \text{EMA}_t) \quad \mathbf{VÀ} \quad (\Delta \text{OBV}_t > 0)$$
- **Quy tắc thoát lệnh:** Bán khi giá vi phạm xu hướng:
  $$\text{Exit}: (\text{Close}_t < \text{EMA}_t) \quad \mathbf{HOẶC} \quad (\text{Thua lỗ chạm ngưỡng Stop Loss } 7\%)$$

---

## 📊 4. Tóm Tắt Kết Quả Kiểm Định (Train vs Test)

Bảng tổng hợp từ quá trình tối ưu hóa 1,066 lần thử nghiệm (Hyperopt TPE) trong Notebook:

| Chiến Lược | Tập Dữ Liệu | Tham Số Tối Ưu | Sharpe Ratio | Tổng Lợi Nhuận (%) | Max Drawdown (%) | Số Lệnh Đã Đóng |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **EMA Riêng lẻ** | **Train (2014-2020)** | EMA=36 | **1.0797** | **211.97%** | -28.69% | 70 |
| **EMA Riêng lẻ** | **Test (2021-2023)** | EMA=36 | -0.8601 | -37.69% | -59.19% | 41 |
| **OBV Riêng lẻ** | **Train (2014-2020)** | OBV_Slope=19 | **1.1735** | **277.80%** | -30.46% | 51 |
| **OBV Riêng lẻ** | **Test (2021-2023)** | OBV_Slope=19 | -0.6135 | -38.73% | -62.50% | 38 |
| **EMA + OBV Kết hợp** | **Train (2014-2020)** | EMA=36, OBV_Slope=20 | **1.1514** | **236.56%** | **-23.87%** (Thấp nhất) | 64 |
| **EMA + OBV Kết hợp** | **Test (2021-2023)** | EMA=36, OBV_Slope=20 | -0.8298 | **-36.47%** (Tốt nhất) | -60.14% | 37 |

### 💡 Bài học Kinh tế & Quản lý Danh mục:
1. **Hiện tượng Quá mức Khớp Dữ liệu (Overfitting):** Giai đoạn 2014–2020 là một đại chu kỳ tăng trưởng dài hạn (Bull Market) của thị trường chứng khoán Việt Nam, giúp các chỉ báo kỹ thuật đạt tỷ suất vượt trội. Đến giai đoạn 2021–2023, thị trường trải qua biến động dữ dội và đợt suy thoái diện rộng năm 2022, khiến hiệu suất out-of-sample bị giảm sút.
2. **Giá trị của việc Phối hợp Chỉ báo:** Chiến lược phối hợp **EMA + OBV** kiểm soát mức sụt giảm tối đa trong tập Train tốt nhất (**-23.87%**) và hạn chế tối đa mức thua lỗ trong tập Test (**-36.47%**) so với việc chỉ sử dụng một chỉ báo riêng lẻ.

---

## 💻 5. Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### Bước 1: Yêu cầu môi trường
- Đã cài đặt **Python 3.9, 3.10 hoặc 3.11**.

### Bước 2: Cài đặt các thư viện phụ thuộc
Mở terminal/command prompt tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi động ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🚀 6. Hướng Dẫn Đẩy Code Lên GitHub

Thực hiện các lệnh sau trong terminal tại thư mục chứa dự án:

```bash
# 1. Khởi tạo kho lưu trữ git (nếu chưa có)
git init

# 2. Thêm tất cả các file vào khu vực chờ commit
git add app.py requirements.txt README.md ACB.csv

# 3. Tạo commit đầu tiên
git commit -m "feat: Khoi tao ung dung Streamlit kiem dinh chien luoc EMA va OBV cho co phieu ACB"

# 4. Đổi tên nhánh chính thành main
git branch -M main

# 5. Liên kết với kho lưu trữ trên GitHub (thay link repo của bạn vào bên dưới)
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/<ten-repository>.git

# 6. Đẩy toàn bộ mã nguồn lên GitHub
git push -u origin main
```

---

## 🌐 7. Hướng Dẫn Triển Khai (Deploy) Lên Streamlit Community Cloud

Streamlit cung cấp dịch vụ hosting miễn phí vĩnh viễn, cực kỳ ổn định và nhanh chóng:

1. Đăng nhập vào [Streamlit Community Cloud](https://share.streamlit.io/) bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** (hoặc **"New app"**).
3. Điền thông tin cấu hình:
   - **Repository:** Chọn kho lưu trữ bạn vừa push ở Bước 6.
   - **Branch:** `main`.
   - **Main file path:** `app.py`.
   - **App URL (tùy chọn):** Đặt tên miền con theo ý muốn (ví dụ: `acb-ema-obv-strategy.streamlit.app`).
4. Nhấn nút **"Deploy!"**.
5. Đợi 1-2 phút để máy chủ tự động cài đặt các thư viện trong `requirements.txt`. Khi xuất hiện bóng bay 🎈, ứng dụng của bạn đã chính thức hoạt động công khai trên toàn cầu!

---

## ✨ 8. Các Tính Năng Nổi Bật Của Ứng Dụng

- [x] **Tải dữ liệu linh hoạt:** Tự động nhận diện `ACB.csv` hoặc hỗ trợ tải lên file CSV của bất kỳ cổ phiếu nào khác (VCB, HPG, FPT, MWG,...).
- [x] **Giao diện tương tác cao cấp:** Biểu đồ Plotly động, hỗ trợ phóng to/thu nhỏ, rê chuột xem giá trị chi tiết (hover tooltips).
- [x] **Bộ tham số cấu hình nhanh:** Dễ dàng chuyển đổi giữa bộ tham số mặc định và bộ tham số tối ưu chỉ bằng 1 cú click.
- [x] **Nhật ký giao dịch chi tiết (Trade Log):** Hiển thị chi tiết từng lệnh mua/bán, giá vào/ra, tỷ lệ lời/lỗ và lý do đóng lệnh (chạm Stop Loss hay do Tín hiệu đảo chiều); hỗ trợ tải file CSV về máy.
- [x] **Sandbox Tối ưu hóa Hyperopt:** Khám phá không gian tham số và vẽ biểu đồ nhiệt (Heatmap) ngay trên giao diện web.
