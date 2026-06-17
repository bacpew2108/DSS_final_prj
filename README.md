# Hệ Thống Hỗ Trợ Quyết Định Cấu Hình Laptop (DSS Final Project)

Dự án này là một Hệ Thống Hỗ Trợ Quyết Định (DSS) nhằm giúp người dùng lựa chọn cấu hình laptop phù hợp với nhu cầu. Hệ thống tích hợp các phương pháp xử lý dữ liệu (ETL), mô hình học máy (Machine Learning) như Random Forest để dự đoán giá, và phương pháp ra quyết định đa tiêu chí (TOPSIS/AHP) để gợi ý laptop tốt nhất.

## Cấu trúc thư mục (Project Structure)

```text
DSS_Laptop_System/ 
│ 
├── analysis/                       
│   └── phan_tich_du_lieu_laptop.ipynb  # Notebook phân tích & EDA
│
├── assets/                         # Thư mục chứa tài nguyên, biểu đồ
│
├── data/                           
│   ├── laptops_dataset_raw.csv         # Dữ liệu gốc (Crawl)
│   └── laptops_dataset_cleaned.csv     # Dữ liệu đã qua tiền xử lý (ETL)
│
├── models/                         # Lưu trữ mô hình Machine Learning (Joblib)
│   └── best_model.joblib               # Mô hình học máy tốt nhất đã được huấn luyện
│
├── modules/                        
│   ├── laptop_etl/                 # Package ETL xử lý & chuẩn hóa dữ liệu
│   │   ├── benchmark.py                # Lấy & match điểm CPU/GPU
│   │   ├── hardware.py                 # Chuẩn hóa Cân nặng, Màn hình, GPU, v.v.
│   │   ├── normalization.py            # Chuẩn hóa OS, Warranty
│   │   ├── pipeline.py                 # File điều phối Pipeline ETL
│   │   ├── price_currency.py           # Chuyển đổi ngoại tệ
│   │   ├── ram_storage.py              # Parse dung lượng RAM & Ổ cứng
│   │   └── utils.py                    # Utilities chuẩn hóa strings
│   ├── __init__.py                 
│   ├── charts.py                   # Thư viện vẽ biểu đồ Plotly/Radar trên UI
│   ├── model_engine.py             # Module định nghĩa mô hình (Random Forest, v.v.)
│   ├── model_gen_charts.py         # Module vẽ biểu đồ cho các mô hình đánh giá
│   └── topsis_engine.py            # Module toán học: Lọc cứng, AHP & TOPSIS
│
├── app.py                          # File chạy chính - Giao diện Web Streamlit
├── train_model.py                  # Script huấn luyện & đóng gói mô hình ML (MLOps)
├── run_etl.py                      # Script chạy tiến trình làm sạch dữ liệu (ETL)
├── README.md                       
└── requirements.txt                # Danh sách thư viện cần thiết
```

## Hướng dẫn cài đặt và chạy ứng dụng

**1. Clone dự án và di chuyển vào thư mục**
```bash
# (Giả định bạn đã tải source code về máy)
cd DSS_final_prj
```

**2. Tạo môi trường ảo (Virtual Environment)**
Việc tạo môi trường ảo giúp các thư viện của dự án không bị xung đột với các thư viện khác trên máy của bạn.
```bash
python3 -m venv venv
```

**3. Kích hoạt môi trường ảo**
- Trên **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```
- Trên **Windows** (Command Prompt):
  ```cmd
  venv\Scripts\activate.bat
  ```
- Trên **Windows** (PowerShell):
  ```powershell
  venv\Scripts\Activate.ps1
  ```
*(Sau khi kích hoạt, bạn sẽ thấy chữ `(venv)` xuất hiện ở đầu dòng lệnh trong terminal)*

**4. Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

**5. Huấn luyện mô hình (Tuỳ chọn nhưng khuyến nghị)**
Hệ thống áp dụng chuẩn MLOps, tách biệt quá trình Training và Inference. Bạn có thể chạy lệnh sau để máy tính tự động huấn luyện lại mô hình học máy (ví dụ: Random Forest) và lưu vào thư mục `models/`.
```bash
python train_model.py
```

**6. Khởi chạy ứng dụng Web**
Dự án sử dụng Streamlit làm giao diện Web UI trực quan. Chạy lệnh sau để khởi động:
```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501` (hoặc `8502`).

**7. Thoát môi trường ảo**
Khi bạn đã hoàn thành công việc, có thể tắt môi trường ảo bằng lệnh:
```bash
deactivate
```