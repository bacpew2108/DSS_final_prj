# DSS_final_prj

```text
DSS_Laptop_System/ 
│ 
├── analysis/                       
│   └── phan_tich_du_lieu_laptop.ipynb  # Notebook phân tích & EDA
│
├── assets/                         # Biểu đồ xuất ra từ quá trình EDA
│
├── data/                           
│   ├── laptops_dataset_raw.csv         # Dữ liệu gốc (Crawl)
│   └── laptops_dataset_cleaned.csv     # Dữ liệu đã qua tiền xử lý (ETL)
│
├── models/                         # Lưu trữ mô hình Machine Learning (Joblib)
│   └── rf_best_model.joblib            # Mô hình Random Forest đã được huấn luyện
│
├── modules/                        
│   ├── laptop_etl/                 # Package ETL xử lý & chuẩn hóa dữ liệu
│   │   ├── benchmark.py                # Lấy & match điểm CPU/GPU
│   │   ├── hardware.py                 # Chuẩn hóa Cân nặng, Màn hình, GPU, v.v.
│   │   ├── normalization.py            # Chuẩn hóa OS, Warranty
│   │   ├── pipeline.py                 # File điều phối Pipeline ETL
│   │   ├── price_currency.py           # Chuyển đổi ngoại tệ (EGP → VND)
│   │   ├── ram_storage.py              # Parse dung lượng RAM & Ổ cứng
│   │   └── utils.py                    # Utilities chuẩn hóa strings
│   ├── __init__.py                 
│   ├── charts.py                   # Thư viện vẽ biểu đồ Plotly/Radar trên UI
│   ├── random_forest_engine.py     # Module định nghĩa 8 kịch bản Random Forest
│   ├── topsis_engine.py            # Module toán học: Lọc cứng, AHP & TOPSIS
│   └── test.py                     # File Unit Test kiểm thử thuật toán Backend
│
├── app.py                          # File chạy chính - Giao diện Web Streamlit
├── train_model.py                  # Script huấn luyện & đóng gói mô hình ML (MLOps)
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
Việc tạo môi trường ảo giúp các thư viện của dự án không bị xung đột với các dự án khác trên máy của bạn.
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
*(Sau khi kích hoạt, bạn sẽ thấy chữ `(venv)` ở đầu dòng lệnh)*

**4. Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```
**5. Huấn luyện mô hình**
Hệ thống áp dụng chuẩn MLOps, tách biệt quá trình Training và Inference. Bạn cần chạy lệnh này để máy tính tự động huấn luyện mô hình Random Forest và lưu vào thư mục models/.
```bash
python train_model.py
```
**5. Khởi chạy ứng dụng**
```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8502` (hoặc `8501`).


Thoát venv: deactivate