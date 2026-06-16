# DSS_final_prj

```text
DSS_Laptop_System/ 
│ 
├── analysis/                       
│   └── phan_tich_du_lieu_laptop.ipynb  # Notebook phân tích & EDA
│
├── assets/                         # Biểu đồ xuất từ EDA
│
├── data/                           
│   ├── laptops_dataset_raw.csv         # Data gốc
│   └── laptops_dataset_cleaned.csv     # Data chuẩn hóa
│
├── modules/                        
│   ├── laptop_etl/                 # Package xử lý dữ liệu
│   │   ├── __init__.py             
│   │   ├── benchmark.py                # Fetch & match điểm CPU/GPU
│   │   ├── hardware.py                 # Weight, màu sắc, brand, màn hình, GPU memory
│   │   ├── normalization.py            # Chuẩn hóa warranty & hệ điều hành
│   │   ├── pipeline.py                 # check_duplicates, fill_fields, clean_laptops_csv
│   │   ├── price_currency.py           # Chuyển đổi giá EGP → VND
│   │   ├── ram_storage.py              # Parse dung lượng RAM & ổ cứng
│   │   └── utils.py                    # Chuẩn hóa tên cột và giá trị
│   ├── __init__.py                 
│   ├── charts.py                   # Code vẽ biểu đồ Plotly/Radar
│   ├── laptop_etl.py               # Xử lý dữ liệu laptop
│   ├── test.py                     # Unit test
│   ├── test_ve.py                  # Test vẽ biểu đồ
│   └── topsis_engine.py            # Thuật toán TOPSIS
│
├── app.py                          # File chạy chính - UI Streamlit
├── README.md                       
└── requirements.txt                # Danh sách thư viện cần cài đặt
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

**5. Khởi chạy ứng dụng**
```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8502` (hoặc `8501`).