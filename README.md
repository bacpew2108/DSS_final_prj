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