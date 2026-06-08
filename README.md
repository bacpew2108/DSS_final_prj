# DSS_final_prj

```text
DSS_Laptop_System/ 
│ 
├── data/                       <-- (Nơi chứa dữ liệu - TV1 quản lý) 
│   ├── raw_data.csv            # File gốc 30 cột ban đầu chưa đụng chạm 
│   └── clean_data.csv          # File 6 cột số chuẩn hóa (Output của TV1) 
│ 
├── modules/                    <-- (Nơi chứa các hàm xử lý độc lập) 
│   ├── __init__.py             # File rỗng (để Python hiểu đây là thư mục code) 
│   ├── data_processing.py      # Code xử lý dữ liệu của TV1 
│   ├── topsis_engine.py        # Code thuật toán cốt lõi của TV2 
│   └── charts.py               # Code vẽ biểu đồ Plotly/Radar của TV3 
├── assets/                     <-- (Nơi chứa tài nguyên giao diện) 
│   └── logo.png                # Ảnh logo nhét vào web cho đẹp (TV4 quản lý)
│ 
├── notebooks/                  <-- (Vẽ biểu đồ Insight) 
│   └── EDA_phantic_insight.ipynb # File Jupyter Notebook (TV3 vẽ biểu đồ) 
├── app.py                      <-- (File chạy chính - UI Streamlit do TV4 code) 
├── requirements.txt            # Danh sách thư viện cần cài đặt 
└── README.md                   # File hướng dẫn  
```