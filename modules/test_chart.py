import pandas as pd
import io
import sys
from pathlib import Path

# ==========================================================
# BỘ GỠ LỖI ĐƯỜNG DẪN TỰ ĐỘNG (CHỐNG LỖI IMPORT)
# ==========================================================
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parents[1]      
MODULES_DIR = FILE_PATH.parent       

for path in [ROOT_DIR, MODULES_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from charts import plot_radar_chart 

# ==========================================================
# DỮ LIỆU GIẢ LẬP (ĐÃ NÂNG LÊN 5 MÁY ĐA DẠNG CẤU HÌNH)
# ==========================================================
csv_data = """product_name,price,ram_capacity,storage,cpu_point,gpu_point,weight
MSI Titan 18 HX (Bản lỗi giá hời),16.0,96,10240,83.0,67.0,3.6
Acer NITRO V15 (Gaming phổ thông),20.2,8,512,50.0,33.0,1.8
ASUS TUF Gaming A14 (Mỏng nhẹ gaming),24.7,16,512,61.0,33.0,1.5
HP Pavilion 14 (Văn phòng siêu nhẹ),15.5,8,256,48.0,15.0,1.2
Lenovo ThinkPad E16 (Máy trạm nhiều RAM),22.0,24,1024,58.0,22.0,1.9"""

if __name__ == "__main__":
    print("1. Đang khởi tạo dữ liệu giả lập (5 Laptop)...")
    df_mock = pd.read_csv(io.StringIO(csv_data))
    print(df_mock.to_string(index=False))
    
    # KÍCH HOẠT VẼ CẢ 5 MÁY ĐỂ XEM ĐỦ 5 DẢI MÀU
    SO_LUONG_VE = 5 
    PHAN_KHUC_TEST = "Phổ thông (15 - 25 Triệu)"
    
    print(f"\n2. Đang gọi hàm vẽ từ modules/charts.py (Vẽ Top {SO_LUONG_VE} máy)...")
    try:
        fig = plot_radar_chart(
            top_df=df_mock, 
            segment_name=PHAN_KHUC_TEST, 
            name_col='product_name',
            top_n=SO_LUONG_VE
        )
        
        print("3. Đang mở trình duyệt để hiển thị kết quả biểu đồ...")
        fig.show()
        print("🎉 Chạy test vẽ 5 máy thành công!")
        
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi vẽ biểu đồ: {e}")