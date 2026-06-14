import pandas as pd
import io
# Import trực tiếp hàm vẽ từ thư mục modules của bạn
from charts import plot_radar_chart 

# Dữ liệu mẫu (Lấy từ file CSV thực tế của bạn)
csv_data = """product_name,price,ram_capacity,storage,cpu_point,gpu_point,weight
MSI Titan 18 HX Dragon Edition,162.0,96,10240,83.0,67.0,3.6
Acer NITRO V15-ANV15,20.2,8,512,50.0,33.0,1.8
ASUS TUF Gaming A14 FA401UU,24.7,16,512,61.0,33.0,1.5"""

if __name__ == "__main__":
    print("1. Đang khởi tạo dữ liệu ảo...")
    df_mock = pd.read_csv(io.StringIO(csv_data))
    
    print("2. Đang gọi hàm vẽ từ modules/charts.py...")
    # Hàm này sẽ chạy code nằm trong file modules/charts.py của bạn
    fig = plot_radar_chart(df_mock, name_col='product_name')
    
    print("3. Đang mở trình duyệt để hiển thị kết quả biểu đồ...")
    fig.show()