# --- ĐOẠN CODE TEST (Viết dưới hàm draw_radar_chart) ---
import pandas as pd
from charts import draw_radar_chart
if __name__ == "__main__":
    # 1. Tạo Fake Data (Tự bịa ra 3 cái máy ảo có tên cột giống hệt của Lanh)
    fake_data = {
        'price': [20000000, 15000000, 30000000],  # Máy 2 rẻ nhất -> Sẽ được 100 điểm giá
        'ram_capacity': [16, 8, 32],              # Máy 3 RAM to nhất -> Được 100 điểm RAM
        'weight': [1.5, 2.0, 1.2],                # Máy 3 nhẹ nhất -> Được 100 điểm nặng
        'storage': [512, 256, 1024],
        'cpu_point': [15000, 10000, 22000],
        'gpu_point': [5000, 3000, 8000]
    }
    
    # Biến nó thành DataFrame giả
    df_fake = pd.DataFrame(fake_data)
    
    # 2. Gọi hàm của bạn
    bieu_do = draw_radar_chart(df_fake)
    
    # 3. Bật trình duyệt lên để xem thành quả
    bieu_do.show()
    