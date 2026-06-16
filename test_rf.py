import pandas as pd
# Gọi hàm từ file động cơ của Lanh mà không cần sửa file đó
from modules.random_forest_engine import run_all_scenarios 

# 1. Đường dẫn tới file data sạch (tính từ thư mục gốc)
data_path = "data/laptops_dataset_cleaned.csv"

print(f"📂 Đang đọc dữ liệu sạch từ: {data_path}...")

try:
    df_clean = pd.read_csv(data_path)
    
    # 2. CHẠY HUẤN LUYỆN 8 CẤU HÌNH
    outputs = run_all_scenarios(df_clean)
    
    # 3. IN BẢNG XẾP HẠNG LEADERBOARD RA MÀN HÌNH TERMINAL
    print("\n📊 BẢNG XẾP HẠNG LEADERBOARD (Xếp theo MAE tăng dần):")
    print("=" * 95)
    print(outputs["summary_df"].to_string(index=False))
    print("=" * 95)
    
except Exception as e:
    print(f"❌ Lỗi khi thực thi: {e}")