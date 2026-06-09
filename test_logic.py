import pandas as pd
from modules.topsis_engine import calculate_topsis

# ========================================================
# CẤU HÌNH PANDAS ĐỂ IN RA TERMINAL KHÔNG BỊ XUỐNG DÒNG
# ========================================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ========================================================
# BƯỚC 1: LOAD DỮ LIỆU TỪ FILE CSV
# ========================================================
csv_path = 'data/laptops_dataset_cleaned.csv'

try:
    df_real = pd.read_csv(csv_path)
    print(f"✅ Đã load thành công {len(df_real)} chiếc laptop từ file '{csv_path}'.")
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file! Hãy kiểm tra xem file đã nằm đúng ở đường dẫn '{csv_path}' chưa.")
    exit()

# ========================================================
# BƯỚC 2: GIẢ LẬP TRỌNG SỐ NGƯỜI DÙNG
# ========================================================
# Thứ tự chuẩn: [Giá, RAM, Ổ cứng, CPU, GPU, Nặng]
mock_weights = [0.4, 0.1, 0.1, 0.2, 0.1, 0.1]

# ========================================================
# BƯỚC 3: GỌI HÀM TOPSIS TỪ MODULE CỦA BẠN
# ========================================================
print("\n⏳ Đang chạy thuật toán TOPSIS...")

try:
    # Lấy ra Top 5 máy tốt nhất
    print("Nhập số lượng laptop muốn xem (mặc định là 5): ", end="")
    top_n = input()
    if top_n.isdigit():
        top_n = int(top_n)
    else:
        top_n = 5
    top_laptops = calculate_topsis(df=df_real, weights_array=mock_weights, top_n=top_n)
    
    # ========================================================
    # BƯỚC 4: IN KẾT QUẢ RA MÀN HÌNH
    # ========================================================
    print("\n🏆 --- TOP 5 LAPTOP TỐT NHẤT DÀNH CHO BẠN --- 🏆\n")
    
    # Đưa toàn bộ 6 tiêu chí quan trọng và Điểm số vào danh sách hiển thị
    columns_to_show = [
        'product_name', 
        'price', 
        'ram_capacity', 
        'storage', 
        'cpu_point', 
        'gpu_point', 
        'weight', 
        'TOPSIS_Score'
    ]
    
    # In ra bảng (index=False để bỏ đi cột số thứ tự 0, 1, 2... cho gọn)
    print(top_laptops[columns_to_show].to_string(index=False))
    
    print("\n✅ Thuật toán hoạt động hoàn hảo!")

except KeyError as e:
    print(f"\n❌ LỖI TÊN CỘT: Cột {e} không tồn tại trong file CSV!")
    print("👉 Báo lại TV1: Hãy đảm bảo file CSV có đủ các cột: 'price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight' và 'product_name'.")
except Exception as e:
    print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH: {e}")