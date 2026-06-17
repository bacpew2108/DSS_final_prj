import os
import numpy as np
import pandas as pd

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN TỚI FILE CSV
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'laptops_dataset_cleaned.csv')

# =====================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ BACKEND (LỌC, AHP, TOPSIS)
# =====================================================================

def map_segment_to_budget(segment_name):
    """Ánh xạ tên Phân khúc sang Ngân sách tối đa (max_price tính bằng Triệu VNĐ)"""
    if segment_name == "Rẻ (< 15 Triệu)":
        return 15.0
    elif segment_name == "Phổ thông (15 - 25 Triệu)":
        return 25.0
    elif segment_name == "Cao cấp (> 25 Triệu)":
        return 999.0  # Đại diện cho không giới hạn / mua mọi giá
    else:
        return 25.0

def apply_hard_filters(df, max_price, min_ram, min_storage, selected_brands=None):
    """Bộ lọc cứng: Loại bỏ các laptop không đạt yêu cầu bắt buộc trước khi chạy TOPSIS"""
    filtered_df = df[
        (pd.to_numeric(df['price'], errors='coerce') <= max_price) & 
        (pd.to_numeric(df['ram_capacity'], errors='coerce') >= min_ram) & 
        (pd.to_numeric(df['storage'], errors='coerce') >= min_storage)
    ].copy()
    
    if selected_brands and len(selected_brands) > 0:
        filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
    return filtered_df

def calculate_ahp_weights_with_cr(matrix):
    """Tính trọng số AHP từ ma trận so sánh cặp và kiểm tra Tỷ số nhất quán (CR)."""
    n = matrix.shape[0]
    col_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)
    
    weighted_sum_vector = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum_vector / weights)
    
    ci = (lambda_max - n) / (n - 1)
    ri_dict = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32}
    ri = ri_dict.get(n, 1.24) 
    
    cr = ci / ri if ri > 0 else 0
    is_consistent = cr < 0.1
    
    return weights, cr, is_consistent

def calculate_topsis(df, weights_array, top_n=5):
    """Thuật toán TOPSIS xếp hạng phương án."""
    result_df = df.copy()
    # THỨ TỰ THUỘC TÍNH: [Giá, RAM, Ổ cứng, CPU, GPU, Nặng]
    criteria_cols = ['price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight']
    
    for col in criteria_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
    matrix = result_df[criteria_cols].values
    impacts = ['min', 'max', 'max', 'max', 'max', 'min']
    
    # 1. Chuẩn hóa Vector
    sqrt_sum_sq = np.sqrt(np.sum(matrix**2, axis=0))
    sqrt_sum_sq[sqrt_sum_sq == 0] = 1e-10 
    normalized_matrix = matrix / sqrt_sum_sq

    # 2. Nhân trọng số
    weighted_matrix = normalized_matrix * weights_array

    # 3. Xác định Cực dương (Best) và Cực âm (Worst)
    ideal_best = np.zeros(len(criteria_cols))
    ideal_worst = np.zeros(len(criteria_cols))
    for i in range(len(criteria_cols)):
        if impacts[i] == 'max':
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        elif impacts[i] == 'min':
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    # 4. Tính khoảng cách và điểm C_i
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
    
    sum_dist = dist_best + dist_worst
    sum_dist[sum_dist == 0] = 1e-10
    topsis_score = dist_worst / sum_dist

    result_df['TOPSIS_Score'] = np.round(topsis_score, 4)
    result_df = result_df.sort_values(by='TOPSIS_Score', ascending=False).reset_index(drop=True)
    return result_df.head(top_n)

def check_outlier_alert(top_1_laptop_price, outlier_threshold=51.4):
    """Cảnh báo nếu sản phẩm có giá vượt ngưỡng dị biệt (Outlier)"""
    if top_1_laptop_price > outlier_threshold:
        return f"⚠️ CẢNH BÁO: Sản phẩm Top 1 có mức giá siêu cao cấp ({top_1_laptop_price}). Khuyên dùng cho dân chuyên nghiệp!"
    return ""

# --- CÁC HÀM XUẤT TRỌNG SỐ CHO 3 CHẾ ĐỘ ---
def get_weights_mode_1(user_profile):
    """Chế độ 1: Trọng số tĩnh lấy từ chuyên gia (0 slider)"""
    profiles = {
        'student':   np.array([0.352, 0.119, 0.058, 0.165, 0.058, 0.248]),
        'gamer':     np.array([0.103, 0.134, 0.061, 0.264, 0.395, 0.043]),
        'developer': np.array([0.146, 0.222, 0.122, 0.380, 0.048, 0.082])
    }
    return profiles.get(user_profile, profiles['student'])

def get_weights_mode_2(user_profile, scores_dict):
    """Chế độ 2: Tùy chỉnh thu nhỏ (Chuyển điểm 1-10 của 4 thuộc tính thành vector trọng số)"""
    weights = np.zeros(6)
    if user_profile == 'student':
        weights[0], weights[1], weights[3], weights[5] = scores_dict.get('price',0), scores_dict.get('ram',0), scores_dict.get('cpu',0), scores_dict.get('weight',0)
    elif user_profile == 'gamer':
        weights[0], weights[1], weights[3], weights[4] = scores_dict.get('price',0), scores_dict.get('ram',0), scores_dict.get('cpu',0), scores_dict.get('gpu',0)
    elif user_profile == 'developer':
        weights[0], weights[1], weights[2], weights[3] = scores_dict.get('price',0), scores_dict.get('ram',0), scores_dict.get('storage',0), scores_dict.get('cpu',0)
        
    total_score = np.sum(weights)
    if total_score > 0:
        weights = weights / total_score
    else:
        weights = get_weights_mode_1(user_profile)
    return weights

def get_weights_mode_3(ahp_matrix_6x6):
    """Chế độ 3: Tính từ Ma trận so sánh cặp đầy đủ 6x6 của người dùng"""
    weights, cr, is_consistent = calculate_ahp_weights_with_cr(ahp_matrix_6x6)
    return weights, cr, is_consistent


# =====================================================================
# PHẦN 2: CHẠY KIỂM THỬ VỚI DỮ LIỆU THẬT
# =====================================================================
if __name__ == "__main__":
    print("\n⏳ Đang load dữ liệu từ Database...")
    try:
        df_laptops = pd.read_csv(csv_path)
        print(f"✅ Đã load thành công {len(df_laptops)} mẫu laptop!")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file CSV tại đường dẫn: {csv_path}")
        exit()

    # Kiểm tra xem giá trong file CSV là đơn vị Triệu (VD: 25.5) hay đơn vị đồng (VD: 25500000)
    is_price_in_millions = df_laptops['price'].max() < 1000
    multiplier = 1 if is_price_in_millions else 1_000_000

    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("▶️ TEST CHẾ ĐỘ 1: SINH VIÊN (GỢI Ý NHANH)")
    # Giả lập UI: User chọn Đối tượng Sinh Viên và Phân khúc Phổ thông
    u1_role = 'student'
    u1_segment = "Phổ thông (15 - 25 Triệu)"
    
    print(f"1. User chọn: {u1_role.upper()} | Phân khúc: {u1_segment}")
    
    # Bước lọc cứng
    budget_1 = map_segment_to_budget(u1_segment) * multiplier
    df_filtered_1 = apply_hard_filters(df_laptops, max_price=budget_1, min_ram=8, min_storage=256)
    
    # Bước lấy trọng số
    w_mode1 = get_weights_mode_1(u1_role)
    
    # Bước chạy TOPSIS
    if df_filtered_1.empty:
        print("❌ Không có laptop nào thỏa mãn bộ lọc!")
    else:
        result_1 = calculate_topsis(df_filtered_1, w_mode1)
        print("\n🏆 KẾT QUẢ XẾP HẠNG:")
        print(result_1[['brand', 'product_name', 'price', 'weight', 'cpu_point', 'TOPSIS_Score']])

    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("▶️ TEST CHẾ ĐỘ 2: LẬP TRÌNH VIÊN (TÙY CHỈNH THU NHỎ)")
    # Giả lập UI: User chọn Developer, Phân khúc Cao cấp, và chấm điểm các thuộc tính
    u2_role = 'developer'
    u2_segment = "Cao cấp (> 25 Triệu)"
    u2_scores = {'price': 3, 'ram': 10, 'storage': 8, 'cpu': 9} # GPU & Nặng bị ẩn nên điểm = 0
    
    print(f"1. User chọn: {u2_role.upper()} | Phân khúc: {u2_segment}")
    print(f"   Đánh giá điểm: {u2_scores}")
    
    # Bước lọc cứng
    budget_2 = map_segment_to_budget(u2_segment) * multiplier
    df_filtered_2 = apply_hard_filters(df_laptops, max_price=budget_2, min_ram=16, min_storage=512)
    
    # Bước lấy trọng số
    w_mode2 = get_weights_mode_2(u2_role, u2_scores)
    
    # Bước chạy TOPSIS
    if df_filtered_2.empty:
        print("❌ Không có laptop nào thỏa mãn bộ lọc!")
    else:
        result_2 = calculate_topsis(df_filtered_2, w_mode2)
        print("\n🏆 KẾT QUẢ XẾP HẠNG:")
        print(result_2[['brand', 'product_name', 'price', 'ram_capacity', 'storage', 'cpu_point', 'TOPSIS_Score']])
        
        # Cảnh báo
        top_1_price = result_2.iloc[0]['price']
        alert_2 = check_outlier_alert(top_1_price, outlier_threshold=(51.4 * multiplier))
        if alert_2: print("\n" + alert_2)

    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("▶️ TEST CHẾ ĐỘ 3: GAME THỦ (AHP 6x6 HOÀN CHỈNH)")
    # Giả lập UI: User chọn Game thủ, Phân khúc Cao cấp, và tự kéo 15 thanh tạo ma trận 6x6
    u3_role = 'gamer'
    u3_segment = "Cao cấp (> 25 Triệu)"
    
    print(f"1. User chọn: {u3_role.upper()} | Phân khúc: {u3_segment}")
    print("   User kéo 15 thanh slider tạo ma trận AHP ưu tiên tuyệt đối GPU.")
    
    # Bước lọc cứng
    budget_3 = map_segment_to_budget(u3_segment) * multiplier
    df_filtered_3 = apply_hard_filters(df_laptops, max_price=budget_3, min_ram=16, min_storage=512)
    
    # Giả lập tạo ma trận 6x6 (Mức độ quan trọng mong muốn: Price:1, RAM:2, Storage:1, CPU:4, GPU:6, Weight:1)
    w_desired = np.array([1, 2, 1, 4, 6, 1], dtype=float)
    matrix_6x6 = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            matrix_6x6[i, j] = w_desired[i] / w_desired[j]
            
    # Bước lấy trọng số
    w_mode3, cr, is_ok = get_weights_mode_3(matrix_6x6)
    
    if not is_ok:
        print("❌ Lỗi: Ma trận của người dùng bị mâu thuẫn logic (CR >= 0.1)!")
    elif df_filtered_3.empty:
        print("❌ Không có laptop nào thỏa mãn bộ lọc!")
    else:
        print(f"✅ Ma trận AHP hợp lệ (CR = {cr:.4f}). Đang chạy TOPSIS...")
        result_3 = calculate_topsis(df_filtered_3, w_mode3)
        print("\n🏆 KẾT QUẢ XẾP HẠNG:")
        print(result_3[['brand', 'product_name', 'price', 'cpu_point', 'gpu_point', 'TOPSIS_Score']])
        
        # Cảnh báo
        top_1_price_3 = result_3.iloc[0]['price']
        alert_3 = check_outlier_alert(top_1_price_3, outlier_threshold=(51.4 * multiplier))
        if alert_3: print("\n" + alert_3)

    print("\n" + "="*80)
    print("🎉 KẾT THÚC KIỂM THỬ TRÊN DỮ LIỆU THẬT!")
    print("="*80 + "\n")