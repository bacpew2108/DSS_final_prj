import numpy as np
import pandas as pd

# Hàm AHP (Thêm kiểm tra tỷ số nhất quán CR cho Chế độ 3)
def calculate_ahp_weights_with_cr(matrix):
    """
    Tính trọng số AHP và kiểm tra Tỷ số nhất quán (CR).
    Input: Ma trận numpy 6x6
    Output: weights (mảng trọng số), cr (tỷ số CR), is_consistent (Boolean)
    """
    n = matrix.shape[0]
    
    # 1. Tính trọng số (Như hàm cũ của bạn)
    col_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)
    
    # 2. Tính Tỷ số nhất quán (Consistency Ratio - CR)
    # Tính Eigenvalue lớn nhất (Lambda max)
    weighted_sum_vector = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum_vector / weights)
    
    # Tính Chỉ số nhất quán (Consistency Index - CI)
    ci = (lambda_max - n) / (n - 1)
    
    # Chỉ số ngẫu nhiên (Random Index - RI) cho n=6 là 1.24
    ri = 1.24 
    cr = ci / ri if ri > 0 else 0
    is_consistent = cr < 0.1
    
    return weights, cr, is_consistent

# Hàm Lọc Cứng (Hard Filter Engine)
def apply_hard_filters(df, price_range, min_ram, min_storage, selected_brands=None):
    """
    Lọc bỏ các laptop không đạt tiêu chí cứng.
    price_range: tuple (min_price, max_price) hoặc float (max_price cho backward compatibility)
    min_ram, min_storage: Nếu bằng 0 hoặc None, không áp dụng filter cho tiêu chí đó
    """
    # Handle backward compatibility: nếu price_range là số, chuyển thành tuple
    if isinstance(price_range, (int, float)):
        min_price, max_price = 0, price_range
    else:
        min_price, max_price = price_range
    
    # Bắt đầu với điều kiện giá
    filtered_df = df[
        (df['price'] >= min_price) & 
        (df['price'] <= max_price)
    ].copy()
    
    # Áp dụng filter RAM nếu min_ram > 0
    if min_ram and min_ram > 0:
        filtered_df = filtered_df[df['ram_capacity'] >= min_ram]
    
    # Áp dụng filter Storage nếu min_storage > 0
    if min_storage and min_storage > 0:
        filtered_df = filtered_df[df['storage'] >= min_storage]
    
    # Nếu người dùng có chọn thương hiệu cụ thể
    if selected_brands and len(selected_brands) > 0:
        filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
        
    return filtered_df

# Ánh xạ phân khúc vào backend (Rẻ <15 triệu, Phổ thông 15-25 triệu, Cao cấp >25 triệu)
#Chú ý: Cần truyền tham số cho đúng vào hàm này khi gọi ở Streamlit (app.py) để đảm bảo tính nhất quán giữa frontend và backend
def map_segment_to_budget(segment_name):
    """
    Hàm ánh xạ tên Phân khúc mà người dùng chọn sang Ngân sách tối đa (max_price).
    Trả về tuple (min_price, max_price) để lọc chính xác theo phạm vi.
    (Phân khúc được lấy từ Insight phân tích dữ liệu ở phần trước).
    """
    if segment_name == "Tất cả phân khúc":
        return (0.0, 999.0)  # Không giới hạn giá
    elif segment_name == "Rẻ (< 15 Triệu)":
        return (0.0, 15.0)  # < 15 triệu
    elif segment_name == "Phổ thông (15 - 25 Triệu)":
        return (15.0, 25.0)  # 15-25 triệu
    elif segment_name == "Cao cấp (> 25 Triệu)":
        return (25.0, 999.0)  # > 25 triệu (999 là một con số rất lớn đại diện cho không giới hạn)
    else:
        return (15.0, 25.0)  # Mặc định: Phổ thông
    
# ==========================================
# CHẾ ĐỘ 1: GỢI Ý NHANH (MẶC ĐỊNH - 0 SLIDER)
# ==========================================
def get_weights_mode_1(user_profile):
    """
    Trả về trọng số tĩnh cấu hình sẵn theo đối tượng.
    Thứ tự: ['price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight']
    """
    profiles = {
        'student': np.array([0.352, 0.119, 0.058, 0.165, 0.058, 0.248]),
        'gamer':   np.array([0.103, 0.134, 0.061, 0.264, 0.395, 0.043]),
        'developer': np.array([0.146, 0.222, 0.122, 0.380, 0.048, 0.082])
    }
    # Trả về trọng số, nếu lỗi trả về student làm mặc định
    return profiles.get(user_profile, profiles['student'])


# ==========================================
# CHẾ ĐỘ 2: TÙY CHỌN THU NHỎ (6 SLIDERS -> 4 TIÊU CHÍ)
# ==========================================
def get_weights_mode_2(user_profile, scores_dict):
    """
    Tính trọng số trực tiếp từ 4 thanh kéo (điểm 1-10) của người dùng.
    Mẹo: Đặt trọng số = 0.0 cho các tiêu chí bị ẩn, hàm TOPSIS tự động bỏ qua cột đó mà không bị lỗi ma trận!
    """
    # Khởi tạo mảng trọng số bằng 0
    # Thứ tự: [Giá(0), RAM(1), Ổ cứng(2), CPU(3), GPU(4), Nặng(5)]
    weights = np.zeros(6)
    
    # Map điểm người dùng kéo vào đúng vị trí
    if user_profile == 'student':
        # Giữ: Giá, RAM, CPU, Nặng. (Bỏ: Ổ cứng, GPU)
        weights[0] = scores_dict.get('price', 0)
        weights[1] = scores_dict.get('ram', 0)
        weights[3] = scores_dict.get('cpu', 0)
        weights[5] = scores_dict.get('weight', 0)
        
    elif user_profile == 'gamer':
        # Giữ: Giá, RAM, CPU, GPU. (Bỏ: Ổ cứng, Nặng)
        weights[0] = scores_dict.get('price', 0)
        weights[1] = scores_dict.get('ram', 0)
        weights[3] = scores_dict.get('cpu', 0)
        weights[4] = scores_dict.get('gpu', 0)
        
    elif user_profile == 'developer':
        # Giữ: Giá, RAM, Ổ cứng, CPU. (Bỏ: GPU, Nặng)
        weights[0] = scores_dict.get('price', 0)
        weights[1] = scores_dict.get('ram', 0)
        weights[2] = scores_dict.get('storage', 0)
        weights[3] = scores_dict.get('cpu', 0)
        
    # Chuẩn hóa để tổng trọng số = 1
    total_score = np.sum(weights)
    if total_score > 0:
        weights = weights / total_score
    else:
        # Fallback nếu người dùng kéo tất cả về 0
        weights = get_weights_mode_1(user_profile)
        
    return weights


# ==========================================
# CHẾ ĐỘ 3: TOÀN DIỆN (15 SLIDERS - MA TRẬN 6x6)
# ==========================================
def get_weights_mode_3(ahp_matrix_6x6, user_profile):
    """
    Tính trọng số từ ma trận 6x6 và check CR.
    """
    weights, cr, is_consistent = calculate_ahp_weights_with_cr(ahp_matrix_6x6)
    
    if not is_consistent:
        # Nếu CR >= 0.1, trả về False để Streamlit hiện Pop-up cảnh báo
        return None, cr, False
        
    return weights, cr, True

# Hàm Cảnh Báo Outlier (Hậu xử lý)
def check_outlier_alert(top_1_laptop_price, outlier_threshold=51.4):
    """
    Hàm kiểm tra cảnh báo nếu giá máy quá cao (dựa trên phân tích IQR)
    """
    if top_1_laptop_price > outlier_threshold:
        return f"⚠️ CẢNH BÁO: Sản phẩm Top 1 có mức giá thuộc phân khúc siêu cao cấp ({top_1_laptop_price} Triệu). Phù hợp nhất cho dân chuyên nghiệp!"
    return ""

# Hàm TOPSIS chính (Dùng chung cho cả 3 chế độ)
def calculate_topsis(df, weights_array, top_n=5):
    """
    Hàm tính toán TOPSIS để xếp hạng Laptop.
    Input: 
        - df: DataFrame chứa dữ liệu laptop (đã làm sạch).
        - weights_array: Mảng 6 trọng số (tổng = 1). VD: [0.2, 0.1, 0.1, 0.2, 0.3, 0.1]
        - top_n: Số lượng laptop muốn trả về (Mặc định là top 5).
    Output:
        - DataFrame chứa Top N laptop đã được xếp hạng kèm điểm TOPSIS_Score.
    """
    
    # Đảm bảo Dataframe không bị thay đổi dữ liệu gốc
    result_df = df.copy()

    # Khai báo mảng tên các cột tiêu chí sẽ tham gia tính toán
    # THỨ TỰ: [Giá, RAM, Ổ cứng, CPU, GPU, Nặng]
    criteria_cols = ['price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight']
    
    # Lấy ma trận dữ liệu chỉ gồm các cột số
    matrix = result_df[criteria_cols].values
    
    # ---------------------------------------------------------
    # Bước 1: Khai báo mảng Tính chất (Impacts)
    # [Min, Max, Max, Max, Max, Min] tương ứng [Giá, RAM, Ổ cứng, CPU, GPU, Nặng]
    # ---------------------------------------------------------
    impacts = ['min', 'max', 'max', 'max', 'max', 'min']

    # ---------------------------------------------------------
    # Bước 2: Ma trận chuẩn hóa (chia cho căn bậc 2 tổng bình phương từng cột)
    # ---------------------------------------------------------
    sqrt_sum_sq = np.sqrt(np.sum(matrix**2, axis=0))
    # Tránh lỗi chia cho 0
    sqrt_sum_sq[sqrt_sum_sq == 0] = 1e-10 
    normalized_matrix = matrix / sqrt_sum_sq

    # ---------------------------------------------------------
    # Bước 3: Nhân ma trận với weights_array
    # ---------------------------------------------------------
    weighted_matrix = normalized_matrix * weights_array

    # ---------------------------------------------------------
    # Bước 4: Tìm ra A+ (Best) và A- (Worst)
    # ---------------------------------------------------------
    ideal_best = np.zeros(len(criteria_cols))
    ideal_worst = np.zeros(len(criteria_cols))

    for i in range(len(criteria_cols)):
        if impacts[i] == 'max':
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        elif impacts[i] == 'min':
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    # ---------------------------------------------------------
    # Bước 5: Tính khoảng cách Euclidean và Điểm số TOPSIS
    # ---------------------------------------------------------
    # Tính khoảng cách từ mỗi laptop tới A+
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    
    # Tính khoảng cách từ mỗi laptop tới A-
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # Tính điểm TOPSIS (Closeness Coefficient)
    # Điểm càng gần 1 thì laptop đó càng tốt
    topsis_score = dist_worst / (dist_best + dist_worst)

    # ---------------------------------------------------------
    # Bước 6: Thêm cột TOPSIS_Score, sort từ cao xuống thấp và trả về Top N
    # ---------------------------------------------------------
    result_df['TOPSIS_Score'] = topsis_score
    
    # Sắp xếp giảm dần theo điểm số
    result_df = result_df.sort_values(by='TOPSIS_Score', ascending=False).reset_index(drop=True)

    # Trả về Top N laptop
    return result_df.head(top_n)