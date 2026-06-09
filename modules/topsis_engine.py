import numpy as np
import pandas as pd

def calculate_ahp_weights(matrix):
    """
    (Hàm bổ trợ) Tính toán trọng số từ ma trận so sánh cặp AHP.
    Input: Ma trận numpy nxn
    Output: Mảng trọng số (weights)
    """
    col_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)
    return weights

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