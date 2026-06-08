# streamlit run test.py
# Đức hình dung web qua đây 
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# 1. THUẬT TOÁN AHP (Tính trọng số từ so sánh cặp)
# ==========================================
def calculate_ahp(matrix):
    """
    Nhận vào ma trận so sánh cặp nxn.
    Trả về: Trọng số (weights) và Tỉ số nhất quán (CR).
    """
    n = matrix.shape[0]
    # Tính toán vector riêng (Eigenvector) bằng phương pháp xấp xỉ dòng
    col_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)

    # Tính toán Tỉ số nhất quán CR
    weighted_sum = np.dot(matrix, weights)
    consistency_vector = weighted_sum / weights
    lambda_max = np.mean(consistency_vector)

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    # Bảng RI chuẩn
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = ri_dict.get(n, 1.41)

    cr = ci / ri if ri > 0 else 0
    return weights, cr


# ==========================================
# 2. THUẬT TOÁN TOPSIS (Xếp hạng giải pháp)
# ==========================================
def run_topsis(decision_matrix, weights, impacts):
    """
    decision_matrix: Ma trận quyết định (m dòng là laptop, n cột là tiêu chí)
    weights: Mảng trọng số n phần tử (từ AHP)
    impacts: Mảng dấu cột ('+' cho Lợi ích, '-' cho Chi phí)
    """
    # Bước 1: Chuẩn hóa ma trận
    norm_matrix = decision_matrix / np.sqrt(
        np.sum(decision_matrix**2, axis=0)
    )

    # Bước 2: Nhân trọng số
    weighted_matrix = norm_matrix * weights

    # Bước 3: Xác định giải pháp lý tưởng tốt (+) và xấu (-)
    ideal_best = []
    ideal_worst = []
    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))

    # Bước 4: Tính khoảng cách Euclidean
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    # Bước 5: Tính điểm tương đồng Closeness Coefficient (C)
    performance_score = dist_worst / (dist_best + dist_worst)
    return performance_score


# ==========================================
# 3. GIAO DIỆN WEB STREAMLIT
# ==========================================
st.title("💻 Hệ Hỗ Trợ Quyết Định Mua Laptop (AHP - TOPSIS)")

st.subheader("Bước 1: So sánh cặp các tiêu chí (AHP)")
st.write(
    "Vui lòng cho biết mức độ ưu tiên giữa các cặp tiêu chí (Giá vs cấu hình):"
)

# Ví dụ so sánh cặp giữa 3 tiêu chí: Giá (Min), RAM (Max), Trọng lượng (Min)
comp_1 = st.slider(
    "Giá so với RAM (1: RAM cực kỳ quan trọng, 9: Giá cực kỳ quan trọng)",
    1.0,
    9.0,
    5.0,
)
comp_2 = st.slider(
    "Giá so với Trọng lượng (1: Trọng lượng cực kỳ quan trọng, 9: Giá cực kỳ quan trọng)",
    1.0,
    9.0,
    5.0,
)
comp_3 = st.slider(
    "RAM so với Trọng lượng (1: Trọng lượng cực kỳ quan trọng, 9: RAM cực kỳ quan trọng)",
    1.0,
    9.0,
    5.0,
)

# Tạo ma trận so sánh cặp 3x3 từ inputs
ahp_matrix = np.array(
    [
        [1.0, comp_1, comp_2],
        [1.0 / comp_1, 1.0, comp_3],
        [1.0 / comp_2, 1.0 / comp_3, 1.0],
    ]
)

weights, cr = calculate_ahp(ahp_matrix)

st.write(f"**Tỉ số nhất quán (CR):** {cr:.4f}")
if cr < 0.1:
    st.success("Đánh giá nhất quán! Trọng số hợp lệ.")
else:
    st.warning("Đánh giá chưa nhất quán (CR >= 0.1). Vui lòng điều chỉnh lại.")

# Hiển thị trọng số
st.write(
    f"Trọng số tính được: **Giá:** {weights[0]:.2f} | **RAM:** {weights[1]:.2f} | **Trọng lượng:** {weights[2]:.2f}"
)