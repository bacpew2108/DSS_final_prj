# streamlit run app.py  
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Import các hàm xử lý từ backend (TV2 & TV3)
from modules.topsis_engine import (
    apply_hard_filters, map_segment_to_budget,
    get_weights_mode_1, get_weights_mode_2, get_weights_mode_3,
    check_outlier_alert, calculate_topsis
)
from modules.charts import plot_radar_chart
# Import Random Forest engine (dùng để dự đoán giá ngầm trong TOPSIS)
from modules.model_engine import predict_price


# --- LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN (INFERENCE) ---
@st.cache_resource
def load_best_model():
    """
    Load mô hình tốt nhất (Random Forest, XGBoost, LightGBM...) đã được train sẵn.
    Siêu nhanh, không tốn RAM và CPU để train lại.
    """
    model_path = "models/best_model.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy model tại `{model_path}`. Vui lòng chạy file `python train_model.py` trước!")
        st.stop()

# Gọi hàm load model
best_model_bg = load_best_model()

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Hỗ Trợ Ra Quyết Định Mua Laptop", layout="wide")

# --- HỘP THOẠI HƯỚNG DẪN (POPUP/MODAL) ---
@st.dialog("Hướng Dẫn Sử Dụng", width="large")
def tutorial_dialog():
    st.markdown("""
    Hệ thống giúp người dùng tìm ra chiếc laptop phù hợp nhất bằng cách kết hợp thuật toán ra quyết định đa tiêu chí (TOPSIS, AHP) và mô hình học máy (Random Forest) để thẩm định giá.

    **1. Lọc yêu cầu cơ bản (Cột bên trái)**
    Thiết lập các yêu cầu tối thiểu như mức giá, RAM hay dung lượng ổ cứng. Các máy không đạt tiêu chuẩn sẽ bị loại bỏ trước khi đưa vào thuật toán để tối ưu tốc độ xử lý.

    **2. Chọn chế độ tư vấn (3 Tab ở màn hình chính)**
    - **Chế độ 1 (Nhanh):** Phù hợp nếu người dùng chưa nắm rõ thông số kỹ thuật. Chỉ cần chọn nhóm nhu cầu, hệ thống sẽ áp dụng bộ trọng số đã được thiết lập sẵn.
    - **Chế độ 2 (Tùy chỉnh):** Sử dụng thanh trượt (1-10) để định lượng yếu tố nào quan trọng hơn. Điểm càng cao, hệ thống càng ưu tiên thông số đó khi xếp hạng.
    - **Chế độ 3 (Chuyên sâu):** Sử dụng ma trận AHP để so sánh trực tiếp từng cặp tiêu chí (Ví dụ: Giá vs RAM). Kéo thanh trượt về bên nào thì bên đó quan trọng hơn. Hệ thống sẽ tự động tính toán Tỷ số nhất quán (CR) để đảm bảo các lựa chọn không bị mâu thuẫn logic.

    **3. Đọc kết quả và Đánh giá rủi ro về giá**
    Sau khi xếp hạng các máy dựa trên **Độ phù hợp (%)**, mô hình AI sẽ ước tính giá trị thực của phần cứng và so sánh với giá niêm yết của cửa hàng:
    - **Món hời:** Giá bán rẻ hơn đáng kể (>20%) so với giá trị thực tế của linh kiện. Cấu hình rất tốt trong tầm giá.
    - **Giá hợp lý:** Giá bán tương xứng với cấu hình phần cứng.
    - **Bị định giá cao:** Máy đắt hơn so với mặt bằng chung linh kiện. Phần tiền chênh lệch chủ yếu nằm ở giá trị thương hiệu hoặc thiết kế vỏ máy.
    - **Cảnh báo giá ảo:** Giá bán quá rẻ so với cấu hình (lệch >40%). Cần kiểm tra kỹ tình trạng máy (máy cũ, lỗi) hoặc cẩn trọng với các bài đăng lừa đảo.
    """)
    
    # Nút bấm để tắt Popup
    if st.button("Đã hiểu", type="primary", use_container_width=True):
        st.session_state.first_visit = False
        st.rerun()

# Quản lý trạng thái: Tự động bật Popup ở lần đầu tiên
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True

if st.session_state.first_visit:
    tutorial_dialog()

# --- BỐ CỤC HEADER (TIÊU ĐỀ & NÚT HƯỚNG DẪN) ---
# Chia cột tỷ lệ 8.5 : 1.5 để đẩy nút sang sát mép phải
col_title, col_btn = st.columns([8.5, 1.5])

with col_title:
    st.title("Hệ Thống Tư Vấn Laptop Đa Tiêu Chí")

with col_btn:
    st.write("") # Dòng trống để đẩy nút xuống cho ngang hàng với tiêu đề
    if st.button("📖 Hướng dẫn", use_container_width=True):
        tutorial_dialog()

# (Phần code dưới giữ nguyên)
# --- LỚP DỮ LIỆU (DATA LAYER) ---

# --- LỚP GIAO DIỆN: RÀNG BUỘC CỨNG (HARD FILTERS) ---
st.sidebar.header("1. Lọc Ràng Buộc Cứng")

# 3. Thêm một nút bấm trên Sidebar để người dùng có thể chủ động mở lại Popup
st.sidebar.button("Hướng Dẫn", on_click=tutorial_dialog, use_container_width=True)
st.sidebar.markdown("---")
# --- LỚP DỮ LIỆU (DATA LAYER) ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/laptops_dataset_cleaned.csv")
    
    # Chuẩn hóa tên cột để khớp với các hàm xử lý
    if 'product_name' in df.columns:
        df = df.rename(columns={'product_name': 'Tên_Máy'})
    if 'brand' not in df.columns and 'Brand' in df.columns:
        df = df.rename(columns={'Brand': 'brand'})
        
    # Chuẩn hóa đơn vị Giá về Triệu VNĐ để đồng bộ thuật toán và giao diện
    if 'price' in df.columns and df['price'].max() > 1000:
        df['price'] = df['price'] / 1000000.0
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Chưa tìm thấy file `data/laptops_dataset_cleaned.csv`. Hãy kiểm tra lại thư mục `data/`")
    st.stop()



# --- LỚP GIAO DIỆN: RÀNG BUỘC CỨNG (HARD FILTERS) ---
st.sidebar.header("1. Lọc Ràng Buộc Cứng")
segment = st.sidebar.selectbox(
    "Phân khúc ngân sách", 
    ["Tất cả phân khúc", "Rẻ (< 15 Triệu)", "Phổ thông (15 - 25 Triệu)", "Cao cấp (> 25 Triệu)"],
    index=1
)
price_range = map_segment_to_budget(segment)

ram_option = st.sidebar.selectbox("RAM tối thiểu (GB)", ["Tất cả", 8, 16, 32, 64], index=1)
min_ram = 0 if ram_option == "Tất cả" else ram_option

storage_option = st.sidebar.selectbox("Ổ cứng tối thiểu (GB)", ["Tất cả", 256, 512, 1024], index=1)
min_storage = 0 if storage_option == "Tất cả" else storage_option

brands_list = df['brand'].dropna().unique().tolist()
selected_brands = st.sidebar.multiselect("Thương hiệu ưu tiên (Bỏ trống để tìm tất cả)", brands_list)

top_n_results = st.sidebar.selectbox("Số lượng laptop đưa ra (Top N)", list(range(1, 11)), index=2)

# --- LỚP NGHIỆP VỤ: 3 CHẾ ĐỘ VẬN HÀNH (MODES) ---
tab1, tab2, tab3 = st.tabs([
    "Chế độ 1: Gợi ý Nhanh",
    "Chế độ 2: Tùy chọn Thu nhỏ",
    "Chế độ 3: Toàn diện (Chuyên gia)",
])

weights_array = None
is_ahp_valid = True
cr_score = 0.0

# TAB 1: GỢI Ý NHANH
with tab1:
    st.info("Hệ thống tự động sử dụng bộ trọng số AHP đã được các chuyên gia tối ưu hóa.")
    profile_1 = st.selectbox("Bạn thuộc nhóm đối tượng nào?", ["Sinh viên / Văn phòng", "Game thủ / Đồ họa", "Lập trình viên / Kỹ sư dữ liệu"], key="p1")
    profile_map = {"Sinh viên / Văn phòng": "student", "Game thủ / Đồ họa": "gamer", "Lập trình viên / Kỹ sư dữ liệu": "developer"}
    
    if st.button("Chạy Tư Vấn (Chế độ 1)", use_container_width=True):
        weights_array = get_weights_mode_1(profile_map[profile_1])

# TAB 2: THU NHỎ
with tab2:
    st.info("Tùy chỉnh độ quan trọng (từ 1 đến 10) cho các thông số cốt lõi. Các thông số phụ sẽ được ẩn để tránh gây nhiễu.")
    profile_2 = st.selectbox("Bạn thuộc nhóm đối tượng nào?", ["Sinh viên / Văn phòng", "Game thủ / Đồ họa", "Lập trình viên / Kỹ sư dữ liệu"], key="p2")
    p_code = profile_map[profile_2]
    
    scores = {}
    col1, col2 = st.columns(2)
    
    if p_code == 'student':
        with col1:
            scores['price'] = st.slider("Quan tâm Giá bán", 1, 10, 8, key="t2_p1")
            scores['weight'] = st.slider("Quan tâm Mỏng nhẹ", 1, 10, 7, key="t2_w1")
        with col2:
            scores['cpu'] = st.slider("Hiệu năng CPU", 1, 10, 5, key="t2_c1")
            scores['ram'] = st.slider("Dung lượng RAM", 1, 10, 5, key="t2_r1")
    elif p_code == 'gamer':
        with col1:
            scores['gpu'] = st.slider("Hiệu năng Card Đồ họa (GPU)", 1, 10, 9, key="t2_g2")
            scores['cpu'] = st.slider("Hiệu năng CPU", 1, 10, 8, key="t2_c2")
        with col2:
            scores['ram'] = st.slider("Dung lượng RAM", 1, 10, 7, key="t2_r2")
            scores['price'] = st.slider("Quan tâm Giá bán", 1, 10, 5, key="t2_p2")
    elif p_code == 'developer':
        with col1:
            scores['cpu'] = st.slider("Hiệu năng CPU", 1, 10, 9, key="t2_c3")
            scores['ram'] = st.slider("Dung lượng RAM", 1, 10, 9, key="t2_r3")
        with col2:
            scores['storage'] = st.slider("Dung lượng Ổ cứng", 1, 10, 6, key="t2_s3")
            scores['price'] = st.slider("Quan tâm Giá bán", 1, 10, 5, key="t2_p3")
            
    if st.button("Chạy Tư Vấn (Chế độ 2)", use_container_width=True):
        weights_array = get_weights_mode_2(p_code, scores)

# TAB 3: TOÀN DIỆN (MA TRẬN 6x6)
with tab3:
    st.warning("Bạn đang thiết lập Ma trận so sánh cặp AHP (15 cặp). Nếu chỉ số Nhất quán CR >= 0.1, hệ thống sẽ cảnh báo.")
    st.markdown("*(**Hướng dẫn:** Kéo về bên nào thì tiêu chí bên đó quan trọng hơn. Ở giữa = Quan trọng ngang nhau)*")
    
    # Khởi tạo session state cho CR
    if 'ahp_cr' not in st.session_state:
        st.session_state.ahp_cr = 0.0
    if 'ahp_valid' not in st.session_state:
        st.session_state.ahp_valid = True
    
    labels = ['Giá', 'RAM', 'Ổ cứng', 'CPU', 'GPU', 'Trọng lượng']
    ahp_matrix = np.ones((6, 6))
    
    # Placeholder để hiển thị CR
    cr_placeholder = st.empty()

    # Hàm map giá trị slider (-8 đến 8) sang thang đo Saaty (1/9 đến 9)
    def map_slider_to_ahp(slider_val):
        if slider_val == 0:
            return 1.0
        elif slider_val < 0:
            # Kéo về bên trái (Tiêu chí i quan trọng hơn)
            return float(abs(slider_val) + 1)
        else:
            # Kéo về bên phải (Tiêu chí j quan trọng hơn)
            return 1.0 / float(slider_val + 1)
    
    with st.expander("Nhóm so sánh: GIÁ CẢ so với các tiêu chí khác", expanded=True):
        i = 0
        for j in range(1, 6):
            val = st.slider(f"{labels[i]}  <------------------------>  {labels[j]}", -8, 8, 0, key=f"ahp_{i}_{j}")
            ahp_val = map_slider_to_ahp(val)
            ahp_matrix[i, j] = ahp_val
            ahp_matrix[j, i] = 1.0 / ahp_val

    with st.expander("Nhóm so sánh: DUNG LƯỢNG RAM so với các tiêu chí khác"):
        i = 1
        for j in range(2, 6):
            val = st.slider(f"{labels[i]}  <------------------------>  {labels[j]}", -8, 8, 0, key=f"ahp_{i}_{j}")
            ahp_val = map_slider_to_ahp(val)
            ahp_matrix[i, j] = ahp_val
            ahp_matrix[j, i] = 1.0 / ahp_val

    with st.expander("Nhóm so sánh: Ổ CỨNG so với các tiêu chí khác"):
        i = 2
        for j in range(3, 6):
            val = st.slider(f"{labels[i]}  <------------------------>  {labels[j]}", -8, 8, 0, key=f"ahp_{i}_{j}")
            ahp_val = map_slider_to_ahp(val)
            ahp_matrix[i, j] = ahp_val
            ahp_matrix[j, i] = 1.0 / ahp_val

    with st.expander("Nhóm so sánh: HIỆU NĂNG CPU so với các tiêu chí khác"):
        i = 3
        for j in range(4, 6):
            val = st.slider(f"{labels[i]}  <------------------------>  {labels[j]}", -8, 8, 0, key=f"ahp_{i}_{j}")
            ahp_val = map_slider_to_ahp(val)
            ahp_matrix[i, j] = ahp_val
            ahp_matrix[j, i] = 1.0 / ahp_val

    with st.expander("Nhóm so sánh: HIỆU NĂNG GPU so với TRỌNG LƯỢNG"):
        i = 4
        j = 5
        val = st.slider(f"{labels[i]}  <------------------------>  {labels[j]}", -8, 8, 0, key=f"ahp_{i}_{j}")
        ahp_val = map_slider_to_ahp(val)
        ahp_matrix[i, j] = ahp_val
        ahp_matrix[j, i] = 1.0 / ahp_val

    # Tính CR sau tất cả slider và cập nhật placeholder ở đầu
    from modules.topsis_engine import calculate_ahp_weights_with_cr
    weights_check, cr_check, is_valid_check = calculate_ahp_weights_with_cr(ahp_matrix)
    
    st.session_state.ahp_cr = cr_check
    st.session_state.ahp_valid = is_valid_check
    
    # Cập nhật UI tính hợp lệ của ma trận
    with cr_placeholder.container():
        st.markdown("### Kiểm tra Hợp Lệ Ma Trận AHP")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tỷ số nhất quán (CR)", f"{st.session_state.ahp_cr:.4f}", delta="< 0.1 = Hợp lệ", delta_color="inverse")
        with col2:
            if st.session_state.ahp_valid:
                st.success("Ma trận hợp lệ!")
            else:
                st.error("Ma trận mâu thuẫn!")
        with col3:
            st.info("Chỉ số CR giúp đảm bảo bạn không đánh giá logic vòng tròn (VD: A>B, B>C nhưng C>A)")
        st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Chạy Tư Vấn (Chế độ 3)", use_container_width=True):
        # FIX LỖI Ở ĐÂY: Hàm get_weights_mode_3 chỉ nhận 1 tham số
        weights_array, cr_score, is_ahp_valid = get_weights_mode_3(ahp_matrix)

# --- LỚP XỬ LÝ: KẾT QUẢ TOPSIS & HIỂN THỊ ---
if weights_array is not None:
    st.markdown("---")
    
    if not is_ahp_valid:
        st.error(f"MA TRẬN KHÔNG NHẤT QUÁN! Tỷ số CR = {cr_score:.3f} (Vượt ngưỡng 0.1). Vui lòng điều chỉnh lại các thanh trượt ở Chế độ 3 sao cho logic hơn.")
    else:
        # 1. Lọc dữ liệu thô
        filtered_df = apply_hard_filters(df, price_range, min_ram, min_storage, selected_brands)
        
        if filtered_df.empty:
            st.error("Không có laptop nào thỏa mãn các ràng buộc cứng ở cột bên trái. Bạn hãy nới lỏng ngân sách hoặc giảm yêu cầu dung lượng nhé!")
        else:
            # 2. Chạy thuật toán lõi
            top3_df = calculate_topsis(filtered_df, weights_array, top_n=top_n_results)
            
            # 3. Cảnh báo Outlier cho Top 1
            top1_price = top3_df.iloc[0]['price']
            alert_msg = check_outlier_alert(top1_price)
            if alert_msg:
                st.warning(alert_msg)
                
            if cr_score > 0:
                st.success(f"Ma trận hợp lệ! Tỷ số CR = {cr_score:.3f}")
                
            # 4. Hiển thị kết quả dạng List Card (UI E-commerce)
            st.subheader(f"Top {top_n_results} Laptop Tốt Nhất Cho Bạn")
            st.markdown("---")

            # Khởi tạo tiêu đề các cột
            header_col1, header_col2, header_col3 = st.columns([2, 5, 2])
            with header_col1: st.markdown("**Tên Sản Phẩm**")
            with header_col2: st.markdown("**Thông Số Kỹ Thuật**")
            with header_col3: st.markdown("**Giá Niêm Yết & RF Định Giá**")
            st.markdown("---")

            # Duyệt qua từng laptop trong Top N để render giao diện
            for rank, (idx, row) in enumerate(top3_df.iterrows(), start=1):

                # ── Chạy ngầm AI để dự đoán giá ──────────────────────────
                rf_pred = predict_price(
                    best_model_bg,
                    cpu_point  = float(row.get('cpu_point',  0)),
                    gpu_point  = float(row.get('gpu_point',  0)),
                    ram_gb     = float(row.get('ram_capacity', 0)),
                    storage_gb = float(row.get('storage',    0)),
                    weight_kg  = float(row.get('weight',     0)),
                )
                rf_price   = rf_pred["price_M"]
                rf_segment = rf_pred["segment"]
                actual_price = float(row.get('price', 0))
                diff = rf_price - actual_price
                diff_pct = (diff / rf_price * 100) if rf_price > 0 else 0

                # 4. Áp dụng Business Rules phân loại nhãn (Bỏ icons)
                if diff_pct > 40:
                    badge_text = "CẢNH BÁO: GIÁ ẢO / LỪA ĐẢO"
                    badge_color = "#e53e3e" # Đỏ nguy hiểm
                elif diff_pct >= 20:
                    badge_text = "MÓN HỜI (DEAL TỐT)"
                    badge_color = "#38a169" # Xanh lá tích cực
                elif diff_pct >= -20:
                    badge_text = "GIÁ HỢP LÝ (AN TOÀN)"
                    badge_color = "#3182ce" # Xanh dương tin cậy
                else:
                    badge_text = "BỊ ĐỊNH GIÁ CAO"
                    badge_color = "#d69e2e" # Vàng/Cam cảnh báo

                with st.container():
                    col1, col2, col3 = st.columns([2, 5, 2])

                    # Cột 1: Hãng, Tên máy, Rank badge
                    with col1:
                        rank_labels = ["#1", "#2", "#3"] + [f"#{i}" for i in range(4, 21)]
                        st.markdown(f"### {rank_labels[rank-1]}")
                        st.markdown(f"**{row.get('brand', 'Unknown')}**")
                        st.markdown(f"{row.get('Tên_Máy', 'No Name')}")

                    # Cột 2: Cấu hình kỹ thuật
                    with col2:
                        cpu_display = row.get('processor', f"{row.get('cpu_point', 0):.1f} Pts")
                        gpu_display = row.get('video_graphics', f"{row.get('gpu_point', 0):.1f} Pts")
                        screen_str = f"{row.get('display', '')} {row.get('display_resolution', '')} {row.get('display_refresh_rate', '')}".strip()
                        screen_display = screen_str if screen_str else "Đang cập nhật"
                        keyboard_display = row.get('keyboard', 'Đang cập nhật')
                        st.markdown(f"""
                        - **CPU:** {cpu_display}
                        - **GPU:** {gpu_display}
                        - **RAM:** {row.get('ram_capacity', 0)} GB
                        - **Ổ cứng:** {row.get('storage', 0)} GB
                        - **Màn hình:** {screen_display}
                        - **Bàn phím:** {keyboard_display}
                        - **Trọng lượng:** {row.get('weight', 0)} kg
                        """)

                    # Cột 3: Giá niêm yết + RF định giá
                    with col3:
                        # Giá thực tế
                        st.markdown(f"### {actual_price:,.1f} Triệu VNĐ")

                        # Độ phù hợp TOPSIS
                        fit_score = row.get('TOPSIS_Score', 0) * 100
                        st.markdown(f"Độ phù hợp: **{fit_score:.1f}%**")
                        st.progress(int(fit_score))

                        # ── Tính toán Ngưỡng Động (Dynamic Threshold) ──
                        diff = rf_price - actual_price
                        # Công thức chuẩn: (AI - Thực tế) / AI * 100
                        diff_pct = (diff / rf_price * 100) if rf_price > 0 else 0

                        # Phân loại đánh giá
                        if diff_pct > 40:
                            eval_text = "CẢNH BÁO: LỪA ĐẢO/GIÁ ẢO"
                            eval_color = "#e53e3e" # Đỏ
                        elif diff_pct >= 20:
                            eval_text = "MÓN HỜI (DEAL TỐT)"
                            eval_color = "#38a169" # Xanh lá
                        elif diff_pct >= -20:
                            eval_text = "GIÁ HỢP LÝ"
                            eval_color = "#3182ce" # Xanh dương
                        else:
                            eval_text = "BỊ ĐỊNH GIÁ CAO"
                            eval_color = "#d69e2e" # Cam

                        # Màu nền theo phân khúc
                        seg_color_map = {
                            "Rẻ (<15 Triệu)":          "#48bb78",
                            "Phổ thông (15–25 Triệu)": "#63b3ed",
                            "Cao cấp (>25 Triệu)":     "#f6ad55",
                        }
                        seg_color = seg_color_map.get(rf_segment, "#63b3ed")
                        
                        # Icon tam giác mũi tên
                        diff_icon = "▲" if diff > 0 else "▼" if diff < 0 else "●"
                        diff_color = "#fc8181" if diff > 0 else "#68d391" if diff < 0 else "#a0aec0"

                        # ── Render Giao diện (Mỏng nhẹ + Đánh giá) ──
                        st.markdown(f"""
                        <div style="
                            margin-top: 10px;
                            background: linear-gradient(135deg, {seg_color}18, {seg_color}08);
                            border: 1px solid {seg_color}55;
                            border-radius: 10px;
                            padding: 8px 10px;
                            font-size: 0.82rem;
                        ">
                            <div style="color:#a0aec0; font-size:0.72rem; margin-bottom:3px;">RF Định giá</div>
                            <div style="color:{seg_color}; font-weight:700; font-size:0.95rem;">
                                {rf_price:,.1f} Tr
                            </div>
                            <div style="color:{diff_color}; font-size:0.75rem; margin-top:2px;">
                                {diff_icon} {abs(diff):.1f} Tr ({abs(diff_pct):.0f}%)
                            </div>
                            <div style="color:{eval_color}; font-weight:700; font-size:0.75rem; margin-top:3px;">
                                {eval_text}
                            </div>
                            <div style="color:{seg_color}; font-size:0.7rem; margin-top:4px;">
                                Phân khúc: {rf_segment}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

            st.subheader("Biểu Đồ Phân Tích Cấu Hình")
            fig = plot_radar_chart(top3_df, segment, name_col='Tên_Máy', top_n=top_n_results)
            st.plotly_chart(fig, use_container_width=True)