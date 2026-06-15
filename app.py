# streamlit run test.py  
import streamlit as st
import pandas as pd
import numpy as np

# Import các hàm xử lý từ backend (TV2 & TV3)
from modules.topsis_engine import (
    apply_hard_filters, map_segment_to_budget,
    get_weights_mode_1, get_weights_mode_2, get_weights_mode_3,
    check_outlier_alert, calculate_topsis
)
from modules.charts import plot_radar_chart

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Hỗ Trợ Ra Quyết Định Mua Laptop", layout="wide")
st.title("💻 Hệ Thống Tư Vấn Laptop Đa Tiêu Chí ")

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
    st.error("⚠️ Chưa tìm thấy file `data/laptops_dataset_cleaned.csv`. Hãy kiểm tra lại thư mục `data/`")
    st.stop()

# --- LỚP GIAO DIỆN: RÀNG BUỘC CỨNG (HARD FILTERS) ---
st.sidebar.header("🛠️ 1. Lọc Ràng Buộc Cứng")
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

top_n_results = st.sidebar.selectbox("Số lượng laptop đưa ra (Top N)", [1, 2, 3, 4, 5, 7, 10], index=2)

# --- LỚP NGHIỆP VỤ: 3 CHẾ ĐỘ VẬN HÀNH (MODES) ---
tab1, tab2, tab3 = st.tabs([
    "⚡ Chế độ 1: Gợi ý Nhanh", 
    "🎛️ Chế độ 2: Tùy chọn Thu nhỏ", 
    "🔬 Chế độ 3: Toàn diện (Chuyên gia)"
])

weights_array = None
is_ahp_valid = True
cr_score = 0.0

# TAB 1: GỢI Ý NHANH
with tab1:
    st.info("Hệ thống tự động sử dụng bộ trọng số AHP đã được các chuyên gia tối ưu hóa.")
    profile_1 = st.selectbox("Bạn thuộc nhóm đối tượng nào?", ["Sinh viên / Văn phòng", "Game thủ / Đồ họa", "Lập trình viên / Kỹ sư dữ liệu"], key="p1")
    profile_map = {"Sinh viên / Văn phòng": "student", "Game thủ / Đồ họa": "gamer", "Lập trình viên / Kỹ sư dữ liệu": "developer"}
    
    if st.button(" Chạy Tư Vấn (Chế độ 1)", use_container_width=True):
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
            
    if st.button(" Chạy Tư Vấn (Chế độ 2)", use_container_width=True):
        weights_array = get_weights_mode_2(p_code, scores)

# TAB 3: TOÀN DIỆN (MA TRẬN 6x6)
with tab3:
    st.warning("Bạn đang thiết lập Ma trận so sánh cặp AHP (15 cặp). Nếu chỉ số Nhất quán CR > 0.1, hệ thống sẽ cảnh báo.")
    st.markdown("*(Thang đo: 1 = Bằng nhau, 9 = Cực kỳ quan trọng hơn)*")
    
    # Khởi tạo session state cho CR
    if 'ahp_cr' not in st.session_state:
        st.session_state.ahp_cr = 0.0
    if 'ahp_valid' not in st.session_state:
        st.session_state.ahp_valid = True
    
    labels = ['Giá', 'RAM', 'Ổ cứng', 'CPU', 'GPU', 'Trọng lượng']
    ahp_matrix = np.ones((6, 6))
    
    # Placeholder để hiển thị CR (sẽ cập nhật sau)
    cr_placeholder = st.empty()
    
    with st.expander("💰 Nhóm so sánh: GIÁ CẢ so với các tiêu chí khác", expanded=True):
        i = 0
        for j in range(1, 6):
            val = st.slider(f"{labels[i]} so với {labels[j]}", 1.0, 9.0, 5.0, step=1.0, key=f"ahp_{i}_{j}")
            ahp_matrix[i, j] = val
            ahp_matrix[j, i] = 1.0 / val

    with st.expander("⚖️ Nhóm so sánh: DUNG LƯỢNG RAM so với các tiêu chí khác"):
        i = 1
        for j in range(2, 6):
            val = st.slider(f"{labels[i]} so với {labels[j]}", 1.0, 9.0, 5.0, step=1.0, key=f"ahp_{i}_{j}")
            ahp_matrix[i, j] = val
            ahp_matrix[j, i] = 1.0 / val

    with st.expander("💾 Nhóm so sánh: Ổ CỨNG so với các tiêu chí khác"):
        i = 2
        for j in range(3, 6):
            val = st.slider(f"{labels[i]} so với {labels[j]}", 1.0, 9.0, 5.0, step=1.0, key=f"ahp_{i}_{j}")
            ahp_matrix[i, j] = val
            ahp_matrix[j, i] = 1.0 / val

    with st.expander("🧠 Nhóm so sánh: HIỆU NĂNG CPU so với các tiêu chí khác"):
        i = 3
        for j in range(4, 6):
            val = st.slider(f"{labels[i]} so với {labels[j]}", 1.0, 9.0, 5.0, step=1.0, key=f"ahp_{i}_{j}")
            ahp_matrix[i, j] = val
            ahp_matrix[j, i] = 1.0 / val

    with st.expander("🎮 Nhóm so sánh: HIỆU NĂNG GPU so với TRỌNG LƯỢNG"):
        i = 4
        j = 5
        val = st.slider(f"{labels[i]} so với {labels[j]}", 1.0, 9.0, 5.0, step=1.0, key=f"ahp_{i}_{j}")
        ahp_matrix[i, j] = val
        ahp_matrix[j, i] = 1.0 / val

    # Tính CR sau tất cả slider và cập nhật placeholder ở đầu
    from modules.topsis_engine import calculate_ahp_weights_with_cr
    weights_check, cr_check, is_valid_check = calculate_ahp_weights_with_cr(ahp_matrix)
    
    # Cập nhật session state
    st.session_state.ahp_cr = cr_check
    st.session_state.ahp_valid = is_valid_check
    
    # Cập nhật placeholder ở đầu với giá trị mới
    with cr_placeholder.container():
        st.markdown("### 📊 Tính Hợp Lệ Của Ma Trận AHP")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tỷ số nhất quán (CR)", f"{st.session_state.ahp_cr:.4f}", delta="< 0.1 = Hợp lệ")
        with col2:
            if st.session_state.ahp_valid:
                st.success("✅ Ma trận hợp lệ!")
            else:
                st.error("❌ Ma trận chưa hợp lệ!")
        with col3:
            st.info(f"Ngưỡng: CR < 0.1")
        st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button(" Chạy Tư Vấn (Chế độ 3)", use_container_width=True):
        weights_array, cr_score, is_ahp_valid = get_weights_mode_3(ahp_matrix, "custom")

# --- LỚP XỬ LÝ: KẾT QUẢ TOPSIS & HIỂN THỊ ---
if weights_array is not None:
    st.markdown("---")
    
    if not is_ahp_valid:
        st.error(f"❌ MA TRẬN KHÔNG NHẤT QUÁN! Tỷ số CR = {cr_score:.3f} (Vượt ngưỡng 0.1). Vui lòng điều chỉnh lại các thanh trượt ở Chế độ 3 sao cho logic hơn.")
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
            st.subheader(f"🏆 TOP {top_n_results} LAPTOP TỐT NHẤT CHO BẠN")
            st.markdown("---")
            
            # Khởi tạo tiêu đề các cột
            header_col1, header_col2, header_col3 = st.columns([2, 5, 2])
            with header_col1: st.markdown("**Tên Sản Phẩm**")
            with header_col2: st.markdown("**Thông Số Kỹ Thuật**")
            with header_col3: st.markdown("**Giá Ước Tính**")
            st.markdown("---")

            # Duyệt qua từng laptop trong Top 3 để render giao diện
            for idx, row in top3_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 5, 2])
                    
                    # Cột 1: Hãng và Tên máy
                    with col1:
                        st.markdown(f"**{row.get('brand', 'Unknown')}**")
                        st.markdown(f"{row.get('Tên_Máy', 'No Name')}")
                        
                    # Cột 2: Cấu hình kỹ thuật
                    with col2:
                        # Lấy tên CPU/GPU thật
                        cpu_display = row.get('processor', f"{row.get('cpu_point', 0):.1f} Benchmark Pts")
                        gpu_display = row.get('video_graphics', f"{row.get('gpu_point', 0):.1f} Benchmark Pts")
                        
                        # Ghép thông số Màn hình từ 3 cột (Kích thước, Độ phân giải, Tần số quét)
                        screen_str = f"{row.get('display', '')} {row.get('display_resolution', '')} {row.get('display_refresh_rate', '')}".strip()
                        screen_display = screen_str if screen_str else "Đang cập nhật"
                        
                        # Lấy thông tin Bàn phím
                        keyboard_display = row.get('keyboard', 'Đang cập nhật')
                        
                        # In ra giao diện
                        st.markdown(f"""
                        - **CPU:** {cpu_display}
                        - **GPU:** {gpu_display}
                        - **RAM:** {row.get('ram_capacity', 0)} GB
                        - **Ổ cứng:** {row.get('storage', 0)} GB
                        - **Màn hình:** {screen_display}
                        - **Bàn phím:** {keyboard_display}
                        - **Trọng lượng:** {row.get('weight', 0)} kg
                        """)
                        
                    # Cột 3: Mức giá và Độ phù hợp
                    with col3:
                        st.markdown(f"### {row.get('price', 0):,.1f} Triệu VNĐ")
                        fit_score = row.get('TOPSIS_Score', 0) * 100
                        st.markdown(f"Độ phù hợp: **{fit_score:.1f}%**")
                        st.progress(int(fit_score))
                        
                st.markdown("---")

            # 5. Vẽ biểu đồ Radar
            st.subheader("📊 Biểu Đồ Phân Tích Cấu Hình")
            fig = plot_radar_chart(top3_df, segment, name_col='Tên_Máy')
            st.plotly_chart(fig, use_container_width=True)