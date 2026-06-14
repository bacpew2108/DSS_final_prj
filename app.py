# streamlit run app.py
import streamlit as st
import pandas as pd

# Import các hàm từ các thành viên khác theo đúng cấu trúc thư mục
from modules.topsis_engine import calculate_topsis
from modules.charts import draw_radar_chart, draw_weights_donut_chart, draw_match_score_gauge

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Hỗ Trợ Quyết Định Mua Laptop", layout="wide")

st.title("💻 Hệ Thống Tư Vấn Mua Laptop ")

# --- DỰNG LAYOUT: SIDEBAR (Bộ lọc bên trái) ---
st.sidebar.header("Tiêu chí quan tâm")

# Code Thanh trượt (Input) từ 1 đến 10
w_price = st.sidebar.slider("Quan tâm Giá", 1, 10, 5)
w_ram = st.sidebar.slider("Quan tâm RAM", 1, 10, 5)
w_storage = st.sidebar.slider("Quan tâm Ổ cứng", 1, 10, 5)
w_cpu = st.sidebar.slider("Quan tâm CPU", 1, 10, 5)
w_gpu = st.sidebar.slider("Quan tâm GPU", 1, 10, 5)
w_weight = st.sidebar.slider("Quan tâm Trọng lượng", 1, 10, 5)

# Thu thập 6 số, tính tổng và quy ra phần trăm (%) lưu vào weights_array
total_weight = w_price + w_ram + w_storage + w_cpu + w_gpu + w_weight
weights_array = [
    w_price / total_weight,
    w_ram / total_weight,
    w_storage / total_weight,
    w_cpu / total_weight,
    w_gpu / total_weight,
    w_weight / total_weight
]

# --- LOAD DATA SẠCH CỦA TV1 ---
@st.cache_data
def load_data():
    # Đọc file laptops_dataset_cleaned.csv từ thư mục data
    df = pd.read_csv("data/laptops_dataset_cleaned.csv")
    
    # Đổi tên cột để khớp với code của cả nhóm (Huy và Lanh)
    df = df.rename(columns={
        'brand': 'Brand',
        'product_name': 'Tên_Máy'
    })
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Chưa tìm thấy file `data/laptops_dataset_cleaned.csv`. Hãy kiểm tra lại thư mục `data/`")
    st.stop()

# --- BỘ LỌC TĨNH (Rule-based) ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Lọc cơ bản")
if 'Brand' in df.columns:
    brands = ["Tất cả"] + list(df['Brand'].dropna().unique())
    selected_brand = st.selectbox("Chọn Thương hiệu (Hãng)", brands)
else:
    selected_brand = "Tất cả"
    st.sidebar.warning("File data không có cột 'Brand' để lọc.")

# --- LẮP GHÉP LOGIC & HIỂN THỊ KẾT QUẢ ---
# Tạo nút bấm
if st.sidebar.button("Tư vấn Laptop"):
    
    # 1. Lọc data sạch theo hãng trước
    filtered_df = df.copy()
    if selected_brand != "Tất cả" and 'Brand' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
        
    if filtered_df.empty:
        st.warning("Không tìm thấy mẫu laptop nào thuộc hãng này!")
    else:
        st.success("Đã tìm thấy các cấu hình phù hợp.")
        
        # 2. Gửi data đã lọc + mảng trọng số vào hàm calculate_topsis() của TV2
        # Ép top_n = 3 để hiển thị 3 máy đúng yêu cầu
        top3_df = calculate_topsis(filtered_df, weights_array, top_n=3)
        
        # 3. Hiển thị kết quả ra màn hình
        st.subheader("🏆 Top 3 Laptop Phù Hợp Nhất")
        
        # Chỉ chọn hiển thị các cột quan trọng cho người dùng dễ nhìn
        display_cols = ['Tên_Máy', 'Brand', 'price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight', 'TOPSIS_Score']
        actual_display_cols = [col for col in display_cols if col in top3_df.columns]
        
        st.dataframe(top3_df[actual_display_cols], use_container_width=True)
        
        # --- HIỂN THỊ BIỂU ĐỒ TỪ TV3 (HUY) ---
        st.markdown("---")
        
        # Hàng trên cùng: Đồng hồ Gauge cho máy Top 1
        st.subheader("🔥 Mức độ phù hợp của lựa chọn Top 1")
        try:
            top1_score = top3_df['TOPSIS_Score'].iloc[0]
            fig_gauge = draw_match_score_gauge(top1_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
        except KeyError:
            st.error("⚠️ Lỗi: Không tìm thấy cột 'TOPSIS_Score'. Kiểm tra lại hàm calculate_topsis.")

        # Hàng dưới: Chia 2 cột cho Radar và Donut
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Phân tích sức mạnh phần cứng")
            fig_radar = draw_radar_chart(top3_df)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col2:
            st.subheader("🎯 Hồ sơ nhu cầu của bạn")
            fig_donut = draw_weights_donut_chart(weights_array)
            st.plotly_chart(fig_donut, use_container_width=True)
else:
    st.info("Hãy điều chỉnh các thanh trượt bên trái và bấm **Tư vấn Laptop** để xem kết quả.")