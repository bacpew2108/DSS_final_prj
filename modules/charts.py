import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_radar_chart(top_df, segment_name, name_col='product_name'):
    """
    Vẽ biểu đồ Radar Chart với hệ quy chiếu (mức sàn/trần) tự động thay đổi
    theo phân khúc giá (Rẻ / Phổ thông / Cao cấp).
    """
    compare_df = top_df.head(3).copy()
    
    categories = ['Price', 'RAM', 'Storage', 'CPU Score', 'GPU Score', 'Weight']
    cols = ['price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight']
    units = ['Million VND', 'GB', 'GB', 'Pts', 'Pts', 'Kg']
    
    # ==========================================
    # BỘ TIÊU CHUẨN (SÀN - TRẦN) THEO 3 PHÂN KHÚC
    # Định dạng: { 'tên cột': (Min_Sàn, Max_Trần) }
    # Lưu ý: Cột Càng thấp càng tốt (Price, Weight) thì hàm chuẩn hóa sẽ tự đảo ngược
    # ==========================================
    anchors = {
        "Rẻ (< 15 Triệu)": {
            'price': (8.0, 15.0),       # Giá từ 8 -> 15 tr
            'ram_capacity': (4, 16),    # RAM 4GB -> 16GB
            'storage': (256, 512),      # Ổ cứng 256GB -> 512GB
            'cpu_point': (45, 60),      # CPU ở phân khúc rẻ thường từ 45-60 điểm
            'gpu_point': (15, 30),      # Chủ yếu là card Onboard (15-30đ)
            'weight': (1.2, 2.5)        
        },
        "Phổ thông (15 - 25 Triệu)": {
            'price': (15.0, 25.0),
            'ram_capacity': (8, 24),
            'storage': (512, 1024),
            'cpu_point': (50, 70),      # Phân khúc tầm trung CPU từ 50-70 điểm
            'gpu_point': (25, 45),      # Thường có card rời cỡ RTX 3050/4050
            'weight': (1.2, 2.7)
        },
        "Cao cấp (> 25 Triệu)": {
            'price': (25.0, 60.0),      # Cận trên có thể tự do, lấy mốc 60 làm chuẩn
            'ram_capacity': (16, 64),
            'storage': (512, 2048),
            'cpu_point': (65, 85),      # Chỉ tính những CPU quái vật (65-85đ)
            'gpu_point': (40, 70),      # Card RTX 4060 trở lên
            'weight': (1.2, 3.2)        # Cao cấp hay có máy gaming nặng
        }
    }
    
    # Lấy hệ quy chiếu dựa theo segment người dùng chọn (Mặc định lấy Phổ thông nếu lỗi)
    current_anchor = anchors.get(segment_name, anchors["Phổ thông (15 - 25 Triệu)"])
    
    # ==========================================
    # CHUẨN HÓA DỮ LIỆU TỪ 0 ĐẾN 1 THEO HỆ QUY CHIẾU
    # ==========================================
    norm_data = {}
    for col in cols:
        vals = compare_df[col].values
        a_min, a_max = current_anchor[col]
        
        # Hàm np.clip để đảm bảo nếu máy thực tế có thông số vượt trần hoặc thủng sàn
        # thì điểm cũng chỉ ghim ở tối đa 1.0 và tối thiểu 0.0
        if col in ['price', 'weight']: 
            # Cost criteria (Thấp là tốt): Đảo ngược (Max - Val) / (Max - Min)
            norm_v = (a_max - vals) / (a_max - a_min)
        else: 
            # Benefit criteria (Cao là tốt): (Val - Min) / (Max - Min)
            norm_v = (vals - a_min) / (a_max - a_min)
            
        # Ghim giá trị trong khoảng [0.0, 1.0]
        norm_data[col] = np.clip(norm_v, 0.0, 1.0).tolist()
                
    norm_df = pd.DataFrame(norm_data)
    
    # ==========================================
    # KHỞI TẠO BIỂU ĐỒ (Giữ nguyên cấu trúc vẽ)
    # ==========================================
    fig = go.Figure()
    
    fill_colors = ['rgba(44, 123, 219, 0.35)', 'rgba(255, 136, 0, 0.35)', 'rgba(38, 166, 91, 0.35)']
    line_colors = ['rgb(44, 123, 219)', 'rgb(255, 136, 0)', 'rgb(38, 166, 91)']
    
    for idx, row in compare_df.iterrows():
        r_values = norm_df.iloc[idx].values.tolist()
        r_values.append(r_values[0]) 
        
        full_name = str(row[name_col])
        short_name = full_name[:35] + "..." if len(full_name) > 35 else full_name
        
        hover_texts = []
        for i, col in enumerate(cols):
            val = row[col]
            val_str = f"{val:,.1f}" if isinstance(val, float) else str(val)
            hover_texts.append(f"<b>{short_name}</b><br>{categories[i]}: <b>{val_str} {units[i]}</b>")
        hover_texts.append(hover_texts[0]) 
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=fill_colors[idx],
            mode='lines+markers', 
            line=dict(color=line_colors[idx], width=2),
            marker=dict(size=7, color=line_colors[idx]),
            name=short_name,
            hoverinfo="text",
            text=hover_texts
        ))
        
    # CĂN CHỈNH GIAO DIỆN VỚI RANGE ÂM TRÁNH CHỤM TÂM
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(245, 247, 250, 1)", 
            radialaxis=dict(
                visible=True, 
                showline=False, 
                range=[-0.25, 1.0], # Vẫn giữ trick range âm để UX đẹp
                showticklabels=False,
                gridcolor="white", 
                gridwidth=2
            ),
            angularaxis=dict(
                gridcolor="white", 
                linecolor="white",
                linewidth=2,
                ticks="outside",           
                ticklen=15,                
                tickcolor="rgba(0,0,0,0)"  
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, font=dict(size=12)
        ), 
        title=dict(
            text=f"<b>Specification comparison</b><br><sup>Segment: {segment_name}</sup>", 
            x=0.5, y=0.95, font=dict(size=18)
        ),
        margin=dict(l=100, r=100, t=100, b=80), 
        paper_bgcolor="white"
    )
    
    return fig