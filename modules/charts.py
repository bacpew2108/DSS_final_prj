import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_radar_chart(top_df, segment_name, name_col='product_name', top_n=3):
    # ==========================================
    # BỘ KIỂM TRA AN TOÀN (SAFETY CHECK) - CHỐNG SẬP APP
    # ==========================================
    if top_df is None or len(top_df) == 0:
        # Nếu bộ lọc trống không có máy nào, trả về biểu đồ trắng kèm thông báo thân thiện
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="<b>Không tìm thấy laptop phù hợp với bộ lọc hiện tại!</b><br><span style='font-size: 14px; color: gray;'>Vui lòng nới lỏng bộ lọc ở cột trái.</span>", 
                x=0.5, y=0.5, font=dict(size=18, color="red")
            ),
            paper_bgcolor="white",
            margin=dict(l=40, r=40, t=100, b=40)
        )
        return fig

    # Đảm bảo số lượng máy vẽ không vượt quá số lượng máy thực tế và TỐI ĐA chỉ vẽ 5 máy để tránh tràn bộ màu
    top_n = min(top_n, len(top_df), 5)
    
    # Lấy đúng số lượng máy cần vẽ
    compare_df = top_df.head(top_n).copy()
    
    categories = ['CPU Score', 'GPU Score', 'RAM', 'Storage', 'Price', 'Weight']
    cols = ['cpu_point', 'gpu_point', 'ram_capacity', 'storage', 'price', 'weight']
    units = ['Pts', 'Pts', 'GB', 'GB', 'Mil VND', 'Kg']
    
    # HỆ QUY CHIẾU TOÀN CẦU
    global_anchors = {
        'cpu_point': (30.0, 90.0),  
        'gpu_point': (10.0, 80.0)   
    }

    # BỘ TIÊU CHUẨN PHÂN KHÚC 
    segment_anchors = {
        "Rẻ (< 15 Triệu)": {
            'price': (4.5, 15.0), 'ram_capacity': (4, 16), 'storage': (256, 512), 'weight': (1.2, 2.5)
        },
        "Phổ thông (15 - 25 Triệu)": {
            'price': (15.0, 25.0), 'ram_capacity': (8, 24), 'storage': (512, 1024), 'weight': (1.2, 2.7)
        },
        "Cao cấp (> 25 Triệu)": {
            'price': (25.0, 170.0), 'ram_capacity': (16, 64), 'storage': (512, 2048), 'weight': (1.2, 3.2)
        }
    }
    
    current_anchor = segment_anchors.get(segment_name, segment_anchors["Phổ thông (15 - 25 Triệu)"])
    
    # TÍNH TOÁN NORMALIZATION
    norm_data = {}
    for col in cols:
        vals = compare_df[col].values
        
        if col in ['cpu_point', 'gpu_point']:
            a_min, a_max = global_anchors[col]
        else:
            a_min, a_max = current_anchor[col]

        if col in ['price', 'weight']: 
            norm_v = (a_max - vals) / (a_max - a_min)
        else: 
            norm_v = (vals - a_min) / (a_max - a_min)
            
        norm_data[col] = np.clip(norm_v, 0.0, 1.0).tolist()
                
    norm_df = pd.DataFrame(norm_data)
    
    # KHỞI TẠO VÀ VẼ BIỂU ĐỒ
    fig = go.Figure()
    
    # BẢNG 5 MÀU TƯƠNG PHẢN ĐẸP (Phục vụ vẽ từ 1 đến 5 máy)
    fill_colors = [
        'rgba(44, 123, 219, 0.25)',  # Xanh dương (Top 1)
        'rgba(255, 136, 0, 0.25)',   # Cam (Top 2)
        'rgba(38, 166, 91, 0.25)',    # Xanh lá (Top 3)
        'rgba(155, 89, 182, 0.25)',   # Tím (Top 4)
        'rgba(231, 76, 60, 0.25)'     # Đỏ (Top 5)
    ]
    line_colors = [
        'rgb(44, 123, 219)',
        'rgb(255, 136, 0)',
        'rgb(38, 166, 91)',
        'rgb(155, 89, 182)',
        'rgb(231, 76, 60)'
    ]
    ranks = ["Top 1", "Top 2", "Top 3", "Top 4", "Top 5"]
    
    for idx, row in compare_df.reset_index(drop=True).iterrows():
        r_values = norm_df.iloc[idx].values.tolist()
        r_values.append(r_values[0]) 
        
        full_name = str(row[name_col])
        short_name = full_name[:35] + "..." if len(full_name) > 35 else full_name
        legend_name = f"<b>{ranks[idx]}</b>: {short_name}"
        
        hover_texts = []
        for i, col in enumerate(cols):
            val = row[col]
            val_str = f"{val:,.1f}" if isinstance(val, float) else str(val)
            hover_texts.append(f"<b>{ranks[idx]}</b><br>{categories[i]}: <b>{val_str} {units[i]}</b>")
        hover_texts.append(hover_texts[0]) 
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=fill_colors[idx],
            mode='lines+markers', 
            line=dict(color=line_colors[idx], width=2.5),
            marker=dict(size=8, color=line_colors[idx]),
            name=legend_name,
            hoverinfo="text",
            text=hover_texts
        ))
        
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(245, 247, 250, 1)", 
            radialaxis=dict(visible=True, showline=False, range=[-0.25, 1.0], showticklabels=False, gridcolor="white", gridwidth=2),
            angularaxis=dict(gridcolor="white", linecolor="white", linewidth=2, ticks="outside", ticklen=20, tickcolor="rgba(0,0,0,0)")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=13)), 
        # Cập nhật tiêu đề tự động theo số lượng máy vẽ thực tế
        title=dict(text=f"<b>Radar Comparison - Top {top_n} Best Laptops</b><br><span style='font-size: 14px; color: gray;'>Segment: {segment_name}</span>", x=0.5, y=0.96, font=dict(size=20)),
        margin=dict(l=80, r=80, t=100, b=80), paper_bgcolor="white"
    )
    
    return fig