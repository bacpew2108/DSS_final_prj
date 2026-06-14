import plotly.graph_objects as go
import pandas as pd

def plot_radar_chart(top_df, name_col='product_name'):
    """
    Vẽ biểu đồ Mạng nhện (Radar Chart) so sánh độ toàn diện của Top 3 Laptop.
    Đã đẩy chữ ra xa bằng thủ thuật ticks="outside" + ticklen=15 (ẩn màu).
    """
    compare_df = top_df.head(3).copy()
    
    categories = ['Price', 'RAM', 'Storage', 'CPU Score', 'GPU Score', 'Weight']
    cols = ['price', 'ram_capacity', 'storage', 'cpu_point', 'gpu_point', 'weight']
    units = ['Million VND', 'GB', 'GB', 'Pts', 'Pts', 'Kg']
    
    # CHUẨN HÓA DỮ LIỆU (CÓ MỨC SÀN)
    norm_data = {}
    base_val = 0.20 
    scale = 1.0 - base_val 
    
    for col in cols:
        vals = compare_df[col].values
        min_v, max_v = vals.min(), vals.max()
        
        if max_v == min_v:
            norm_data[col] = [1.0] * len(vals)
        else:
            if col in ['price', 'weight']: 
                norm_data[col] = base_val + ((max_v - vals) / (max_v - min_v)) * scale
            else: 
                norm_data[col] = base_val + ((vals - min_v) / (max_v - min_v)) * scale
                
    norm_df = pd.DataFrame(norm_data)
    
    # KHỞI TẠO BIỂU ĐỒ
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
        
    # CĂN CHỈNH GIAO DIỆN
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(245, 247, 250, 1)", 
            radialaxis=dict(
                visible=True, 
                showline=False, 
                range=[0, 1], 
                showticklabels=False,
                gridcolor="white", 
                gridwidth=2
            ),
            
            angularaxis=dict(
                gridcolor="white", 
                linecolor="white",
                linewidth=2,
                ticks="outside",           # Tạo vạch chia hướng ra ngoài
                ticklen=15,                # Độ dài vạch chia 15px giúp đẩy chữ ra xa
                tickcolor="rgba(0,0,0,0)"  # Ẩn vạch chia đi bằng màu trong suốt
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.18, 
            xanchor="center", 
            x=0.5,
            font=dict(size=12)
        ), 
        title=dict(
            text="<b>Top 3 laptops specification comparison</b>", 
            x=0.5,
            y=0.95,
            font=dict(size=20)
        ),
        margin=dict(l=100, r=100, t=100, b=80), 
        paper_bgcolor="white"
    )
    
    return fig