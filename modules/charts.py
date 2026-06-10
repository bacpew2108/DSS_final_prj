import plotly.graph_objects as go
import pandas as pd

# CẤU HÌNH MÀU SẮC CYBER-DARK
BG_COLOR = "#0B0F19"
CARD_COLOR = "#111827"
GRID_COLOR = "#374151"
TEXT_COLOR = "#F3F4F6"
NEON_COLORS = ['#00F2FE', '#F355DA', '#00F5A0', '#A855F7', '#F59E0B', '#3B82F6']

# --- 1. BIỂU ĐỒ RADAR (ĐA GIÁC ĐỒNG TÂM) ---
def draw_radar_chart(top3_df):
    df_scaled = top3_df.copy()
    max_cols = ['ram_capacity', 'storage', 'cpu_point', 'gpu_point'] 
    min_cols = ['price', 'weight'] 
    col_names = ['price', 'ram_capacity', 'weight', 'storage', 'cpu_point', 'gpu_point']
    categories = ['GIÁ RẺ', 'RAM', 'MỎNG NHẸ', 'Ổ CỨNG', 'CPU', 'GPU']

    for col in col_names:
        c_min, c_max = df_scaled[col].min(), df_scaled[col].max()
        if c_max == c_min:
            df_scaled[col] = 80
        else:
            if col in max_cols:
                df_scaled[col] = ((df_scaled[col] - c_min) / (c_max - c_min)) * 70 + 25
            elif col in min_cols:
                df_scaled[col] = ((c_max - df_scaled[col]) / (c_max - c_min)) * 70 + 25

    fig = go.Figure()
    laptop_names = top3_df['Tên_Máy'].tolist() if 'Tên_Máy' in top3_df.columns else [f"Top {i+1} Match" for i in range(len(top3_df))]

    for i in range(len(df_scaled)):
        values = df_scaled[col_names].iloc[i].tolist()
        values.append(values[0])
        row_origin = top3_df.iloc[i]
        real_vals = [
            f"{int(row_origin['price']):,} VND" if 'price' in row_origin else "N/A",
            f"{int(row_origin['ram_capacity'])} GB" if 'ram_capacity' in row_origin else "N/A",
            f"{row_origin['weight']} kg" if 'weight' in row_origin else "N/A",
            f"{int(row_origin['storage'])} GB" if 'storage' in row_origin else "N/A",
            f"{int(row_origin['cpu_point'])} pts" if 'cpu_point' in row_origin else "N/A",
            f"{int(row_origin['gpu_point'])} pts" if 'gpu_point' in row_origin else "N/A"
        ]
        real_vals.append(real_vals[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=str(laptop_names[i]),
            line=dict(width=3, color=NEON_COLORS[i % len(NEON_COLORS)]),
            marker=dict(size=8, symbol="circle", line=dict(color='white', width=1)),
            opacity=0.2,
            customdata=real_vals,
            hovertemplate="<b>%{theta}</b><br>Chỉ số: %{r:.1f}/100<br>Thực tế: <b>%{customdata}</b><extra></extra>"
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_COLOR,
            gridshape='linear', # <--- ĐÃ SỬA: 'linear' thay vì 'polygon'
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR, showticklabels=False, ticks="", showline=False),
            angularaxis=dict(showgrid=False, linecolor=GRID_COLOR, ticklen=20, tickcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#9CA3AF"))
        ),
        paper_bgcolor=BG_COLOR,
        font=dict(family="Courier New, monospace", size=12, color=TEXT_COLOR),
        margin=dict(t=100, b=80, l=50, r=50),
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
        title=dict(text="<b>ANALYSIS MATRIX</b><br><span style='font-size:11px;color:#6B7280;'>So sánh tương quan sức mạnh sản phẩm</span>", x=0.5, y=0.95)
    )
    return fig

# --- 2. BIỂU ĐỒ BÁNH DONUT (HỒ SƠ NHU CẦU) ---
def draw_weights_donut_chart(weights_array):
    labels = ['Giá tiền', 'RAM', 'Cân nặng', 'Ổ cứng', 'CPU', 'GPU']
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=weights_array, hole=.7,
        marker=dict(colors=NEON_COLORS, line=dict(color=CARD_COLOR, width=3)),
        textinfo='percent', pull=[0.01] * 6
    )])
    fig.add_annotation(text="<b>TARGET</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color=TEXT_COLOR))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, font=dict(family="Courier New", color=TEXT_COLOR),
        title=dict(text="<b>USER PREFERENCE</b>", x=0.5, y=0.9),
        showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(t=100, b=50, l=20, r=20)
    )
    return fig

# --- 3. BIỂU ĐỒ ĐỒNG HỒ (GAUGE) ---
def draw_match_score_gauge(top1_score):
    score_100 = round(top1_score * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score_100,
        number={'suffix': "%", 'font': {'size': 50, 'color': '#00F5A0'}},
        gauge={
            'axis': {'range': [0, 100], 'ticks': ""},
            'bar': {'color': '#00F2FE', 'thickness': 0.3},
            'bgcolor': CARD_COLOR, 'borderwidth': 0,
            'threshold': {'line': {'color': "#00F5A0", 'width': 3}, 'value': score_100}
        }
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, height=320, margin=dict(t=100, b=20, l=30, r=30),
        title=dict(text="<b>MATCH SUITABILITY</b>", x=0.5, font=dict(color=TEXT_COLOR))
    )
    return fig