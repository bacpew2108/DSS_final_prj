"""
rf_charts.py
============
Module sinh các biểu đồ phân tích Random Forest Regression.
Tất cả hàm nhận đầu vào từ run_all_scenarios() và trả về Plotly Figure.

Danh sách biểu đồ:
  1. plot_mae_comparison       – Cột MAE + đường CV MAE ± Std (8 kịch bản)
  2. plot_r2_comparison        – Cột R² (%) cho 8 kịch bản
  3. plot_actual_vs_predicted  – Scatter Actual vs Predicted + đường y=x lý tưởng
  4. plot_feature_importance   – Cột ngang Feature Importance (Viridis)
  5. plot_residuals            – Histogram phân phối sai số (residuals)
  6. plot_learning_curve       – MAE theo số lượng cây (learning curve)
  7. plot_scenario_heatmap     – Heatmap siêu tham số × metric

Cách dùng:
    from modules.rf_charts import *
    from modules.random_forest_engine import run_all_scenarios

    rf = run_all_scenarios(df)
    fig = plot_mae_comparison(rf["results"], rf["best_result"])
    fig.show()          # nếu dùng script độc lập
    # hoặc
    st.plotly_chart(fig)  # nếu dùng trong Streamlit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Palette trắng / sáng ──────────────────────────────────────────────────────────────────────────────
_PAPER_BG  = "#ffffff"
_PLOT_BG   = "#f7f9fc"
_FONT      = dict(color="#1a202c", family="Inter, sans-serif")
_BLUE      = "#3182ce"
_GOLD      = "#d69e2e"
_GREEN     = "#38a169"
_RED       = "#e53e3e"
_PURPLE    = "#6b46c1"
_GRID_CLR  = "rgba(160,174,192,0.3)"

FEAT_LABELS = ["CPU Score", "GPU Score", "RAM (GB)", "Storage (GB)", "Cân nặng (kg)"]


def _base_layout(**kwargs) -> dict:
    """Layout Plotly nền trắng dùng chung."""
    return dict(
        template="plotly_white",
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        **kwargs,
    )


def _apply_grid(fig) -> None:
    """Thêm gridlines nhẹ cho trục x/y sau khi tạo figure."""
    fig.update_xaxes(gridcolor=_GRID_CLR, linecolor="#cbd5e0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor=_GRID_CLR, linecolor="#cbd5e0", zerolinecolor="#cbd5e0")


# =============================================================================
# 2. Biểu đồ so sánh MAE – 8 kịch bản
# =============================================================================

def plot_mae_comparison(results: list[dict], best_result: dict) -> go.Figure:
    """
    Biểu đồ cột MAE (Test) + đường CV MAE ± Std.
    Cột kịch bản tốt nhất được tô vàng.

    Parameters
    ----------
    results     : list[dict] từ run_all_scenarios()["results"]
    best_result : dict từ run_all_scenarios()["best_result"]

    Returns
    -------
    go.Figure
    """
    ids        = [f"#{r['scenario']['id']}\n{r['scenario']['name']}" for r in results]
    short_ids  = [f"#{r['scenario']['id']}" for r in results]
    mae_vals   = [r["mae"]          for r in results]
    cv_means   = [r["cv_mae_mean"]  for r in results]
    cv_stds    = [r["cv_mae_std"]   for r in results]
    best_id    = best_result["scenario"]["id"]

    bar_colors = [_GOLD if r["scenario"]["id"] == best_id else _BLUE for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE Test set",
        x=short_ids, y=mae_vals,
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in mae_vals],
        textposition="outside",
        customdata=ids,
        hovertemplate="<b>%{customdata}</b><br>MAE = %{y:.3f} Triệu VNĐ<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="CV MAE ± Std (5-fold)",
        x=short_ids, y=cv_means,
        mode="lines+markers",
        line=dict(color=_RED, width=2.5),
        error_y=dict(type="data", array=cv_stds, visible=True, color=_RED, thickness=1.5),
        marker=dict(size=9, color=_RED, symbol="diamond"),
        hovertemplate="CV MAE = %{y:.3f} ± σ Triệu<extra></extra>",
    ))

    best_mae = best_result["mae"]
    fig.add_hline(
        y=best_mae,
        line_dash="dot", line_color=_GOLD, line_width=1,
        annotation_text=f" Best MAE = {best_mae:.2f} Tr",
        annotation_font_color=_GOLD,
    )

    fig.update_layout(
        **_base_layout(height=460),
        title=dict(text="📉 So sánh MAE – 8 Biến thể Random Forest", font=dict(size=16)),
        xaxis_title="Kịch bản",
        yaxis_title="MAE (Triệu VNĐ)",
        legend=dict(x=0.01, y=0.99),
        bargap=0.35,
    )
    _apply_grid(fig)
    return fig


# =============================================================================
# 3. Biểu đồ R² – 8 kịch bản
# =============================================================================

def plot_r2_comparison(results: list[dict], best_result: dict) -> go.Figure:
    """
    Biểu đồ cột R² (%) cho 8 kịch bản.
    Cột tốt nhất tô vàng.
    """
    short_ids = [f"#{r['scenario']['id']}" for r in results]
    r2_vals   = [r["r2"] * 100 for r in results]
    best_id   = best_result["scenario"]["id"]
    bar_colors = [_GOLD if r["scenario"]["id"] == best_id else _GREEN for r in results]

    fig = go.Figure(go.Bar(
        name="R² (%)",
        x=short_ids, y=r2_vals,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in r2_vals],
        textposition="outside",
        hovertemplate="<b>#%{x}</b><br>R² = %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(height=400),
        title=dict(text="📈 Hệ số R² (%) – Khả năng giải thích phương sai giá", font=dict(size=16)),
        xaxis_title="Kịch bản",
        yaxis_title="R² (%)",
        yaxis=dict(range=[0, 115]),
        bargap=0.35,
    )
    _apply_grid(fig)
    return fig

# =============================================================================
# 1. Biểu đồ so sánh MAPE (%) – 8 kịch bản (BỔ SUNG MỚI)
# =============================================================================

def plot_mape_comparison(results: list[dict], best_result: dict) -> go.Figure:
    """
    Biểu đồ cột MAPE (%) cho 8 kịch bản.
    Cột tốt nhất được tô màu vàng.
    """
    short_ids = [f"#{r['scenario']['id']}" for r in results]
    # mape đang ở dạng tỷ lệ (0-1), nhân 100 để đổi sang đơn vị %
    mape_vals = [r["mape"] * 100 for r in results] 
    best_id   = best_result["scenario"]["id"]
    
    # Sai số càng thấp càng tốt -> Tốt nhất tô vàng, còn lại tô màu Tím
    bar_colors = [_GOLD if r["scenario"]["id"] == best_id else _PURPLE for r in results]

    fig = go.Figure(go.Bar(
        name="MAPE (%)",
        x=short_ids, y=mape_vals,
        marker_color=bar_colors,
        text=[f"{v:.2f}%" for v in mape_vals],
        textposition="outside",
        hovertemplate="<b>#%{x}</b><br>MAPE = %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(height=400),
        title=dict(text="📉 Sai số phần trăm tương đối trung bình MAPE (%) – 8 Biến thể", font=dict(size=16)),
        xaxis_title="Kịch bản",
        yaxis_title="MAPE (%)",
        bargap=0.35,
    )
    _apply_grid(fig)
    return fig

# =============================================================================
# 4. Actual vs Predicted scatter
# =============================================================================

def plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scenario_name: str = "",
    mae: float | None = None,
    mape: float | None = None,
    r2: float | None = None,
) -> go.Figure:
    """
    Scatter plot Giá thực tế vs Giá dự đoán + đường y = x lý tưởng.

    Parameters
    ----------
    y_test        : mảng giá thực tế
    y_pred        : mảng giá dự đoán
    scenario_name : tên kịch bản hiển thị trên tiêu đề
    mae, r2       : metric để hiển thị annotation
    """
    max_val = max(y_test.max(), y_pred.max()) * 1.08
    residuals = y_pred - y_test

    fig = go.Figure()

    # Scatter điểm
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(
            color=np.abs(residuals),
            colorscale="RdYlGn_r",
            size=7, opacity=0.75,
            colorbar=dict(title="Sai số (Tr)", thickness=12),
            showscale=True,
        ),
        name="Laptop",
        hovertemplate=(
            "Thực tế: %{x:.1f} Tr<br>"
            "Dự đoán: %{y:.1f} Tr<br>"
            "Sai số: %{marker.color:.1f} Tr<extra></extra>"
        ),
    ))

    # Đường y = x lý tưởng
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=_GOLD, dash="dash", width=2),
        name="Lý tưởng (y = x)",
        hoverinfo="skip",
    ))

    # Annotation metric
    if mae is not None and mape is not None and r2 is not None:
        fig.add_annotation(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=f"MAE = {mae:.2f} Tr  |  MAPE = {mape*100:.2f}%  |  R² = {r2*100:.1f}%",
            showarrow=False,
            font=dict(color=_GOLD, size=12),
            bgcolor="rgba(26,32,44,0.7)",
            bordercolor=_GOLD, borderwidth=1,
            borderpad=6,
        )

    title_suffix = f" – {scenario_name}" if scenario_name else ""
    fig.update_layout(
        **_base_layout(height=430),
        title=dict(text=f"🎯 Actual vs Predicted{title_suffix}", font=dict(size=16)),
        xaxis_title="Giá thực tế (Triệu VNĐ)",
        yaxis_title="Giá dự đoán (Triệu VNĐ)",
        legend=dict(x=0.01, y=0.99),
    )
    _apply_grid(fig)
    return fig


# =============================================================================
# 5. Feature Importance
# =============================================================================

def plot_feature_importance(
    model,
    scenario_name: str = "",
    feat_labels: list[str] | None = None,
) -> go.Figure:
    """
    Biểu đồ cột ngang Feature Importance với gradient màu Viridis.

    Parameters
    ----------
    model         : RandomForestRegressor đã huấn luyện
    scenario_name : tên kịch bản
    feat_labels   : nhãn tiếng Việt cho từng đặc trưng
    """
    if feat_labels is None:
        feat_labels = FEAT_LABELS

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feat_labels[i] for i in sorted_idx],
        orientation="h",
        marker=dict(
            color=importances[sorted_idx],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Tầm quan trọng", thickness=12),
        ),
        text=[f"{v*100:.1f}%" for v in importances[sorted_idx]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    top_feat = feat_labels[int(np.argmax(importances))]
    title_suffix = f" – {scenario_name}" if scenario_name else ""
    fig.update_layout(
        **_base_layout(height=380),
        title=dict(text=f"🌿 Feature Importance{title_suffix}", font=dict(size=16)),
        xaxis_title="Mức độ quan trọng",
        annotations=[dict(
            x=0.99, y=0.02, xref="paper", yref="paper",
            text=f"Yếu tố quan trọng nhất: <b>{top_feat}</b>",
            showarrow=False,
            font=dict(color=_GOLD, size=11),
            bgcolor="rgba(26,32,44,0.7)",
            bordercolor=_GOLD, borderwidth=1, borderpad=5,
            align="right",
        )],
    )
    return fig


# =============================================================================
# 5. Phân phối sai số (Residuals)
# =============================================================================

def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scenario_name: str = "",
) -> go.Figure:
    """
    Histogram phân phối sai số dự đoán (residuals = y_pred – y_test).
    Đường đứt tại 0 và tại ± MAE.
    """
    residuals = y_pred - y_test
    mae = float(np.abs(residuals).mean())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker=dict(
            color=_BLUE,
            opacity=0.75,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        name="Sai số",
        hovertemplate="Sai số: %{x:.1f} Tr<br>Số lượng: %{y}<extra></extra>",
    ))

    # Đường trung tâm
    fig.add_vline(x=0,    line_dash="dash", line_color=_GOLD, line_width=2,annotation_text=" Lý tưởng (0)", annotation_font_color=_GOLD)
    fig.add_vline(x=mae,  line_dash="dot",  line_color=_RED,  line_width=1.5,annotation_text=f" +MAE={mae:.1f}", annotation_font_color=_RED)
    fig.add_vline(x=-mae, line_dash="dot",  line_color=_RED,  line_width=1.5,annotation_text=f" -MAE", annotation_font_color=_RED)

    title_suffix = f" – {scenario_name}" if scenario_name else ""
    fig.update_layout(
        **_base_layout(height=380),
        title=dict(text=f"📊 Phân phối Sai số (Residuals){title_suffix}", font=dict(size=16)),
        xaxis_title="Sai số dự đoán (Triệu VNĐ)",
        yaxis_title="Số lượng laptop",
        showlegend=False,
    )
    _apply_grid(fig)
    return fig


# =============================================================================
# 6. Learning Curve (MAE theo n_estimators)
# =============================================================================

def plot_learning_curve(df, test_size: float = 0.2) -> go.Figure:
    """
    Vẽ Learning Curve: MAE (Test) theo số lượng cây (n_estimators).
    Sử dụng kịch bản Combo Tối ưu 2 làm base, thay đổi n_estimators từ 10 → 500.

    Parameters
    ----------
    df        : DataFrame dữ liệu laptop đã làm sạch
    test_size : tỉ lệ test set (mặc định 20%)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from modules.model_engine import prepare_rf_data

    X, y = prepare_rf_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    n_trees_list = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    mae_train_list, mae_test_list = [], []

    for n in n_trees_list:
        m = RandomForestRegressor(
            n_estimators=n, max_depth=15, min_samples_split=5, random_state=42
        )
        m.fit(X_train, y_train)
        mae_train_list.append(mean_absolute_error(y_train, m.predict(X_train)))
        mae_test_list.append(mean_absolute_error(y_test,  m.predict(X_test)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_trees_list, y=mae_train_list,
        mode="lines+markers", name="Train MAE",
        line=dict(color=_GREEN, width=2.5),
        marker=dict(size=8),
        hovertemplate="n=%{x} cây<br>Train MAE = %{y:.3f} Tr<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=n_trees_list, y=mae_test_list,
        mode="lines+markers", name="Test MAE",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=8),
        hovertemplate="n=%{x} cây<br>Test MAE = %{y:.3f} Tr<extra></extra>",
    ))

    # Điểm hội tụ (n cây mà test MAE thấp nhất)
    best_n = n_trees_list[int(np.argmin(mae_test_list))]
    best_mae = min(mae_test_list)
    fig.add_vline(
        x=best_n, line_dash="dot", line_color=_GOLD, line_width=1.5,
        annotation_text=f" Tốt nhất: {best_n} cây<br> MAE={best_mae:.2f} Tr",
        annotation_font_color=_GOLD,
    )

    fig.update_layout(
        **_base_layout(height=420),
        title=dict(text="📈 Learning Curve – MAE theo số lượng cây", font=dict(size=16)),
        xaxis_title="Số lượng cây (n_estimators)",
        yaxis_title="MAE (Triệu VNĐ)",
        legend=dict(x=0.7, y=0.95),
    )
    _apply_grid(fig)
    return fig


# =============================================================================
# 7. Heatmap siêu tham số × metric
# =============================================================================

def plot_scenario_heatmap(results: list[dict]) -> go.Figure:
    """
    Heatmap ma trận [kịch bản × metric].
    Giúp nhìn tổng thể hiệu năng của 8 kịch bản trên 3 chiều cùng lúc.

    Metrics: MAE (Test), CV MAE, R²
    """
    sc_names = [f"#{r['scenario']['id']} {r['scenario']['name']}" for r in results]
    metrics  = ["MAE (Test)", "MAPE (%)", "R² (%)"]

    # Chuẩn hóa min-max mỗi metric về [0, 1] để so sánh trên cùng thang
    raw = {
        "MAE (Test)":     np.array([r["mae"]          for r in results]),
        "MAPE (%)":       np.array([r["mape"] * 100   for r in results]),
        "R² (%)":         np.array([r["r2"] * 100     for r in results]),
    }

    # MAE: thấp = tốt → đảo chiều để vẽ heatmap (xanh = tốt)
    def norm(arr, invert=False):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        n = (arr - mn) / (mx - mn)
        return (1 - n) if invert else n

    z = np.column_stack([
        norm(raw["MAE (Test)"],      invert=True),
        norm(raw["MAPE (%)"],        invert=True),
        norm(raw["R² (%)"],          invert=False),
    ])

    # Annotation text = giá trị thực
    text = []
    for r in results:
        text.append([
            f"{r['mae']:.2f}",
            f"{r['mape']*100:.2f}%",
            f"{r['r2']*100:.1f}%",
        ])

    fig = go.Figure(go.Heatmap(
        z=z,
        x=metrics,
        y=sc_names,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale="RdYlGn",
        showscale=True,
        colorbar=dict(
            title="Hiệu năng<br>(chuẩn hóa)",
            thickness=14,
            tickvals=[0, 0.5, 1],
            ticktext=["Kém", "TB", "Tốt"],
        ),
        hovertemplate="<b>%{y}</b><br>%{x} = %{text}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(height=420),
        title=dict(
            text="🗺️ Heatmap Hiệu năng – 8 Kịch bản × 3 Metrics (chuẩn hóa)",
            font=dict(size=16),
        ),
        xaxis=dict(side="top"),
        margin=dict(l=260),
    )
    _apply_grid(fig)
    return fig


# =============================================================================
# Hàm tiện ích: sinh tất cả biểu đồ và lưu ra file PNG
# =============================================================================

# – Cấu hình xuất ảnh –
_PNG_SCALE  = 2        # độ phân giải nhân 2x (retina-quality)
_PNG_WIDTH  = 1400     # pixel
_PNG_HEIGHT = None     # None → dùng height từ fig.layout.height

# Map tên file → mô tả (dùng khi in log)
_CHART_META = {
    "rf_1_mae_comparison":     "So sánh MAE – 8 biến thể",
    "rf_2_r2_comparison":      "So sánh R² (%) – 8 biến thể",
    "rf_3_mape_comparison":    "So sánh MAPE (%) – 8 biến thể",
    "rf_4_actual_vs_predicted":"Actual vs Predicted (mô hình tốt nhất)",
    "rf_5_feature_importance": "Feature Importance (mô hình tốt nhất)",
    "rf_6_residuals":          "Phân phối sai số (Residuals)",
    "rf_7_learning_curve":     "Learning Curve – MAE theo số cây",
    "rf_8_scenario_heatmap":   "Heatmap hiệu năng 8 kịch bản",
}


def save_all_charts(
    rf_output: dict,
    df,
    out_dir: str = "assets",
    scale: int = _PNG_SCALE,
    width: int = _PNG_WIDTH,
) -> dict[str, str]:
    """
    Sinh 7 biểu đồ RF và lưu ra PNG vào thư mục ``out_dir`` (mặc định: assets/).
    Yêu cầu: kaleido được cài (đã thêm vào requirements.txt).

    Parameters
    ----------
    rf_output : dict trả về từ run_all_scenarios()
    df        : DataFrame dữ liệu laptop đã làm sạch
    out_dir   : thư mục đầu ra (tạo nếu chưa có)
    scale     : hệ số scale của PNG (2 = retina)
    width     : chiều rộng ảnh pixel

    Returns
    -------
    dict mapping tên biểu đồ → đưᤁng dẫn file PNG đã lưu
    """
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results  = rf_output["results"]
    best_res = rf_output["best_result"]
    y_test   = rf_output["y_test"]

    # Xây dựng 7 figure
    figures = {
        "rf_1_mae_comparison":     plot_mae_comparison(results, best_res),
        "rf_2_r2_comparison":      plot_r2_comparison(results, best_res),
        "rf_3_mape_comparison":    plot_mape_comparison(results, best_res),
        "rf_4_actual_vs_predicted":plot_actual_vs_predicted(
                                       y_test, best_res["y_pred"],
                                       best_res["scenario"]["name"],
                                       best_res["mae"], best_res["mape"], best_res["r2"]),
        "rf_5_feature_importance": plot_feature_importance(
                                       best_res["model"],
                                       best_res["scenario"]["name"]),
        "rf_6_residuals":          plot_residuals(
                                       y_test, best_res["y_pred"],
                                       best_res["scenario"]["name"]),
        "rf_7_learning_curve":     plot_learning_curve(df),
        "rf_8_scenario_heatmap":   plot_scenario_heatmap(results),
    }

    saved_paths: dict[str, str] = {}

    print(f"\n🎨 Xuất {len(figures)} biểu đồ → {out_path.resolve()}/")
    print("-" * 60)

    for name, fig in figures.items():
        # Đảm bảo nền trắng khi xuất PNG
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f7f9fc",
            font=dict(color="#1a202c"),
        )

        png_path = out_path / f"{name}.png"
        fig_height = fig.layout.height or 450

        fig.write_image(
            str(png_path),
            format="png",
            width=width,
            height=int(fig_height * scale * 0.75),
            scale=scale,
        )

        desc = _CHART_META.get(name, name)
        print(f"  ✅ {name}.png  —  {desc}")
        saved_paths[name] = str(png_path)

    print("-" * 60)
    print(f"🌲 Đã lưu {len(saved_paths)} ảnh PNG vào '{out_dir}/'")
    return saved_paths


# =============================================================================
# Entry point: chạy trực tiếp bằng `python modules/rf_charts.py`
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Thêm root vào sys.path để import được modules
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))

    import pandas as pd
    from modules.model_engine import run_all_scenarios

    print("=" * 60)
    print("  RF Charts Generator – DSS Laptop Project")
    print("=" * 60)

    print("\n📥 Đọc dữ liệu từ data/laptops_dataset_cleaned.csv ...")
    df = pd.read_csv(root / "data" / "laptops_dataset_cleaned.csv")
    if df["price"].max() > 1000:
        df["price"] = df["price"] / 1_000_000
    print(f"   ✔ {len(df)} laptop đã được nạp.")

    print("\n🥊 Huấn luyện 8 mô hình Random Forest Regressor...")
    rf_output = run_all_scenarios(df, test_size=0.2)

    # Lưu PNG vào assets/
    assets_dir = str(root / "assets")
    save_all_charts(rf_output, df, out_dir=assets_dir)

    print("\n🎉 Xong! Kiểm tra thư mục assets/ để xem ảnh.")
