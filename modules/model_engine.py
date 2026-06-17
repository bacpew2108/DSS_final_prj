"""
random_forest_engine.py
=======================
Module Random Forest Regression với 8 biến thể siêu tham số
cho bài toán DỰ ĐOÁN GIÁ laptop (đơn vị: Triệu VNĐ).

Phương pháp luận: So sánh 8 cấu hình, chọn mô hình tốt nhất theo MAE thấp nhất.

Đặc trưng (features):  cpu_point, gpu_point, ram_capacity, storage, weight
Target:                 price (Triệu VNĐ)
Metrics đánh giá:       MAE (Triệu VNĐ), R² (%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

FEATURE_COLS = ['cpu_point', 'gpu_point', 'ram_capacity', 'storage', 'weight']
TARGET_COL   = 'price'   # Đã là Triệu VNĐ sau khi load_data() xử lý

# ---------------------------------------------------------------------------
# 8 Kịch bản (biến thể) — giữ nguyên theo thiết kế gốc của nhóm
# ---------------------------------------------------------------------------

HYPERPARAMETER_SCENARIOS: list[dict] = [
    {
        "id": 1,
        "name": "RF Mặc định (Baseline)",
        "description": "Không chỉnh tham số — đường cơ sở để so sánh",
        "params": {"random_state": 42},
    },
    {
        "id": 2,
        "name": "Cây Nông (max_depth=5)",
        "description": "Giới hạn độ sâu = 5, mô hình đơn giản, tránh overfit",
        "params": {"max_depth": 5, "random_state": 42},
    },
    {
        "id": 3,
        "name": "Cây Sâu (max_depth=20)",
        "description": "Cho phép cây mọc sâu tới 20 tầng, học chi tiết hơn",
        "params": {"max_depth": 20, "random_state": 42},
    },
    {
        "id": 4,
        "name": "Rừng Thưa (10 cây)",
        "description": "Chỉ 10 cây — huấn luyện nhanh, dự đoán kém ổn định",
        "params": {"n_estimators": 10, "random_state": 42},
    },
    {
        "id": 5,
        "name": "Rừng Rậm (300 cây)",
        "description": "300 cây — ổn định cao, tốn thời gian huấn luyện hơn",
        "params": {"n_estimators": 300, "random_state": 42},
    },
    {
        "id": 6,
        "name": "Rẽ nhánh khắt khe (min_split=10)",
        "description": "Cần tối thiểu 10 mẫu để tách nút, giảm nhiễu",
        "params": {"min_samples_split": 10, "random_state": 42},
    },
    {
        "id": 7,
        "name": "XGBoost (State-of-the-Art)",
        "description": "Mô hình XGBoost mạnh mẽ, tối ưu hóa",
        "model_class": XGBRegressor,
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "random_state": 42},
    },
    {
        "id": 8,
        "name": "LightGBM (State-of-the-Art)",
        "description": "Mô hình LightGBM tốc độ cao, hiệu quả",
        "model_class": LGBMRegressor,
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "random_state": 42, "verbose": -1},
    },
]

# ---------------------------------------------------------------------------
# Tiền xử lý dữ liệu
# ---------------------------------------------------------------------------

def prepare_rf_data(df: pd.DataFrame):
    """
    Chuẩn bị X, y từ DataFrame đã làm sạch.
    Trả về (X, y) — đã lọc NaN.
    """
    needed = FEATURE_COLS + [TARGET_COL]
    data = df[needed].dropna()

    # price đã là Triệu VNĐ (load_data chia /1_000_000)
    X = data[FEATURE_COLS]
    y = data[TARGET_COL].values
    return X, y


# ---------------------------------------------------------------------------
# Chạy toàn bộ 8 kịch bản — đúng theo thiết kế gốc của nhóm
# ---------------------------------------------------------------------------

def run_all_scenarios(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Huấn luyện và so sánh 8 biến thể RandomForestRegressor.

    Returns
    -------
    dict:
        "results"       : list[dict] — kết quả từng mô hình
        "summary_df"    : pd.DataFrame — bảng leaderboard (sort theo MAE)
        "best_model"    : RandomForestRegressor — mô hình vô địch
        "best_result"   : dict — thông tin mô hình vô địch
        "X_test"        : array
        "y_test"        : array
        "y_pred_best"   : array — dự đoán của mô hình tốt nhất
        "feature_names" : list[str]
    """
    X, y = prepare_rf_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = []
    best_mae   = float('inf')
    best_result = None

    print("🥊 BẮT ĐẦU HUẤN LUYỆN 8 BIẾN THỂ RANDOM FOREST...")
    print("-" * 75)

    for scenario in HYPERPARAMETER_SCENARIOS:
        model_class = scenario.get("model_class", RandomForestRegressor)
        model = model_class(**scenario["params"])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        mape   = mean_absolute_percentage_error(y_test, y_pred)
        
        # Cross-validation MAE (5-fold)
        cv_neg_mae = cross_val_score(
            model_class(**scenario["params"]),
            X, y, cv=5, scoring="neg_mean_absolute_error"
        )
        cv_mae_mean = float(-cv_neg_mae.mean())
        cv_mae_std  = float(cv_neg_mae.std())

        res = {
            "scenario": scenario,
            "model":    model,
            "mae":      mae,
            "r2":       r2,
            "mape":     mape,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std":  cv_mae_std,
            "y_pred":   y_pred,
        }
        results.append(res)

        print(f"  #{scenario['id']:>2} {scenario['name']:<42} MAE={mae:.2f}M  MAPE={mape*100:.2f}%  R²={r2*100:.1f}%")

        if mae < best_mae:
            best_mae    = mae
            best_result = res



    # Leaderboard (sort theo MAE tăng dần)
    summary_rows = []
    for r in results:
        sc = r["scenario"]
        summary_rows.append({
            "Biến thể Mô hình": f"#{sc['id']} {sc['name']}",
            "n_estimators": getattr(r["model"], "n_estimators", "N/A"),
            "max_depth": str(getattr(r["model"], "max_depth", None)) if getattr(r["model"], "max_depth", None) else "∞",
            "MAE (Triệu VNĐ)": round(r["mae"], 3),
            "MAPE (%)": round(r["mape"] * 100, 2),
            "R² (%)": round(r["r2"] * 100, 2),
            "CV MAE (Tr)": round(r["cv_mae_mean"], 3),
            "CV Std": round(r["cv_mae_std"], 3),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("MAE (Triệu VNĐ)").reset_index(drop=True)

    return {
        "results":       results,
        "summary_df":    summary_df,
        "best_model":    best_result["model"],
        "best_result":   best_result,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_pred_best":   best_result["y_pred"],
        "feature_names": FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# Dự đoán giá cho laptop tùy chỉnh
# ---------------------------------------------------------------------------

def predict_price(model, cpu_point: float, gpu_point: float,
                  ram_gb: float, storage_gb: float, weight_kg: float) -> dict:
    """
    Dự đoán giá (Triệu VNĐ) cho một cấu hình laptop.

    Returns
    -------
    dict:
        "price_M"   : float — giá dự đoán (Triệu VNĐ)
        "segment"   : str   — phân khúc tương ứng
    """
    X_new = pd.DataFrame([[cpu_point, gpu_point, ram_gb, storage_gb, weight_kg]], columns=FEATURE_COLS)
    price = float(model.predict(X_new)[0])
    price = max(0.0, round(price, 2))

    if price < 15:
        segment = "Rẻ (<15 Triệu)"
    elif price <= 25:
        segment = "Phổ thông (15–25 Triệu)"
    else:
        segment = "Cao cấp (>25 Triệu)"

    return {"price_M": price, "segment": segment}
