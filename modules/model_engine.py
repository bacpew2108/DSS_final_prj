"""
model_engine.py
===============
Module huấn luyện & so sánh 8 mô hình hồi quy
cho bài toán DỰ ĐOÁN GIÁ laptop (đơn vị: Triệu VNĐ).

8 mô hình = 6 biến thể Random Forest + XGBoost + LightGBM:
  1. RF Mặc định (Baseline)               – Đường cơ sở, không chỉnh tham số
  2. Cây Nông (max_depth=5)               – Mô hình đơn giản, tránh overfit
  3. Cây Sâu (max_depth=20)               – Học chi tiết, mọc sâu 20 tầng
  4. Rừng Rậm (300 cây)                   – Nhiều cây, ổn định cao
  5. Combo Tối ưu 1 (100 cây, depth=10)   – Cân bằng tốc độ & chính xác
  6. Combo Tối ưu 2 (200 cây, depth=15)   – Ứng cử viên vô địch RF
  7. XGBoost (Tuned)                      – Gradient boosting state-of-the-art  ★ BẮT BUỘC
  8. LightGBM (Tuned)                     – Gradient boosting tốc độ cao        ★ BẮT BUỘC

Phương pháp luận: So sánh 8 mô hình, chọn tốt nhất theo Composite Score
                  (Average Rank trên 3 tiêu chí: MAE, MAPE, R²).

Đặc trưng (features):  cpu_point, gpu_point, ram_capacity, storage, weight
Target:                 price (Triệu VNĐ)
Metrics đánh giá:       MAE (Triệu VNĐ), MAPE (%), R² (%)
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
# 8 Kịch bản: 6 biến thể RF + XGBoost + LightGBM
# ---------------------------------------------------------------------------

HYPERPARAMETER_SCENARIOS: list[dict] = [
    # ── 1. RF Mặc định (Baseline) ────────────────────────────────────────
    {
        "id": 1,
        "name": "RF Mặc định (Baseline)",
        "description": "Không chỉnh tham số — đường cơ sở để so sánh",
        "params": {"random_state": 42},
    },
    # ── 2. Cây Nông ──────────────────────────────────────────────────────
    {
        "id": 2,
        "name": "Cây Nông (max_depth=5)",
        "description": "Giới hạn độ sâu = 5, mô hình đơn giản, tránh overfit",
        "params": {"max_depth": 5, "random_state": 42},
    },
    # ── 3. Cây Sâu ──────────────────────────────────────────────────────
    {
        "id": 3,
        "name": "Cây Sâu (max_depth=20)",
        "description": "Cho phép cây mọc sâu tới 20 tầng, học chi tiết hơn",
        "params": {"max_depth": 20, "random_state": 42},
    },
    # ── 4. Rừng Rậm ─────────────────────────────────────────────────────
    {
        "id": 4,
        "name": "Rừng Rậm (300 cây)",
        "description": "300 cây — ổn định cao, tốn thời gian huấn luyện hơn",
        "params": {"n_estimators": 300, "random_state": 42},
    },
    # ── 5. Rẽ nhánh khắt khe ────────────────────────────────────────────────
    {
        "id": 5,
        "name": "Rẽ nhánh khắt khe (min_split=10)",
        "description": "Cần tối thiểu 10 mẫu để tách nút, giảm nhiễu",
        "params": {"min_samples_split": 10, "random_state": 42},
    },
    # ── 6. Combo Tối ưu ────────────────────────────────────────────────
    {
        "id": 6,
        "name": "Combo Tối ưu (200 cây, depth=15, split=5)",
        "description": "Mô hình cân bằng toàn diện",
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
        },
    },
    # ── 7. XGBoost ────────────────────────────────────
    {
        "id": 7,
        "name": "XGBoost",
        "description": "Gradient boosting state-of-the-art, learning rate thấp + regularization mạnh",
        "model_class": XGBRegressor,
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42,
        },
    },
    # ── 8. LightGBM ──────────────────────────────────
    {
        "id": 8,
        "name": "LightGBM",
        "description": "Gradient boosting tốc độ cao, num_leaves lớn + feature fraction để tổng quát hoá",
        "model_class": LGBMRegressor,
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 5,
            "random_state": 42,
            "verbose": -1,
        },
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
# Helper: Tạo model từ scenario config
# ---------------------------------------------------------------------------

def _build_model(scenario: dict):
    """
    Trả về model object từ scenario config.
    Nếu scenario có model_class → dùng class đó (XGBoost, LightGBM).
    Nếu không → mặc định RandomForestRegressor.
    """
    model_class = scenario.get("model_class", RandomForestRegressor)
    return model_class(**scenario["params"])

# ---------------------------------------------------------------------------
# Helper: Tính rank cho composite scoring
# ---------------------------------------------------------------------------

def _rank_asc(values: list[float]) -> list[int]:
    """Rank tăng dần: giá trị nhỏ nhất → rank 1."""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks


def _rank_desc(values: list[float]) -> list[int]:
    """Rank giảm dần: giá trị lớn nhất → rank 1."""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks



def run_all_scenarios(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Huấn luyện và so sánh 8 mô hình hồi quy.

    Returns
    -------
    dict:
        "results"       : list[dict] — kết quả từng mô hình
        "summary_df"    : pd.DataFrame — bảng leaderboard (sort theo MAE)
        "best_model"    : model — mô hình vô địch
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

    print("🥊 BẮT ĐẦU HUẤN LUYỆN 8 MÔ HÌNH TỐI ƯU...")
    print("-" * 75)

    for scenario in HYPERPARAMETER_SCENARIOS:
        model = _build_model(scenario)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        mape   = mean_absolute_percentage_error(y_test, y_pred)
        
        # Cross-validation MAE (5-fold)
        cv_model = _build_model(scenario)
        cv_neg_mae = cross_val_score(
            cv_model, X, y, cv=5, scoring="neg_mean_absolute_error"
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

    # -------------------------------------------------------------------
    # COMPOSITE SCORE: Đánh giá tổng hợp bằng Average Rank (3 tiêu chí)
    #   - MAE  : rank tăng dần (thấp = tốt)
    #   - MAPE : rank tăng dần (thấp = tốt)
    #   - R²   : rank giảm dần (cao = tốt)
    #   → Composite Score = trung bình 3 rank → càng nhỏ càng tốt
    # -------------------------------------------------------------------
    n = len(results)

    # Tính rank cho từng metric
    mae_vals  = [r["mae"]  for r in results]
    mape_vals = [r["mape"] for r in results]
    r2_vals   = [r["r2"]   for r in results]

    mae_ranks  = _rank_asc(mae_vals)     # MAE  nhỏ → rank thấp
    mape_ranks = _rank_asc(mape_vals)    # MAPE nhỏ → rank thấp
    r2_ranks   = _rank_desc(r2_vals)     # R²   lớn → rank thấp

    best_composite = float('inf')
    best_result = None

    for i, res in enumerate(results):
        composite = (mae_ranks[i] + mape_ranks[i] + r2_ranks[i]) / 3.0
        res["composite_score"] = composite
        res["rank_mae"]  = mae_ranks[i]
        res["rank_mape"] = mape_ranks[i]
        res["rank_r2"]   = r2_ranks[i]

        if composite < best_composite:
            best_composite = composite
            best_result = res

    # In bảng rank chi tiết
    print("\n📊 BẢNG XẾP HẠNG TỔNG HỢP (3 tiêu chí):")
    print(f"  {'Mô hình':<42} {'Rank MAE':>9} {'Rank MAPE':>10} {'Rank R²':>8} {'Composite':>10}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: x["composite_score"]):
        sc = r["scenario"]
        print(f"  {sc['name']:<42} {r['rank_mae']:>9} {r['rank_mape']:>10} {r['rank_r2']:>8} {r['composite_score']:>10.2f}")

    # Leaderboard (sort theo Composite Score)
    summary_rows = []
    for r in results:
        sc = r["scenario"]
        model_obj = r["model"]

        n_est = getattr(model_obj, "n_estimators", "—")
        m_depth = getattr(model_obj, "max_depth", None)
        depth_str = str(m_depth) if m_depth is not None else "∞"

        summary_rows.append({
            "Mô hình": f"#{sc['id']} {sc['name']}",
            "n_estimators": n_est,
            "max_depth": depth_str,
            "MAE (Triệu VNĐ)": round(r["mae"], 3),
            "MAPE (%)": round(r["mape"] * 100, 2),
            "R² (%)": round(r["r2"] * 100, 2),
            "Composite Score": round(r["composite_score"], 2),
            "CV MAE (Tr)": round(r["cv_mae_mean"], 3),
            "CV Std": round(r["cv_mae_std"], 3),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Composite Score").reset_index(drop=True)

    print("\n" + "=" * 75)
    print(f"🏆 MÔ HÌNH TỐT NHẤT (Composite Score): #{best_result['scenario']['id']} {best_result['scenario']['name']}")
    print(f"   MAE = {best_result['mae']:.3f} Triệu  |  MAPE = {best_result['mape']*100:.2f}%  |  R² = {best_result['r2']*100:.1f}%")
    print(f"   Composite Score = {best_result['composite_score']:.2f}  (Rank MAE={best_result['rank_mae']}, Rank MAPE={best_result['rank_mape']}, Rank R²={best_result['rank_r2']})")
    print("=" * 75)

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
