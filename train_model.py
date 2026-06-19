import os
import joblib
import pandas as pd

from modules.model_engine import (
    FEATURE_COLS, TARGET_COL, HYPERPARAMETER_SCENARIOS,
    run_all_scenarios, _build_model,
)
from modules.model_gen_charts import save_all_charts

def train_and_save_model():
    print("⏳ Đang đọc dữ liệu...")
    df = pd.read_csv("data/laptops_dataset_cleaned.csv")
    
    # Chuẩn hóa giá về Triệu VNĐ nếu cần
    if df['price'].max() > 1000:
        df['price'] = df['price'] / 1000000.0

    # Lọc bỏ NaN
    needed_cols = FEATURE_COLS + [TARGET_COL]
    data = df[needed_cols].dropna()

    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    # Đánh giá tất cả kịch bản để tìm model tốt nhất
    print("🔍 Đang đánh giá 8 mô hình để tìm ra mô hình tốt nhất...\n")
    output = run_all_scenarios(df)
    
    # Xuất biểu đồ tĩnh ra thư mục assets/
    if not os.path.exists("assets"):
        os.makedirs("assets")
    save_all_charts(output, df, out_dir="assets")
    
    best_scenario = output["best_result"]["scenario"]

    print(f"\n🏆 Mô hình tốt nhất tìm được: {best_scenario['name']}")
    print(f"🌲 Đang huấn luyện lại trên toàn bộ dữ liệu với tham số: {best_scenario['params']}...")
    
    # Dùng _build_model để tự động wrap Pipeline nếu cần (Ridge, SVR, KNN)
    model = _build_model(best_scenario)
    model.fit(X, y)
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Lưu model ra file
    model_path = "models/best_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"✅ Đã lưu mô hình thành công tại: {model_path}")

if __name__ == "__main__":
    train_and_save_model()