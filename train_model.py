import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from modules.random_forest_engine import FEATURE_COLS, TARGET_COL, HYPERPARAMETER_SCENARIOS

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

    # Chọn cấu hình tốt nhất (Combo Tối ưu 2 - Index 7 trong list của bạn)
    best_params = HYPERPARAMETER_SCENARIOS[7]["params"]
    
    print(f"🌲 Đang huấn luyện Random Forest với tham số: {best_params}...")
    model = RandomForestRegressor(**best_params)
    model.fit(X, y)
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Lưu model ra file
    model_path = "models/rf_best_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"✅ Đã lưu mô hình thành công tại: {model_path}")

if __name__ == "__main__":
    train_and_save_model()