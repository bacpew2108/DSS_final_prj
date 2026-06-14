# modules/processing/pipeline.py
"""Pipeline chính: kiểm tra duplicate, điền trường mô tả, và hàm clean toàn bộ CSV."""

from pathlib import Path
import pandas as pd

from .utils import normalize_col_name, clean_text_value
from .ram_storage import compute_ram_capacity, compute_storage_capacity
from .benchmark import fetch_cpu_bench, match_cpu_scores, fetch_gpu_bench, match_gpu_scores
from .hardware import (
    compute_weight,
    compute_fill_colors,
    infer_brand_from_product_name,
    extract_display_resolution,
    extract_display_refresh_rate,
    normalize_display_refresh_rate,
    extract_gpu_memory,
    infer_video_graphics,
)
from .price_currency import compute_price_vnd
from .normalization import compute_normalize_warranty, compute_normalize_os


def check_duplicates(df: pd.DataFrame, subset: list = None, verbose: bool = True) -> pd.DataFrame:
    """Kiểm tra và báo cáo các dòng trùng lặp trong DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần kiểm tra.
    subset : list, optional
        Danh sách cột dùng để xác định bản ghi trùng (mặc định: tất cả cột).
        Ví dụ: ['product_name', 'brand', 'processor'] để xác định duplicate
        dựa trên tên sản phẩm, thương hiệu và CPU.
    verbose : bool
        Nếu True, in báo cáo chi tiết ra console.

    Returns
    -------
    pd.DataFrame
        DataFrame đã loại bỏ các dòng trùng lặp (giữ lại occurrence đầu tiên).
    """
    total_rows = len(df)

    # Xóa duplicate hoàn toàn, giữ lại dòng đầu tiên
    df_clean = df.drop_duplicates(keep='first')
    removed = total_rows - len(df_clean)
    if removed > 0:
        print(f"[check_duplicates] Đã xóa {removed:,} dòng trùng lặp hoàn toàn. "
              f"Còn lại {len(df_clean):,} dòng.")
    else:
        print("[check_duplicates] Không tìm thấy dòng trùng lặp hoàn toàn.")

    return df_clean


def fill_descriptive_fields(df):
    """Điền các trường mô tả còn thiếu bằng cách suy luận hoặc dùng mode theo nhóm."""
    if 'product_name' not in df.columns:
        df['product_name'] = pd.NA

    if 'brand' in df.columns:
        df['brand'] = df['brand'].apply(clean_text_value)
        inferred_brand = df['product_name'].apply(infer_brand_from_product_name)
        df['brand'] = df['brand'].fillna(inferred_brand)
        df['brand'] = df['brand'].fillna('Unknown')

    if 'video_graphics' in df.columns:
        df['video_graphics'] = df.apply(infer_video_graphics, axis=1)

    if 'video_graphics' in df.columns and 'video_graphics_memory' in df.columns:
        df['video_graphics_memory'] = df['video_graphics_memory'].apply(clean_text_value)
        inferred_gpu_memory = df['video_graphics'].apply(extract_gpu_memory)
        df['video_graphics_memory'] = df['video_graphics_memory'].fillna(inferred_gpu_memory)
        df['video_graphics_memory'] = df['video_graphics_memory'].fillna('Unknown')

    if 'display' in df.columns and 'display_resolution' in df.columns:
        df['display_resolution'] = df['display_resolution'].apply(clean_text_value)
        inferred_resolution = df['display'].apply(extract_display_resolution)
        df['display_resolution'] = df['display_resolution'].fillna(inferred_resolution)
        df['display_resolution'] = df['display_resolution'].fillna('Unknown')

    if 'display' in df.columns and 'display_refresh_rate' in df.columns:
        # Chuẩn hóa format cho cả các giá trị đã có (vd: '165 HZ' → '165 Hz')
        df['display_refresh_rate'] = df['display_refresh_rate'].apply(normalize_display_refresh_rate)
        inferred_refresh = df['display'].apply(extract_display_refresh_rate)
        df['display_refresh_rate'] = df['display_refresh_rate'].fillna(inferred_refresh)
        df['display_refresh_rate'] = df['display_refresh_rate'].fillna('60 Hz')

    fill_from_group = ['operating_system', 'keyboard', 'battery', 'webcam', 'connections', 'dimensions', 'display']

    for column in fill_from_group:
        if column not in df.columns:
            continue
        df[column] = df[column].apply(clean_text_value)
        if df[column].isna().any():
            mode_by_product = df.groupby('product_name')[column].transform(
                lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else None
            )
            df[column] = df[column].fillna(mode_by_product)
        if df[column].isna().any() and 'brand' in df.columns:
            mode_by_brand = df.groupby('brand')[column].transform(
                lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else None
            )
            df[column] = df[column].fillna(mode_by_brand)
        df[column] = df[column].fillna('Unknown')

    return df


def clean_laptops_csv(input_path: Path, output_path: Path, drop_columns: list):
    """Đọc, làm sạch và xuất CSV laptop đã được chuẩn hóa.

    Quy trình:
    1. Đọc CSV và chuẩn hóa tên cột.
    2. Loại bỏ duplicate.
    3. Xóa các cột không cần thiết.
    4. Enrich với điểm CPU/GPU benchmark.
    5. Parse RAM, storage, weight, màu sắc, giá.
    6. Chuẩn hóa bảo hành, hệ điều hành.
    7. Điền các trường mô tả còn thiếu.
    8. Ghi output CSV.
    """
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    orig_columns = list(df.columns)
    norm_map = {c: normalize_col_name(c) for c in orig_columns}
    df = df.rename(columns=norm_map)

    # --- Kiểm tra duplicate ngay sau khi đọc dữ liệu ---
    key_cols = ['product_name', 'brand', 'processor', 'ram', 'hard_drive', 'price']
    df = check_duplicates(df, subset=key_cols, verbose=True)

    # Expect normalized columns `processor` and `video_graphics` to exist
    # (raw file contains these). If not found, create empty columns and warn.
    if 'processor' not in df.columns:
        print("Warning: 'processor' column not found after normalization; creating empty 'processor' column.")
        df['processor'] = pd.NA
    if 'video_graphics' not in df.columns:
        print("Warning: 'video_graphics' column not found after normalization; creating empty 'video_graphics' column.")
        df['video_graphics'] = pd.NA

    drop_norm = [normalize_col_name(c) for c in drop_columns]
    existing_to_drop = [c for c in drop_norm if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    # Enrich with CPU/GPU benchmarks and RAM capacity
    try:
        cpu_bench = fetch_cpu_bench()
        df = match_cpu_scores(df, cpu_bench)
    except Exception as e:
        print('CPU benchmark fetch/match failed:', e)
        df['cpu_point'] = 0
    # ensure numeric and default 0 for unmatched
    try:
        df['cpu_point'] = df.get('cpu_point', pd.Series()).fillna(0).astype(float)
    except Exception:
        df['cpu_point'] = df.get('cpu_point', pd.Series()).fillna(0)

    try:
        gpu_bench = fetch_gpu_bench()
        df = match_gpu_scores(df, gpu_bench)
    except Exception as e:
        print('GPU benchmark fetch/match failed:', e)
        df['gpu_point'] = 0
    # NaN = GPU tích hợp (iGPU) hoặc không match được → điền 0
    try:
        df['gpu_point'] = df.get('gpu_point', pd.Series()).fillna(0).astype(float)
    except Exception:
        df['gpu_point'] = df.get('gpu_point', pd.Series()).fillna(0)

    try:
        df = compute_ram_capacity(df)
    except Exception as e:
        print('RAM capacity parsing failed:', e)
        df['ram_capacity'] = None

    try:
        df = compute_storage_capacity(df)
    except Exception as e:
        print('Storage capacity parsing failed:', e)
        df['storage'] = None

    try:
        df = compute_weight(df)
    except Exception as e:
        print('Weight parsing failed:', e)
        df['weight'] = None

    try:
        df = compute_fill_colors(df)
    except Exception as e:
        print('Colors fill failed:', e)
        df['colors'] = df.get('colors', pd.Series()).fillna('Black')

    try:
        df = compute_price_vnd(df)
    except Exception as e:
        print('Price conversion failed:', e)
        df['price'] = df.get('price', pd.Series()).fillna(0)

    try:
        df = compute_normalize_warranty(df)
    except Exception as e:
        print('Warranty normalization failed:', e)

    try:
        df = compute_normalize_os(df)
    except Exception as e:
        print('OS normalization failed:', e)

    try:
        df = fill_descriptive_fields(df)
    except Exception as e:
        print('Descriptive field fill failed:', e)

    # Save cleaned and enriched CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Wrote cleaned CSV to {output_path}')
