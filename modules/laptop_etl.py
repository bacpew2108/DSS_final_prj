# modules/laptop_etl.py
"""Backward-compatible shim — tất cả logic đã được chuyển vào package
``modules/laptop_etl/``. File này chỉ re-export để các import cũ tiếp tục hoạt động.

Cấu trúc package:
  laptop_etl/
  ├── __init__.py        — re-export toàn bộ public API
  ├── utils.py           — normalize_col_name, clean_text_value
  ├── ram_storage.py     — parse/compute RAM & storage
  ├── benchmark.py       — fetch/match CPU & GPU benchmark (NanoReview)
  ├── hardware.py        — weight, colors, brand, display, GPU memory
  ├── price_currency.py  — EGP → VND conversion
  ├── normalization.py   — warranty & OS normalization
  └── pipeline.py        — check_duplicates, fill_descriptive_fields, clean_laptops_csv
"""

from pathlib import Path
import sys

# ensure project root is on sys.path so `modules` package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export everything from the new sub-package
from modules.laptop_etl import (  # noqa: F401, E402
    normalize_col_name,
    clean_text_value,
    parse_ram_gb,
    compute_ram_capacity,
    parse_storage_gb,
    compute_storage_capacity,
    fetch_cpu_bench,
    match_cpu_scores,
    fetch_gpu_bench,
    match_gpu_scores,
    _CPU_SCORE_FALLBACK,
    _IGPU_KEYWORDS,
    parse_weight_kg,
    compute_weight,
    compute_fill_colors,
    infer_brand_from_product_name,
    extract_display_resolution,
    extract_display_refresh_rate,
    normalize_display_refresh_rate,
    extract_gpu_memory,
    infer_video_graphics,
    get_egp_to_vnd_rate,
    parse_price_to_vnd,
    compute_price_vnd,
    normalize_warranty,
    compute_normalize_warranty,
    normalize_operating_system,
    compute_normalize_os,
    check_duplicates,
    fill_descriptive_fields,
    clean_laptops_csv,
)

if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    input_path = base / 'data' / 'laptops_dataset_raw.csv'
    output_path = base / 'data' / 'laptops_dataset_cleaned.csv'
    drop_columns = [
        'Processor Generation', 'Processor Details', 'Product number', 'Power supply type',
        'Laptop Color', 'Finger Print', 'SUPPORT SSD M2', 'Pointing device', 'Optical drive', 'Series'
    ]
    clean_laptops_csv(input_path, output_path, drop_columns)
