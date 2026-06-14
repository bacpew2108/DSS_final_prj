# modules/processing/__init__.py
# Re-export every public symbol so existing imports keep working.

from .utils import normalize_col_name, clean_text_value
from .ram_storage import parse_ram_gb, compute_ram_capacity, parse_storage_gb, compute_storage_capacity
from .benchmark import (
    fetch_cpu_bench,
    match_cpu_scores,
    fetch_gpu_bench,
    match_gpu_scores,
    _CPU_SCORE_FALLBACK,
    _IGPU_KEYWORDS,
)
from .hardware import (
    parse_weight_kg,
    compute_weight,
    compute_fill_colors,
    infer_brand_from_product_name,
    extract_display_resolution,
    extract_display_refresh_rate,
    normalize_display_refresh_rate,
    extract_gpu_memory,
    infer_video_graphics,
)
from .price_currency import get_egp_to_vnd_rate, parse_price_to_vnd, compute_price_vnd
from .normalization import (
    normalize_warranty,
    compute_normalize_warranty,
    normalize_operating_system,
    compute_normalize_os,
)
from .pipeline import check_duplicates, fill_descriptive_fields, clean_laptops_csv

__all__ = [
    # utils
    'normalize_col_name', 'clean_text_value',
    # ram_storage
    'parse_ram_gb', 'compute_ram_capacity', 'parse_storage_gb', 'compute_storage_capacity',
    # benchmark
    'fetch_cpu_bench', 'match_cpu_scores', 'fetch_gpu_bench', 'match_gpu_scores',
    '_CPU_SCORE_FALLBACK', '_IGPU_KEYWORDS',
    # hardware
    'parse_weight_kg', 'compute_weight', 'compute_fill_colors',
    'infer_brand_from_product_name',
    'extract_display_resolution', 'extract_display_refresh_rate',
    'normalize_display_refresh_rate', 'extract_gpu_memory', 'infer_video_graphics',
    # price_currency
    'get_egp_to_vnd_rate', 'parse_price_to_vnd', 'compute_price_vnd',
    # normalization
    'normalize_warranty', 'compute_normalize_warranty',
    'normalize_operating_system', 'compute_normalize_os',
    # pipeline
    'check_duplicates', 'fill_descriptive_fields', 'clean_laptops_csv',
]
