# modules/laptop_etl/utils.py
"""Các hàm tiện ích chung dùng trong toàn bộ package processing."""

import re
import unicodedata


def normalize_col_name(name: str) -> str:
    """Chuẩn hóa tên cột: lowercase, bỏ dấu, thay ký tự đặc biệt bằng '_'."""
    if name is None:
        return ''
    s = str(name).strip().lower()
    # remove diacritics
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # replace any non-alphanumeric sequence with underscore
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s


def clean_text_value(value):
    """Làm sạch giá trị chuỗi: strip, trả về None nếu rỗng/placeholder."""
    import pandas as pd
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {'nan', 'none', 'n/a', 'na', '-', '—'}:
        return None
    return text
