# modules/processing/normalization.py
"""Chuẩn hóa các trường dạng text: bảo hành và hệ điều hành."""

import re
import pandas as pd


def normalize_warranty(s):
    """Chuẩn hóa cột warranty về dạng '<N> Year' / '<N> Years'.

    Đầu vào điển hình: '1 Year', '1 YEAR', '2 Years', '1year',
                         '1-year limited hardware warranty', v.v.
    Đầu ra: '1 Year', '2 Years', '3 Years', hoặc 'Unknown'.
    """
    if pd.isna(s) or not str(s).strip():
        return 'Unknown'
    txt = str(s).lower()
    m = re.search(r'(\d+)', txt)
    if not m:
        return 'Unknown'
    n = int(m.group(1))
    return f"{n} Year" if n == 1 else f"{n} Years"


def compute_normalize_warranty(df):
    """Áp dụng normalize_warranty cho toàn bộ cột ``warranty``."""
    if 'warranty' in df.columns:
        df['warranty'] = df['warranty'].apply(normalize_warranty)
    return df


def normalize_operating_system(s):
    """Chuẩn hóa cột operating_system về các category chuẩn:
    'DOS', 'FreeDOS', 'Windows 10 Home', 'Windows 10 Pro',
    'Windows 11 Home', 'Windows 11 Pro', hoặc 'Unknown'.
    """
    if pd.isna(s) or not str(s).strip():
        return 'Unknown'
    txt = str(s).strip().lower()
    # Loại bỏ ký tự trademark và chuẩn hóa khoảng trắng
    txt = re.sub(r'[\u00ae\u2122\u00a9]', '', txt)   # ®, ™, ©
    txt = re.sub(r'\s+', ' ', txt).strip()

    # FreeDOS
    if re.search(r'free\s*dos', txt):
        return 'FreeDOS'
    # DOS (bao gồm 'dos', 'Dos', 'DOS')
    if re.search(r'\bdos\b', txt):
        return 'DOS'

    # Windows – xác định phiên bản
    m = re.search(r'win(?:dows)?\s*\.?\s*(10|11)', txt)
    if m:
        ver = m.group(1)
        if re.search(r'\bpro\b', txt):
            return f'Windows {ver} Pro'
        return f'Windows {ver} Home'

    return 'Unknown'


def compute_normalize_os(df):
    """Áp dụng normalize_operating_system cho toàn bộ cột ``operating_system``."""
    if 'operating_system' in df.columns:
        df['operating_system'] = df['operating_system'].apply(normalize_operating_system)
    return df
