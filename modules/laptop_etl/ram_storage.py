# modules/processing/ram_storage.py
"""Parse và tính dung lượng RAM và ổ cứng từ chuỗi mô tả."""

import re
import pandas as pd


def parse_ram_gb(s):
    """Parse a RAM description string and return total capacity in GB (int) or None.

    Hỗ trợ các dạng:
    - 'N x M GB' / 'NxMGB'   → N * M  (e.g. '2x8GB' → 16)
    - 'M GB * N'              → M * N  (e.g. '48GB*2' → 96)
    - 'X GB'                  → X      (lấy số GB đầu tiên — tổng thường ghi trước)
    - Trùng lặp cùng pattern  → deduplicate trước khi tính
    """
    if s is None:
        return None
    txt = str(s).lower().replace(',', ' ')

    # Bước 1: N×M GB (e.g. '2 x 8 GB', '2x8GB', '1x16GB')
    m1 = re.findall(r"(\d+)\s*[x×]\s*(\d+)\s*gb", txt)
    if m1:
        unique = list(dict.fromkeys(m1))   # bỏ trùng lặp, giữ thứ tự
        try:
            return sum(int(mult) * int(size) for mult, size in unique)
        except Exception:
            pass

    # Bước 2: M GB * N (size trước, multiplier sau — e.g. '48GB*2', '8GB*2')
    m2 = re.findall(r"(\d+)\s*gb\s*\*\s*(\d+)", txt)
    if m2:
        unique = list(dict.fromkeys(m2))
        try:
            return sum(int(size) * int(mult) for size, mult in unique)
        except Exception:
            pass

    # Bước 3: lấy số GB đầu tiên (tổng tổng thường được ghi trước)
    nums = re.findall(r"(\d+)\s*gb", txt)
    if nums:
        try:
            return int(nums[0])
        except Exception:
            pass

    # Fallback: số hợp lý đầu tiên trong khoảng 2–1024
    for n in re.findall(r"(\d+)", txt):
        v = int(n)
        if 2 <= v <= 1024:
            return v
    return None


def compute_ram_capacity(df):
    """Thêm cột ``ram_capacity`` (GB) vào DataFrame."""
    df['ram_capacity'] = df.get('ram', pd.Series()).apply(
        lambda x: parse_ram_gb(x) if pd.notna(x) else None
    )
    return df


def parse_storage_gb(s):
    """Parse storage string and return total capacity in GB (int or None).

    Rules:
    - Accept patterns like '512GB', '1 TB', '2 x 512GB', '256GB SSD + 1TB HDD'.
    - Sum multiple drives where present.
    - Convert TB to GB using 1024 multiplier.
    - Return None if no reasonable capacity found.
    """
    if s is None:
        return None
    txt = str(s).lower().replace(',', ' ')
    total_gb = 0
    found = False

    # handle multiplier patterns like '2 x 512gb' or '2x512gb'
    for mult, size, unit in re.findall(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(tb|gb)", txt):
        try:
            m = int(mult)
            sz = float(size)
            if unit == 'tb':
                sz_gb = sz * 1024
            else:
                sz_gb = sz
            total_gb += int(m * sz_gb)
            found = True
        except Exception:
            pass

    # handle standalone sizes like '1tb', '512 gb'
    for size, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(tb|gb)", txt):
        try:
            sz = float(size)
            if unit == 'tb':
                sz_gb = sz * 1024
            else:
                sz_gb = sz
            total_gb += int(sz_gb)
            found = True
        except Exception:
            pass

    if found:
        return int(total_gb)

    # fallback: any reasonable standalone number (choose largest reasonable candidate)
    nums = [int(n) for n in re.findall(r"(\d+)", txt)]
    candidates = [n for n in nums if 8 <= n <= 8192]
    if candidates:
        return max(candidates)
    return None


def compute_storage_capacity(df):
    """Thêm cột ``storage`` (GB) vào DataFrame từ cột ``hard_drive``."""
    df['storage'] = df.get('hard_drive', pd.Series()).apply(
        lambda x: parse_storage_gb(x) if pd.notna(x) else None
    )
    return df
