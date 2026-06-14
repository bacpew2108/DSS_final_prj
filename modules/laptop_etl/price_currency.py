# modules/processing/price_currency.py
"""Chuyển đổi giá tiền từ EGP sang VND."""

import re
import pandas as pd


def get_egp_to_vnd_rate():
    """Return fixed EGP->VND exchange rate (no external calls)."""
    return 506.14


def parse_price_to_vnd(s, rate):
    """Parse a price string assumed in EGP and convert to VND using ``rate``.

    Returns integer VND or None.
    """
    if s is None:
        return None
    txt = str(s)
    # remove non-digit except . and , and spaces and - (range)
    # replace commas used as thousand separators
    txt = txt.replace('\u00A0', ' ')
    # find first number or range
    nums = re.findall(r"(\d+[\d,\.\s]*)", txt)
    if not nums:
        return None
    # take the first numeric token, strip separators
    token = nums[0]
    token = token.replace(' ', '')
    # remove commas
    token = token.replace(',', '')
    try:
        egp = float(token)
    except Exception:
        # maybe token like '12.345.00' -> remove dots
        token2 = token.replace('.', '')
        try:
            egp = float(token2)
        except Exception:
            return None
    vnd = int(round(egp * rate))
    return vnd


def compute_price_vnd(df):
    """Chuyển đổi cột ``price`` từ EGP sang VND và làm tròn đến 100,000 VND."""
    rate = get_egp_to_vnd_rate()
    df['price'] = df.get('price', pd.Series()).apply(
        lambda x: parse_price_to_vnd(x, rate) if pd.notna(x) else None
    )
    # fill missing prices with 0
    df['price'] = df['price'].fillna(0).astype(int)
    # round to nearest 100,000 VND
    try:
        df['price'] = (df['price'] / 100000).round().astype(int) * 100000
    except Exception:
        pass
    return df
