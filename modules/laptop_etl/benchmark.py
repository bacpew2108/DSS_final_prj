# modules/processing/benchmark.py
"""Fetch và match điểm benchmark CPU/GPU từ NanoReview."""

import re
import difflib
import requests
import pandas as pd


# Điểm fallback cho các CPU cấp thấp không có trên NanoReview
_CPU_SCORE_FALLBACK = {
    'celeron n4020': 14.0,
    'celeron n4500': 17.0,
    'celeron n5100': 20.0,
    'celeron n6000': 21.0,
    'pentium silver n6000': 21.0,
    'amd 3020e': 13.0,
    'amd athlon silver 3050u': 16.0,
    'amd athlon silver 3050e': 15.0,
}

# Từ khóa nhận diện GPU tích hợp (không có benchmark riêng)
_IGPU_KEYWORDS = [
    'intel graphics', 'intel uhd', 'intel iris xe', 'intel iris',
    'intel arc graphics', 'amd radeon graphics', 'amd radeon™ graphics',
    'qualcomm adreno', 'integrated graphics', 'intel® graphics',
]


def fetch_cpu_bench():
    """Tải bảng điểm CPU laptop từ NanoReview và trả về DataFrame chuẩn hóa."""
    url = 'https://nanoreview.net/en/cpu-list/laptop-chips-rating'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    df = tables[0][['CPU', 'Rating']].dropna()
    df = df.rename(columns={'CPU': 'CPU_Name_Chuan', 'Rating': 'Diem_NanoReview'})
    df['CPU_Name_Chuan'] = df['CPU_Name_Chuan'].astype(str).str.strip()
    # extract numeric part of rating (e.g., '83 A' -> 83)
    df['Diem_NanoReview'] = pd.to_numeric(
        df['Diem_NanoReview'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0],
        errors='coerce'
    )
    df = df.dropna(subset=['CPU_Name_Chuan', 'Diem_NanoReview'])
    return df


def match_cpu_scores(df, bench_df):
    """Gán điểm CPU cho từng dòng trong ``df`` dựa trên ``bench_df`` từ NanoReview."""
    names = bench_df['CPU_Name_Chuan'].astype(str).tolist()
    # clean benchmark names similarly to input keys for better matching
    names_lower = [re.sub(r'[^a-z0-9 ]+', ' ', str(n).lower()) for n in names]

    def clean_cpu_text(t: str) -> str:
        if pd.isna(t):
            return ''
        s = str(t).lower()
        for token in ['intel', 'amd', 'processor', 'with', 'radeon', 'graphics', 'ghz']:
            s = s.replace(token, '')
        s = re.sub(r'[^a-z0-9 ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def find_match(raw):
        if pd.isna(raw) or not str(raw).strip():
            return None
        raw_lower = str(raw).lower()
        # Fallback cho CPU cấp thấp không có trong NanoReview
        for known_cpu, score in _CPU_SCORE_FALLBACK.items():
            if known_cpu in raw_lower:
                return score
        key = clean_cpu_text(raw)
        matches = difflib.get_close_matches(key, names_lower, n=1, cutoff=0.35)
        if matches:
            idx = names_lower.index(matches[0])
            try:
                return float(bench_df.iloc[idx]['Diem_NanoReview'])
            except Exception:
                return None
        return None

    df['cpu_point'] = df.get('processor', pd.Series()).apply(find_match)
    return df


def fetch_gpu_bench():
    """Tải bảng điểm GPU laptop từ NanoReview và trả về DataFrame chuẩn hóa."""
    url = 'https://nanoreview.net/en/gpu-list/laptop-graphics-rating'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    if not tables:
        raise ValueError('Không tìm thấy bảng dữ liệu GPU trên NanoReview.')
    raw = tables[0]
    # Prefer explicit 'GPU' and 'Rating' columns when present
    if 'GPU' in raw.columns and 'Rating' in raw.columns:
        df = raw[['GPU', 'Rating']].dropna().copy()
        df = df.rename(columns={'GPU': 'GPU_Name_Chuan', 'Rating': 'Diem_NanoReview'})
    else:
        # fallback to first and last columns
        name_col = raw.columns[0]
        score_col = raw.columns[-1]
        df = raw[[name_col, score_col]].dropna().copy()
        df = df.rename(columns={name_col: 'GPU_Name_Chuan', score_col: 'Diem_NanoReview'})

    df['GPU_Name_Chuan'] = df['GPU_Name_Chuan'].astype(str).str.strip()
    df['Diem_NanoReview'] = pd.to_numeric(
        df['Diem_NanoReview'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0],
        errors='coerce'
    )
    df = df.dropna(subset=['GPU_Name_Chuan', 'Diem_NanoReview'])
    return df


def match_gpu_scores(df, bench_df):
    """Gán điểm GPU cho từng dòng trong ``df`` dựa trên ``bench_df`` từ NanoReview."""
    names = bench_df['GPU_Name_Chuan'].astype(str).tolist()
    # prepare normalized bench names and token sets for matching
    names_lower = [re.sub(r'[^a-z0-9 ]+', ' ', str(n).lower()).strip() for n in names]
    bench_tokens = [set(n.split()) for n in names_lower]

    def clean_gpu_text(t: str) -> str:
        if pd.isna(t):
            return ''
        s = str(t).lower()
        # remove common noise: trademarks, memory sizes, bus types, punctuation
        s = re.sub(r'\(.*?\)', ' ', s)
        s = re.sub(r"gddr\d+", ' ', s)
        s = re.sub(r"\d+\s?gb", ' ', s)
        s = re.sub(r'®|™', ' ', s)
        for tok in ['nvidia', 'geforce', 'mobile', 'amd', 'radeon', 'intel', 'graphics', 'gpu', 'series']:
            s = s.replace(tok, ' ')
        s = re.sub(r'[^a-z0-9 ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def token_overlap_score(a_tokens: set, b_tokens: set) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        inter = a_tokens & b_tokens
        union = a_tokens | b_tokens
        return len(inter) / len(union)

    def find_match(raw):
        if pd.isna(raw) or not str(raw).strip():
            return None
        raw_lower = str(raw).strip().lower()
        # GPU tích hợp → không có benchmark rời, trả về NaN
        if any(kw in raw_lower for kw in _IGPU_KEYWORDS):
            return None
        key = clean_gpu_text(raw)
        if not key:
            return None

        # 1) exact substring match (preferred)
        for i, bench in enumerate(names_lower):
            if key in bench or bench in key:
                try:
                    return float(bench_df.iloc[i]['Diem_NanoReview'])
                except Exception:
                    return None

        # 2) token-overlap scoring
        key_tokens = set(key.split())
        best_idx = None
        best_score = 0.0
        for i, b_tokens in enumerate(bench_tokens):
            score = token_overlap_score(key_tokens, b_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score >= 0.45 and best_idx is not None:
            try:
                return float(bench_df.iloc[best_idx]['Diem_NanoReview'])
            except Exception:
                return None

        # 3) fallback to fuzzy matching
        matches = difflib.get_close_matches(key, names_lower, n=1, cutoff=0.30)
        if matches:
            idx = names_lower.index(matches[0])
            try:
                return float(bench_df.iloc[idx]['Diem_NanoReview'])
            except Exception:
                return None
        return None

    df['gpu_point'] = df.get('video_graphics', pd.Series()).apply(find_match)
    return df
