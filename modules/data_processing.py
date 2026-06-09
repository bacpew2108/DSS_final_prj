import re
import unicodedata
from pathlib import Path
import sys
import requests
import difflib
import pandas as pd

# ensure project root is on sys.path so `modules` package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import `parse_ram_gb` from modules; if package import fails, load module by path.
try:
    from modules.data_processing import parse_ram_gb
except Exception:
    # If the project's modules/data_processing.py is missing, provide a simple
    # fallback implementation of `parse_ram_gb` so the cleaner can still run.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('data_processing', str(ROOT / 'modules' / 'data_processing.py'))
        dp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dp)
        parse_ram_gb = getattr(dp, 'parse_ram_gb')
    except Exception:
        def parse_ram_gb(s):
            if s is None:
                return None
            txt = str(s).lower().replace(',', ' ')
            # patterns like '2 x 8 GB' or '2x8GB'
            m = re.findall(r"(\d+)\s*[x×]\s*(\d+)\s*gb", txt)
            if m:
                try:
                    return sum(int(mult) * int(size) for mult, size in m)
                except Exception:
                    pass
            # all occurrences of numbers followed by 'gb'
            nums = re.findall(r"(\d+)\s*gb", txt)
            if nums:
                try:
                    return sum(int(n) for n in nums)
                except Exception:
                    pass
            # fallback: any standalone reasonable number (2-1024)
            nums2 = re.findall(r"(\d+)", txt)
            for n in nums2:
                v = int(n)
                if 2 <= v <= 1024:
                    return v
            return None


def normalize_col_name(name: str) -> str:
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


def fetch_cpu_bench():
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


def compute_ram_capacity(df):
    df['ram_capacity'] = df.get('ram', pd.Series()).apply(lambda x: parse_ram_gb(x) if pd.notna(x) else None)
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
    df['storage'] = df.get('hard_drive', pd.Series()).apply(lambda x: parse_storage_gb(x) if pd.notna(x) else None)
    return df


def parse_weight_kg(s):
    """Parse weight string and return weight in kilograms as float, or None.

    Heuristics:
    - Look for patterns like '2.3 kg', '2.3kg', '2300 g'.
    - Convert grams to kg when detected.
    - If multiple numbers are present, prefer values in reasonable kg range (0.3-10).
    """
    if s is None:
        return None
    txt = str(s).lower().replace(',', '.')

    # check for explicit kg
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg\b", txt)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # check for grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", txt)
    if m:
        try:
            return float(m.group(1)) / 1000.0
        except Exception:
            pass

    # fallback: find numeric tokens and pick the first reasonable kg value
    nums = re.findall(r"(\d+(?:\.\d+)?)", txt)
    candidates = []
    for n in nums:
        try:
            v = float(n)
        except Exception:
            continue
        if 0.3 <= v <= 10:
            candidates.append(v)
        elif 300 <= v <= 10000:
            # grams
            candidates.append(v / 1000.0)
    if candidates:
        return float(candidates[0])
    return None


def compute_weight(df):
    # parse weight into kg floats
    df['weight'] = df.get('weight', pd.Series()).apply(lambda x: parse_weight_kg(x) if pd.notna(x) else None)
    # compute mean of parsed weights
    # Temporarily disable filling missing weights with the mean as requested.
    # try:
    #     mean_val = float(df['weight'].dropna().astype(float).mean())
    # except Exception:
    #     mean_val = None
    # if mean_val is None or pd.isna(mean_val):
    #     # if no valid weights, leave as-is
    #     return df
    # # fill missing or unparsable with mean
    # df['weight'] = df['weight'].fillna(mean_val)
    # round to 1 decimal place for existing values

    # drop rows where weight is blank/unparsable
    df = df.dropna(subset=['weight']).copy()

    try:
        df['weight'] = df['weight'].astype(float).round(1)
    except Exception:
        pass
    return df


def compute_fill_colors(df):
    # fill missing or empty colors with 'Black'
    def fix_color(x):
        if pd.isna(x):
            return 'Black'
        s = str(x).strip()
        if not s:
            return 'Black'
        low = s.lower()
        if low in {'nan', 'n/a', 'none', '-', '—'}:
            return 'Black'
        return s

    df['colors'] = df.get('colors', pd.Series()).apply(fix_color)
    return df


def get_egp_to_vnd_rate():
    """Return fixed EGP->VND exchange rate (no external calls)."""
    return 506.14


def parse_price_to_vnd(s, rate):
    """Parse a price string assumed in EGP and convert to VND using `rate`.

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
    rate = get_egp_to_vnd_rate()
    df['price'] = df.get('price', pd.Series()).apply(lambda x: parse_price_to_vnd(x, rate) if pd.notna(x) else None)
    # fill missing prices with 0
    df['price'] = df['price'].fillna(0).astype(int)
    # round to nearest 100,000 VND
    try:
        df['price'] = (df['price'] / 100000).round().astype(int) * 100000
    except Exception:
        pass
    return df


def clean_laptops_csv(input_path: Path, output_path: Path, drop_columns: list):
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    orig_columns = list(df.columns)
    norm_map = {c: normalize_col_name(c) for c in orig_columns}
    df = df.rename(columns=norm_map)

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
    # ensure numeric and default 0 for unmatched
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

    # Save cleaned and enriched CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Wrote cleaned CSV to {output_path}')


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    input_path = base / 'data' / 'laptops_dataset_raw.csv'
    output_path = base / 'data' / 'laptops_dataset_cleaned.csv'
    drop_columns = [
        'Processor Generation', 'Processor Details', 'Product number', 'Power supply type',
        'Laptop Color', 'Finger Print', 'SUPPORT SSD M2', 'Pointing device', 'Optical drive', 'Series'
    ]
    clean_laptops_csv(input_path, output_path, drop_columns)
