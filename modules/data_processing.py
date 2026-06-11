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
    # Parse weight into kg floats first; keep blanks as missing for imputation.
    df['weight'] = df.get('weight', pd.Series()).apply(
        lambda x: parse_weight_kg(x) if pd.notna(x) and str(x).strip() else None
    )

    if 'product_name' in df.columns:
        product_median = df.groupby('product_name')['weight'].transform(
            lambda s: float(s.dropna().median()) if not s.dropna().empty else None
        )
        df['weight'] = df['weight'].fillna(product_median)

    if 'brand' in df.columns:
        brand_median = df.groupby('brand')['weight'].transform(
            lambda s: float(s.dropna().median()) if not s.dropna().empty else None
        )
        df['weight'] = df['weight'].fillna(brand_median)

    try:
        overall_median = float(df['weight'].dropna().astype(float).median())
    except Exception:
        overall_median = None

    if overall_median is not None and not pd.isna(overall_median):
        df['weight'] = df['weight'].fillna(overall_median)

    # If anything is still missing, leave it explicitly marked as Unknown.
    df['weight'] = df['weight'].apply(lambda x: round(float(x), 1) if pd.notna(x) else 'Unknown')
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


def normalize_warranty(s):
    """Chuẩn hóa cỗt warranty về dạng '<N> Year' / '<N> Years'.

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
    if 'warranty' in df.columns:
        df['warranty'] = df['warranty'].apply(normalize_warranty)
    return df


def clean_text_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {'nan', 'none', 'n/a', 'na', '-', '—'}:
        return None
    return text


def infer_brand_from_product_name(product_name):
    text = clean_text_value(product_name)
    if text is None:
        return None

    low = text.lower()
    brand_aliases = [
        # ASUS: tên thương hiệu, dòng sản phẩm, và các model code phổ biến
        # (FX5/FX6 = TUF F-series; G5/G7/G8 ROG Strix; GA4/GA5/GA6 = Zephyrus G14/G15/G16;
        #  GU6 = Zephyrus M16; GV3 = ROG Flow; GX6 = Zephyrus Duo; GZ3 = Zephyrus G13;
        #  D35 = ProArt Studiobook)
        ('ASUS', ['asus', 'rog', 'strix', 'tuf', 'zenbook', 'vivobook', 'expertbook',
                  'proart', 'zephyrus',
                  'fx5', 'fx6', 'gv3', 'ga4', 'ga5', 'ga6', 'gu6', 'gz3', 'gx6',
                  'g713', 'g733', 'g513', 'g533', 'g814', 'g834', 'd3500']),
        ('HP', ['hp', 'victus', 'omen']),
        ('Lenovo', ['lenovo', 'legion', 'loq', 'thinkpad', 'ideapad', 'thinkbook']),
        ('Dell', ['dell', 'alienware']),
        ('MSI', ['msi', 'katana', 'raider', 'vector', 'crosshair', 'cyborg', 'prestige',
                  'summit', 'stealth', 'pulse', 'alpha', 'bravo', 'modern', 'creator',
                  'venture', 'sword']),
        ('Acer', ['acer', 'aspire', 'nitro', 'predator', 'swift']),
        ('Microsoft', ['microsoft', 'surface']),
        ('Apple', ['apple', 'macbook']),
        ('Samsung', ['samsung', 'galaxy book']),
        ('Gigabyte', ['gigabyte', 'aero']),
        ('Razer', ['razer', 'blade']),
        ('Huawei', ['huawei', 'matebook']),
    ]

    for brand, aliases in brand_aliases:
        if any(alias in low for alias in aliases):
            return brand
    return None


def extract_display_resolution(display_text):
    text = clean_text_value(display_text)
    if text is None:
        return None
    match = re.search(r'(\d{3,4})\s*[x×*]\s*(\d{3,4})', text.lower())
    if not match:
        return None
    return f"{match.group(1)} x {match.group(2)}"


def extract_display_refresh_rate(display_text):
    text = clean_text_value(display_text)
    if text is None:
        return None
    low = text.lower()
    match = re.search(r'(\d{2,3})\s*hz', low)
    if match:
        return f"{match.group(1)} Hz"
    # Fallback mặc định: mọi màn hình không xác định được refresh rate → 60 Hz
    return '60 Hz'


def extract_gpu_memory(video_graphics_text):
    text = clean_text_value(video_graphics_text)
    if text is None:
        return None

    low = text.lower()
    if any(token in low for token in ['integrated', 'intel arc graphics', 'qualcomm adreno', 'intel uhd', 'intel iris', 'intel graphics']):
        return 'Integrated'

    match = re.search(r'(\d+(?:\.\d+)?)\s*gb\s*(gddr\d+)?', low)
    if match:
        size = match.group(1)
        gddr = match.group(2)
        return f"{size}GB {gddr.upper()}" if gddr else f"{size}GB"

    match = re.search(r'(gddr\d+)\s*(\d+(?:\.\d+)?)\s*gb', low)
    if match:
        return f"{match.group(2)}GB {match.group(1).upper()}"

    return None


def infer_video_graphics(row):
    video_graphics = clean_text_value(row.get('video_graphics'))
    if video_graphics is not None:
        return video_graphics

    processor_text = clean_text_value(row.get('processor'))
    product_name = clean_text_value(row.get('product_name'))
    processor_low = processor_text.lower() if processor_text else ''
    product_low = product_name.lower() if product_name else ''

    if 'snapdragon' in processor_low or 'surface pro 11' in product_low:
        return 'Qualcomm Adreno GPU'
    if 'intel core ultra' in processor_low or 'intel arc' in processor_low:
        return 'Intel Arc Graphics'
    if 'intel' in processor_low:
        return 'Intel Graphics'
    if 'amd ryzen' in processor_low or 'ryzen ai' in processor_low:
        return 'AMD Radeon Graphics'
    return 'Unknown'


def fill_descriptive_fields(df):
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
        df['display_refresh_rate'] = df['display_refresh_rate'].apply(clean_text_value)
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


def clean_laptops_csv(input_path: Path, output_path: Path, drop_columns: list):
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

    try:
        df = compute_normalize_warranty(df)
    except Exception as e:
        print('Warranty normalization failed:', e)

    try:
        df = fill_descriptive_fields(df)
    except Exception as e:
        print('Descriptive field fill failed:', e)

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
