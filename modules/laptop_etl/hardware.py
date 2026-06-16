# modules/laptop_etl/hardware.py
"""Parse và chuẩn hóa các thuộc tính phần cứng: trọng lượng, màu sắc,
GPU memory, màn hình, thương hiệu."""

import re
import pandas as pd

from .utils import clean_text_value


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
    """Tính và điền cột ``weight`` (kg) vào DataFrame.

    Ưu tiên: parse từ chuỗi → median theo sản phẩm → median theo brand → overall median.
    """
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
    """Điền màu sắc mặc định ('Black') cho các giá trị thiếu hoặc không hợp lệ."""
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


def infer_brand_from_product_name(product_name):
    """Suy luận thương hiệu laptop từ tên sản phẩm dựa trên danh sách aliases."""
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
                  'fx5', 'fx6', 'fx7', 'gv3', 'ga4', 'ga5', 'ga6', 'gu6', 'gz3', 'gx6',
                  'g713', 'g733', 'g513', 'g533', 'g814', 'g834', 'd3500',
                  # TUF A-series model codes (e.g. FA506, FA608, FA401)
                  'fa506', 'fa606', 'fa507', 'fa607', 'fa608', 'fa401', 'fa417',
                  # Vivobook / Zenbook model codes
                  'd515', 'x515', 'x513', 'k513', 'k3605',
                  'ux3', 'ux4', 'ux5', 'ux8', 'um56']),
        ('HP', ['hp', 'victus', 'omen', '15s-', '14s-', 'envy', 'pavilion', 'elitebook',
                'spectre', 'probook', 'omnibook']),
        ('Lenovo', ['lenovo', 'legion', 'loq', 'thinkpad', 'ideapad', 'thinkbook', 'yoga']),
        ('Dell', ['dell', 'alienware', 'vostro', 'inspiron', 'latitude', 'xps', 'precision']),
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


# Bảng chuyển đổi chữ số Arabic-Indic → ASCII
_ARABIC_DIGIT_MAP = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')


def normalize_display_resolution(s):
    """Chuẩn hóa giá trị độ phân giải màn hình về dạng 'WxH' (e.g. '1920x1080').

    Xử lý các trường hợp:
    - Nhiều ký tự phân cách: 'x', 'X', '×', '*'
    - Khoảng trắng xung quanh dấu phân cách: '1920 x 1080' → '1920x1080'
    - Prefix tên marketing: 'FHD (1920 x 1080)', 'WQHD (2560 x 1440)',
      '2.5K (2560 x 1600, WQXGA)', '3K (2880 x 1800)'
    - Nội dung thừa: '2880 X 1920 (267 PPI)', 'FHD 1920x1080,'
    - Dấu ngoặc/dấu phẩy thừa: '(1920x1080)', '1920 x 1080)'
    - Ký tự số Arabic-Indic: '۱۹۲۰ x ۱۰۸۰'
    - Trả về None nếu không tìm thấy pattern hợp lệ.
    """
    text = clean_text_value(s)
    if text is None:
        return None
    # Chuyển chữ số Arabic-Indic sang ASCII
    text = text.translate(_ARABIC_DIGIT_MAP)
    # Tìm pattern WxH (3–4 chữ số × 3–4 chữ số)
    match = re.search(r'(\d{3,4})\s*[xX×*]\s*(\d{3,4})', text)
    if not match:
        return None
    return f"{match.group(1)}x{match.group(2)}"


def extract_display_resolution(display_text):
    """Trích xuất độ phân giải màn hình từ chuỗi mô tả màn hình đầy đủ.

    Sử dụng normalize_display_resolution để đảm bảo output nhất quán.
    """
    return normalize_display_resolution(display_text)


def extract_display_refresh_rate(display_text):
    """Trích xuất tần số quét (e.g. '144 Hz') từ chuỗi mô tả màn hình.
    Mặc định trả về '60 Hz' nếu không tìm thấy.
    """
    text = clean_text_value(display_text)
    if text is None:
        return None
    low = text.lower()
    match = re.search(r'(\d{2,3})\s*hz', low)
    if match:
        return f"{match.group(1)} Hz"
    # Fallback mặc định: mọi màn hình không xác định được refresh rate → 60 Hz
    return '60 Hz'


def normalize_display_refresh_rate(s):
    """Chuẩn hóa giá trị refresh rate đã có về dạng 'N Hz'.

    Ví dụ: '165 HZ' → '165 Hz', '360Hz' → '360 Hz', '240 HZ' → '240 Hz'.
    Trả về None nếu không tìm được số (để fillna xử lý tiếp).
    """
    text = clean_text_value(s)
    if text is None:
        return None
    m = re.search(r'(\d{2,3})\s*hz', text.lower())
    if m:
        return f"{m.group(1)} Hz"
    return None


def extract_gpu_memory(video_graphics_text):
    """Trích xuất dung lượng VRAM (e.g. '8GB GDDR6') từ chuỗi mô tả GPU."""
    text = clean_text_value(video_graphics_text)
    if text is None:
        return None

    low = text.lower()
    if any(token in low for token in [
        'integrated', 'intel arc graphics', 'qualcomm adreno',
        'intel uhd', 'intel iris', 'intel graphics'
    ]):
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
    """Suy luận card đồ họa từ thông tin processor/product_name nếu cột video_graphics trống."""
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
