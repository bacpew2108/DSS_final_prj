import os
from pathlib import Path
from modules.laptop_etl import clean_laptops_csv

def main():
    base = Path(__file__).resolve().parent
    input_path = base / 'data' / 'laptops_dataset_raw.csv'
    output_path = base / 'data' / 'laptops_dataset_cleaned.csv'
    
    drop_columns = [
        'Processor Generation', 'Processor Details', 'Product number', 'Power supply type',
        'Laptop Color', 'Finger Print', 'SUPPORT SSD M2', 'Pointing device', 'Optical drive', 'Series'
    ]
    
    print(f"Bắt đầu quá trình ETL...")
    print(f"Đọc dữ liệu từ: {input_path}")
    clean_laptops_csv(input_path, output_path, drop_columns)
    print(f"Hoàn thành! Đã lưu dữ liệu sạch tại: {output_path}")

if __name__ == '__main__':
    main()
