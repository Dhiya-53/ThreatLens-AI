import os
import sys
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Reconfigure stdout to utf-8 if possible to prevent Windows console encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def discover_csv_files(data_dir: str) -> List[str]:
    """Find all CSV files in the specified directory."""
    search_path = os.path.join(data_dir, "*.csv")
    csv_files = glob.glob(search_path)
    csv_files.sort()
    return csv_files

def find_label_column(df: pd.DataFrame) -> str:
    """Identify the label/target column dynamically regardless of whitespace or casing."""
    for col in df.columns:
        if col.strip().lower() == 'label':
            return col
    for col in df.columns:
        if 'label' in col.strip().lower() or 'class' in col.strip().lower() or 'attack' in col.strip().lower():
            return col
    return ""

def inspect_single_csv(file_path: str) -> Dict[str, Any]:
    """Inspect a single CSV file and return detailed metrics."""
    filename = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f" INSPECTING FILE: {filename}")
    print(f"{'='*80}")
    
    # Try reading with utf-8, fallback to latin1 if needed
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8', encoding_errors='replace')
    except Exception:
        df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
    
    num_rows, num_cols = df.shape
    print(f"\n1. Dimensions:")
    print(f"   - Rows: {num_rows:,}")
    print(f"   - Columns: {num_cols}")
    
    print(f"\n2. Column Names & Data Types:")
    for idx, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        clean_col = str(col).encode('ascii', errors='replace').decode('ascii')
        print(f"   [{idx:02d}] '{clean_col}' -> {dtype}")
        
    label_col = find_label_column(df)
    print(f"\n3. Target / Label Column Identification:")
    if label_col:
        clean_label_col = str(label_col).encode('ascii', errors='replace').decode('ascii')
        print(f"   - Identified Label Column: '{clean_label_col}'")
        label_counts = df[label_col].value_counts(dropna=False)
        print("   - Label Distribution (Raw values & counts):")
        for label, count in label_counts.items():
            pct = (count / num_rows) * 100
            label_str = str(label).encode('ascii', errors='replace').decode('ascii').strip()
            print(f"     * '{label_str}': {count:,} ({pct:.2f}%)")
    else:
        print("   - WARNING: Could not automatically detect a 'Label' column.")
        label_counts = pd.Series(dtype=int)

    # Missing values check
    missing_series = df.isna().sum()
    total_missing = int(missing_series.sum())
    cols_with_missing = missing_series[missing_series > 0]
    print(f"\n4. Missing / NaN Values:")
    print(f"   - Total Missing Values: {total_missing:,}")
    if not cols_with_missing.empty:
        print("   - Columns with Missing Values:")
        for col, count in cols_with_missing.items():
            print(f"     * '{col}': {count:,} missing")
    else:
        print("   - No missing values found.")

    # Infinite values check
    inf_counts = {}
    total_inf = 0
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            count = int(np.isinf(s).sum())
        else:
            count = int(s.astype(str).str.strip().str.lower().isin(['inf', '+inf', '-inf', 'infinity', '-infinity']).sum())
        if count > 0:
            inf_counts[col] = count
            total_inf += count

    print(f"\n5. Infinite Values:")
    print(f"   - Total Infinite Values: {total_inf:,}")
    if inf_counts:
        print("   - Columns with Infinite Values:")
        for col, count in inf_counts.items():
            print(f"     * '{col}': {count:,} infinite values")
    else:
        print("   - No infinite values found.")

    # Duplicate rows check
    duplicate_count = int(df.duplicated().sum())
    pct_dup = (duplicate_count / num_rows) * 100 if num_rows > 0 else 0
    print(f"\n6. Duplicate Rows:")
    print(f"   - Exact Duplicate Rows: {duplicate_count:,} ({pct_dup:.2f}%)")

    # Basic Numerical Statistics Summary
    print(f"\n7. Basic Numerical Statistics (First 10 Numerical Features):")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats_df = numeric_df.describe().T[['mean', 'std', 'min', '50%', 'max']]
        print(stats_df.head(10).to_string())
    else:
        print("   - No numeric columns available for statistical summary.")

    return {
        'filename': filename,
        'rows': num_rows,
        'cols': num_cols,
        'label_col': label_col,
        'label_counts': label_counts.to_dict(),
        'total_missing': total_missing,
        'total_inf': total_inf,
        'duplicate_count': duplicate_count
    }

def inspect_all_datasets(data_dir: str = None) -> List[Dict[str, Any]]:
    """Discover and inspect all CSV files in data/raw and print an overall summary."""
    if data_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_dir = os.path.join(base_dir, "data", "raw")

    data_dir = os.path.abspath(data_dir)
    csv_files = discover_csv_files(data_dir)
    
    if not csv_files:
        print(f"No CSV files found in directory '{data_dir}'.")
        return []

    print(f"Discovered {len(csv_files)} CSV file(s) in '{data_dir}'. Starting detailed inspection...\n")
    
    results = []
    total_dataset_rows = 0
    overall_labels: Dict[str, int] = {}
    total_dataset_missing = 0
    total_dataset_inf = 0
    total_dataset_duplicates = 0

    for file_path in csv_files:
        metrics = inspect_single_csv(file_path)
        results.append(metrics)
        
        total_dataset_rows += metrics['rows']
        total_dataset_missing += metrics['total_missing']
        total_dataset_inf += metrics['total_inf']
        total_dataset_duplicates += metrics['duplicate_count']
        
        for label, count in metrics['label_counts'].items():
            clean_label = str(label).encode('ascii', errors='replace').decode('ascii').strip()
            overall_labels[clean_label] = overall_labels.get(clean_label, 0) + count

    # Overall Summary Across All Files
    print(f"\n{'#'*80}")
    print(f" OVERALL CICIDS2017 DATASET SUMMARY (ACROSS {len(csv_files)} CSV FILES)")
    print(f"{'#'*80}")
    print(f"- Total CSV Files Inspected: {len(csv_files)}")
    print(f"- Total Rows across all files: {total_dataset_rows:,}")
    print(f"- Total Missing/NaN Values: {total_dataset_missing:,}")
    print(f"- Total Infinite Values: {total_dataset_inf:,}")
    print(f"- Total Duplicate Rows: {total_dataset_duplicates:,}")
    print("\n- Aggregated Label / Traffic Class Distribution Across All Files:")
    
    sorted_labels = sorted(overall_labels.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels:
        pct = (count / total_dataset_rows) * 100 if total_dataset_rows > 0 else 0
        clean_display_label = str(label).encode('ascii', errors='replace').decode('ascii')
        print(f"  * '{clean_display_label}': {count:,} ({pct:.2f}%)")
        
    print(f"\n{'#'*80}\n")
    return results

if __name__ == "__main__":
    inspect_all_datasets()
