import os
import sys
import glob
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Ensure stdout uses utf-8 encoding for clean printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file to check content equality."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

class CICIDSDataCleaner:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = os.path.abspath(raw_dir)
        self.processed_dir = os.path.abspath(processed_dir)
        self.report: Dict[str, Any] = {
            'before': {},
            'after': {},
            'transformations': {},
            'file_verification': []
        }
        os.makedirs(self.processed_dir, exist_ok=True)

    def verify_raw_files(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Find all CSV files and verify content duplicates using SHA-256 hashes."""
        csv_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        csv_files.sort()
        
        seen_hashes = {}
        unique_files = []
        file_metrics = []
        
        for fpath in csv_files:
            fname = os.path.basename(fpath)
            fsize = os.path.getsize(fpath)
            fhash = compute_file_sha256(fpath)
            
            if fhash in seen_hashes:
                file_metrics.append({
                    'filename': fname,
                    'size_bytes': fsize,
                    'sha256': fhash,
                    'status': 'EXCLUDED_DUPLICATE_CONTENT',
                    'identical_to': seen_hashes[fhash]
                })
            else:
                seen_hashes[fhash] = fname
                unique_files.append(fpath)
                file_metrics.append({
                    'filename': fname,
                    'size_bytes': fsize,
                    'sha256': fhash,
                    'status': 'RETAINED_UNIQUE',
                    'identical_to': None
                })
                
        self.report['file_verification'] = file_metrics
        return unique_files, file_metrics

    def clean(self) -> pd.DataFrame:
        print("="*80)
        print(" THREATLENS AI — STEP 2: DATA CLEANING PIPELINE INITIALIZATION")
        print("="*80)

        # 1. File Verification & Discovery
        print("\n[STAGE 1] Verifying Raw CSV Files Content & Hashing...")
        unique_files, file_verification = self.verify_raw_files()
        
        print(f"Total CSV Files Discovered: {len(file_verification)}")
        for item in file_verification:
            if item['status'] == 'RETAINED_UNIQUE':
                print(f" - [RETAINED UNIQUE] {item['filename']} ({item['size_bytes']:,} bytes)")
            else:
                print(f" - [EXCLUDED DUPLICATE COPY] {item['filename']} (Content-identical to {item['identical_to']})")

        # 2. Load & Validate Schemas
        print("\n[STAGE 2] Schema Validation & Column Header Standardization...")
        loaded_dfs = []
        reference_cols = None
        
        raw_total_rows = 0
        raw_total_missing = 0
        raw_total_inf = 0
        raw_total_duplicates = 0
        raw_class_distribution: Dict[str, int] = {}
        
        for fpath in unique_files:
            fname = os.path.basename(fpath)
            try:
                df_curr = pd.read_csv(fpath, low_memory=False, encoding='utf-8', encoding_errors='replace')
            except Exception:
                df_curr = pd.read_csv(fpath, low_memory=False, encoding='latin1')

            # Normalize column headers (strip leading/trailing whitespace)
            df_curr.columns = [str(col).strip() for col in df_curr.columns]
            
            # Map any label column variant to 'Label'
            for col in df_curr.columns:
                if col.lower() == 'label':
                    df_curr.rename(columns={col: 'Label'}, inplace=True)
                    break

            # Validate Schema Compatibility
            curr_cols = list(df_curr.columns)
            if reference_cols is None:
                reference_cols = curr_cols
            else:
                if curr_cols != reference_cols:
                    diff = set(curr_cols).symmetric_difference(set(reference_cols))
                    raise ValueError(f"Schema mismatch detected in file {fname}! Discrepant columns: {diff}")
            
            # Record Before Metrics per file
            n_rows = len(df_curr)
            raw_total_rows += n_rows
            raw_total_missing += int(df_curr.isna().sum().sum())
            
            # Count infinities
            for c in df_curr.select_dtypes(include=[np.number]).columns:
                raw_total_inf += int(np.isinf(df_curr[c]).sum())
                
            raw_total_duplicates += int(df_curr.duplicated().sum())
            
            if 'Label' in df_curr.columns:
                lbl_counts = df_curr['Label'].value_counts(dropna=False)
                for lbl, cnt in lbl_counts.items():
                    clean_lbl = str(lbl).encode('ascii', errors='replace').decode('ascii').strip()
                    raw_class_distribution[clean_lbl] = raw_class_distribution.get(clean_lbl, 0) + cnt
            
            loaded_dfs.append(df_curr)

        print(f"Schema validation passed! All {len(unique_files)} unique files have matching {len(reference_cols)} columns.")
        
        # Combine DataFrame
        print("\nCombining verified unique datasets...")
        df = pd.concat(loaded_dfs, ignore_index=True)
        del loaded_dfs  # Free memory
        
        # Store Before Metrics
        self.report['before'] = {
            'total_files_discovered': len(file_verification),
            'unique_files_loaded': len(unique_files),
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'total_missing': int(df.isna().sum().sum()),
            'total_inf': self._count_infinities(df),
            'total_duplicates': int(df.duplicated().sum()),
            'class_distribution': raw_class_distribution
        }

        # 3. Convert Infinities to NaN (STAGE 3)
        print("\n[STAGE 3] Converting Positive & Negative Infinities (±inf) to NaN...")
        inf_converted_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_mask = np.isinf(df[col])
            cnt = int(inf_mask.sum())
            if cnt > 0:
                inf_converted_count += cnt
                df.loc[inf_mask, col] = np.nan
        print(f"Converted {inf_converted_count:,} infinite values to NaN.")
        self.report['transformations']['inf_converted_to_nan'] = inf_converted_count

        # 4. Impute Missing Values (STAGE 4)
        print("\n[STAGE 4] Imputing Missing Values (NaN) with Feature Medians...")
        imputed_counts = {}
        total_imputed = 0
        missing_cols = df.columns[df.isna().any()].tolist()
        
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Compute median strictly from valid finite entries
                col_median = df[col].dropna().median()
                n_missing = int(df[col].isna().sum())
                df[col] = df[col].fillna(col_median)
                imputed_counts[col] = {'count': n_missing, 'median_used': float(col_median)}
                total_imputed += n_missing
                print(f" - Column '{col}': Imputed {n_missing:,} missing values with median = {col_median:.4f}")
            else:
                print(f" - WARNING: Non-numeric column '{col}' has missing values. Filling with mode.")
                df[col] = df[col].fillna(df[col].mode()[0])

        print(f"Total Missing Values Imputed: {total_imputed:,}")
        self.report['transformations']['imputed_missing_details'] = imputed_counts
        self.report['transformations']['total_imputed'] = total_imputed

        # 5. Label Sanitization & ASCII Repair (STAGE 5)
        print("\n[STAGE 5] Sanitizing Labels & Repairing Non-ASCII Characters...")
        raw_labels_count = len(df['Label'].unique())
        
        # Label Repair Mapping
        def repair_label_string(val: Any) -> str:
            s = str(val).encode('ascii', errors='replace').decode('ascii').strip()
            if 'Web Attack' in s:
                if 'Brute Force' in s:
                    return 'Web Attack - Brute Force'
                elif 'XSS' in s:
                    return 'Web Attack - XSS'
                elif 'Sql Injection' in s or 'SQL' in s:
                    return 'Web Attack - Sql Injection'
            return s

        df['Label'] = df['Label'].apply(repair_label_string)
        sanitized_labels_count = len(df['Label'].unique())
        print(f"Sanitized label column. Unique classes verified: {sanitized_labels_count} (Raw unique: {raw_labels_count}).")
        
        # Verify no missing labels
        null_labels = int(df['Label'].isna().sum())
        if null_labels > 0:
            raise ValueError(f"Error: Found {null_labels} null/empty labels after sanitization!")

        # 6. Safe Range-Checked Downcasting (STAGE 6)
        print("\n[STAGE 6] Safe Range-Checked Numeric Downcasting...")
        initial_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        for col in df.columns:
            if col == 'Label':
                continue
            s = df[col]
            if pd.api.types.is_integer_dtype(s):
                c_min, c_max = s.min(), s.max()
                if c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = s.astype(np.int32)
            elif pd.api.types.is_float_dtype(s):
                c_min, c_max = s.min(), s.max()
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = s.astype(np.float32)

        final_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        mem_saved_pct = ((initial_mem_mb - final_mem_mb) / initial_mem_mb) * 100
        print(f"Memory downcasted safely: {initial_mem_mb:.2f} MB -> {final_mem_mb:.2f} MB ({mem_saved_pct:.1f}% RAM reduction).")
        self.report['transformations']['memory_before_mb'] = initial_mem_mb
        self.report['transformations']['memory_after_mb'] = final_mem_mb

        # 7. Per-Class Duplicate Analysis & Deterministic Deduplication (STAGE 7)
        print("\n[STAGE 7] Per-Class Duplicate Analysis & Exact-Row Deduplication...")
        
        # Analyze duplicates per class before removal
        dup_mask = df.duplicated(keep='first')
        total_dup_rows = int(dup_mask.sum())
        dup_by_class = df[dup_mask]['Label'].value_counts().to_dict()
        
        print(f"Total Exact Duplicate Rows Identified: {total_dup_rows:,}")
        print("Duplicate Rows Impact Breakdown by Class:")
        class_before_counts = df['Label'].value_counts().to_dict()
        
        for cls, count in class_before_counts.items():
            dups = dup_by_class.get(cls, 0)
            pct_removed = (dups / count) * 100 if count > 0 else 0
            print(f" - '{cls}': {count:,} total rows | {dups:,} duplicates ({pct_removed:.2f}% removed)")

        # Perform deterministic exact deduplication
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        class_after_counts = df['Label'].value_counts().to_dict()
        self.report['transformations']['duplicates_by_class'] = dup_by_class
        self.report['transformations']['total_duplicates_removed'] = total_dup_rows

        # 8. Store After Metrics (STAGE 8)
        self.report['after'] = {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'total_missing': int(df.isna().sum().sum()),
            'total_inf': self._count_infinities(df),
            'total_duplicates': int(df.duplicated().sum()),
            'class_distribution': class_after_counts
        }

        # Save Artifacts
        print("\n[STAGE 8] Saving Cleaned Dataset to data/processed/...")
        csv_path = os.path.join(self.processed_dir, "cleaned_cicids2017.csv")
        parquet_path = os.path.join(self.processed_dir, "cleaned_cicids2017.parquet")

        # Save CSV
        print(f"Saving CSV: {csv_path}...")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved successfully! ({os.path.getsize(csv_path) / (1024*1024):.2f} MB)")
        self.report['output_csv'] = csv_path

        # Save Parquet if pyarrow/fastparquet available
        try:
            print(f"Saving Parquet: {parquet_path}...")
            df.to_parquet(parquet_path, index=False)
            print(f"Parquet saved successfully! ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")
            self.report['output_parquet'] = parquet_path
        except Exception as e:
            print(f"WARNING: Parquet export skipped (Reason: {e}). CSV export complete.")
            self.report['output_parquet'] = None

        print("\nData Cleaning Pipeline Completed Successfully!\n")
        return df

    def _count_infinities(self, df: pd.DataFrame) -> int:
        total_inf = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            total_inf += int(np.isinf(df[col]).sum())
        return total_inf

    def print_audit_report(self):
        print("\n" + "="*80)
        print(" THREATLENS AI — DATA CLEANING AUDIT REPORT (STEP 2)")
        print("="*80)
        
        b = self.report['before']
        a = self.report['after']
        t = self.report['transformations']
        
        print("\n1. RAW FILE VERIFICATION SUMMARY:")
        for f in self.report['file_verification']:
            if f['status'] == 'RETAINED_UNIQUE':
                print(f" - [RETAINED UNIQUE] {f['filename']} (SHA-256: {f['sha256'][:12]}...)")
            else:
                print(f" - [EXCLUDED COPY] {f['filename']} (Content-identical to {f['identical_to']})")

        print("\n2. BEFORE vs. AFTER CLEANING SUMMARY:")
        print(f" {'Metric':<30} | {'BEFORE CLEANING':<22} | {'AFTER CLEANING':<22}")
        print("-" * 80)
        print(f" {'Total Rows':<30} | {b['total_rows']:<22,} | {a['total_rows']:<22,}")
        print(f" {'Total Columns':<30} | {b['total_cols']:<22} | {a['total_cols']:<22}")
        print(f" {'Missing Values (NaN)':<30} | {b['total_missing']:<22,} | {a['total_missing']:<22}")
        print(f" {'Infinite Values (±inf)':<30} | {b['total_inf']:<22,} | {a['total_inf']:<22}")
        print(f" {'Exact Duplicate Rows':<30} | {b['total_duplicates']:<22,} | {a['total_duplicates']:<22}")
        print(f" {'Memory Footprint':<30} | {t.get('memory_before_mb', 0):.2f} MB{'':<14} | {t.get('memory_after_mb', 0):.2f} MB")

        print("\n3. FINAL CLASS / ATTACK DISTRIBUTION (15 CLASSES PRESERVED):")
        print(f" {'Attack / Traffic Class':<32} | {'Cleaned Rows':<15} | {'Percentage (%)':<15}")
        print("-" * 70)
        
        total_final = a['total_rows']
        sorted_classes = sorted(a['class_distribution'].items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_classes:
            pct = (count / total_final) * 100 if total_final > 0 else 0
            print(f" '{cls}':{'' if len(cls) > 30 else ' '*(30-len(cls))} | {count:<15,} | {pct:.4f}%")

        print("\n4. OUTPUT ARTIFACT PATHS:")
        print(f" - CSV Output:     {self.report.get('output_csv')}")
        print(f" - Parquet Output: {self.report.get('output_parquet')}")
        print("="*80 + "\n")
