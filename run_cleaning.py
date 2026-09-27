import sys
import os

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath("src"))

from preprocessing.cleaning import CICIDSDataCleaner

if __name__ == "__main__":
    raw_dir = os.path.abspath(os.path.join("data", "raw"))
    processed_dir = os.path.abspath(os.path.join("data", "processed"))
    
    print("Starting ThreatLens AI - Step 2: CICIDS2017 Data Cleaning Pipeline...")
    print(f"Raw Data Directory:       {raw_dir}")
    print(f"Processed Output Target:  {processed_dir}\n")
    
    cleaner = CICIDSDataCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
    df_clean = cleaner.clean()
    cleaner.print_audit_report()
