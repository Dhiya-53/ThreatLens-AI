import sys
import os

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath("src"))

from preprocessing.inspect_dataset import inspect_all_datasets

if __name__ == "__main__":
    raw_dir = os.path.abspath(os.path.join("data", "raw"))
    print(f"Starting ThreatLens AI - Step 1: CICIDS2017 Dataset Inspection...")
    print(f"Target Directory: {raw_dir}\n")
    inspect_all_datasets(raw_dir)
