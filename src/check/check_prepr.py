#!/usr/bin/env python3
"""
Check the output of preprocess.py
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/processed")

# List all CSVs in output directory
csv_files = list(OUTPUT_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in {OUTPUT_DIR}:")
for f in csv_files:
    print(" -", f.name)

# Load and show a quick preview of processed_sentences.csv
processed_file = OUTPUT_DIR / "processed_sentences.csv"
if processed_file.exists():
    df = pd.read_csv(processed_file)
    print(f"\n✅ Sample from {processed_file.name}:")
    print(df.head())
    print(f"\nTotal sentences: {len(df)}, Total columns: {df.columns.tolist()}")
else:
    print(f"⚠️ {processed_file.name} not found!")
