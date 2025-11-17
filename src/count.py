#!/usr/bin/env python3
"""
Analyze raw and preprocessed dataset spoiler and review counts.
- Counts reviews & spoiler sentences in raw CSV (Balance_imdb.csv)
- Counts sentences & spoiler sentences in preprocessed CSV (processed_sentences.csv)
"""

import ast
import pandas as pd
from pathlib import Path

def count_spoiler_sentences_in_raw(raw_csv_path: Path) -> int:
    df_raw = pd.read_csv(raw_csv_path, dtype=str).fillna("")
    count = 0
    for spoiler_text in df_raw["Spoiler_text"]:
        try:
            spoiler_list = ast.literal_eval(spoiler_text)
            if isinstance(spoiler_list, list):
                count += len(spoiler_list)
        except Exception:
            continue
    return count, len(df_raw)

def analyze_preprocessed(preprocessed_csv_path: Path):
    df_pre = pd.read_csv(preprocessed_csv_path)
    total_sentences = len(df_pre)
    spoiler_sentences = (df_pre["is_spoiler"] == 1).sum()
    return total_sentences, spoiler_sentences

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze raw and preprocessed spoiler dataset statistics.")
    parser.add_argument("--raw_csv", type=str, default="data/raw/Balance_imdb.csv",
                        help="Path to raw Balance_imdb.csv")
    parser.add_argument("--preprocessed_csv", type=str, default="data/processed/processed_sentences.csv",
                        help="Path to preprocessed processed_sentences.csv")
    args = parser.parse_args()

    raw_spoiler_count, total_reviews = count_spoiler_sentences_in_raw(Path(args.raw_csv))
    pre_total_sentences, pre_spoiler_count = analyze_preprocessed(Path(args.preprocessed_csv))

    print("\n===== RAW DATASET STATISTICS =====")
    print(f"Total Reviews:              {total_reviews:,}")
    print(f"Total Spoiler Sentences:    {raw_spoiler_count:,}")

    print("\n===== PREPROCESSED DATASET STATISTICS =====")
    print(f"Total Sentences:            {pre_total_sentences:,}")
    print(f"Total Spoiler Labelled:     {pre_spoiler_count:,}")

    spoiler_ratio_raw = raw_spoiler_count / total_reviews if total_reviews > 0 else 0
    spoiler_ratio_pre = pre_spoiler_count / pre_total_sentences if pre_total_sentences > 0 else 0

    print("\n===== SPOILER SENTENCE RATIOS =====")
    print(f"Spoilers per review in RAW data:       {spoiler_ratio_raw:.3f}")
    print(f"Spoiler sentences ratio in preprocessed: {spoiler_ratio_pre:.3f}")
