#!/usr/bin/env python3
"""
Preprocess IMDb-like movie review dataset:
- Load CSV (Balance_imdb.csv)
- Split each review_text into individual sentences using spaCy
- Parse "Spoiler_text" from each row (stringified Python list)
- Mark each sentence as spoiler (1) if it matches any Spoiler_text line (relaxed matching)
- Export a single processed_sentences.csv
"""

import ast
import csv
import string
from pathlib import Path
from typing import Iterable, List
import pandas as pd
import spacy

# ---------- Configurable parameters ----------
DEFAULT_INPUT = "data/raw/Balance_imdb.csv"
DEFAULT_OUTPUT_DIR = "data/processed"
SEED = 42

# ---------- spaCy sentence splitter ----------
print("🔹 Initializing spaCy sentencizer...")
nlp = spacy.blank("en")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def split_sentences_batch(texts: Iterable[str], batch_size: int = 256, n_process: int = 1) -> List[List[str]]:
    """Split a batch of texts into sentences"""
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        results.append(sentences)
    return results

def parse_spoiler_list(value: str) -> List[str]:
    """Safely parse Spoiler_text string -> list of sentences"""
    if not isinstance(value, str) or value.strip() == "" or value.strip() == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str)]
        return []
    except Exception:
        return []

def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for relaxed matching"""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def preprocess(input_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / "processed_sentences.csv"

    print(f"🔹 Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False, dtype=str)
    df = df.fillna("")

    # Verify required columns
    required_cols = {"movie_id", "review_text", "Spoiler_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    print(f"✅ Found {len(df)} reviews. Starting sentence splitting...")

    # Split review_text into sentences
    reviews = df["review_text"].astype(str).tolist()
    batched_sents = split_sentences_batch(reviews, batch_size=256, n_process=1)

    rows = []
    for (row_idx, _), movie_id, review_text, spoiler_raw, sents in zip(
        df.iterrows(), df["movie_id"], df["review_text"], df["Spoiler_text"], batched_sents):
        spoiler_lines = parse_spoiler_list(spoiler_raw)
        normalized_spoiler_lines = [normalize_text(sl) for sl in spoiler_lines]

        for i, sent in enumerate(sents):
            sent_norm = normalize_text(sent)
            is_spoiler = 1 if sent_norm in normalized_spoiler_lines else 0

            rows.append({
                "movie_id": movie_id,
                "sentence_id": f"{movie_id}_{row_idx}_{i}",
                "sentence_text": sent,
                "is_spoiler": is_spoiler
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(processed_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Saved {len(out_df)} sentences to {processed_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess IMDb-like spoiler dataset into per-sentence file with relaxed matching.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT, help="Path to Balance_imdb.csv")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to store processed_sentences.csv")
    args = parser.parse_args()

    preprocess(Path(args.input_csv), Path(args.output_dir))
