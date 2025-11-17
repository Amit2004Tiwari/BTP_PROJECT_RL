#!/usr/bin/env python3
from train_content import train_content

train_content(
    input_csv="data/processed/processed_sentences.csv",  # a small test CSV
    output_dir="outputs/test_rl_content",
    batch_size=2,
    epochs=1,
)
print("train_content.py ran successfully on small dataset.")
