# check_datasets.py
import os
from datasets import SentenceDataset, make_collate_fn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    # Path to a small CSV sample
    csv_path = "data/processed/processed_sentences.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Please place a CSV there for testing.")
        return

    # Load dataset
    dataset = SentenceDataset(csv_path, max_rows=5)
    print(f"Loaded {len(dataset)} sentences from CSV.")

    # Check individual items
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"Item {i}:")
        print(f"  text: {item['text'][:60]}...")  # only first 60 chars
        print(f"  desc: {item['desc'][:30]}...")
        print(f"  sent: {item['sent'][:30]}...")
        print("-" * 50)

    # Test collate_fn with tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=True)
    collate_fn = make_collate_fn(tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("Batch input_ids shape:", batch["input_ids"].shape)
    print("Batch raw_desc:", batch["raw_desc"])
    print("Batch raw_sent:", batch["raw_sent"])

if __name__ == "__main__":
    main()
