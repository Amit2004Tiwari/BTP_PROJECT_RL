# check_judge_cache.py
from datasets import SentenceDataset, make_collate_fn
from judge_cache import JudgeCache
from transformers import AutoTokenizer

# Load dataset (small sample)
dataset = SentenceDataset("data/processed/processed_sentences.csv", max_rows=5)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=True)
collate_fn = make_collate_fn(tokenizer, max_length=128)
batch = collate_fn([dataset[i] for i in range(len(dataset))])

# Initialize JudgeCache
cache = JudgeCache("outputs/test_judge_cache.sqlite")

# Test storing and retrieving rewards
for i, (desc, sent) in enumerate(zip(batch["raw_desc"], batch["raw_sent"])):
    key = cache.make_key("content", desc, sent, action=1)
    cache.put(key, reward=0.9 + i*0.01)
    retrieved = cache.get(key)
    print(f"Sentence {i} | Reward stored: {0.9 + i*0.01} | Retrieved: {retrieved}")

cache.close()
