# src/check_judge_model.py
import torch
from judge_model import load_judges, judge_batch_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# You can change to True if GPU memory allows
use_4bit = False

# Example model
model_name = "Qwen/Qwen2.5-14B-Instruct"
judges = load_judges([model_name], device=device, use_4bit=use_4bit)
print(f"Loaded {len(judges)} judge(s) successfully.\n")

# Sample test prompts
descs = [
    "A detective uncovers a conspiracy in a coastal town.",
    "A fantasy adventure of a young hero in a magical land."
]
sents = [
    "The twist where the mayor is the culprit shocked me.",
    "I loved the cover art and fast shipping."
]

# Proposed actions (1 = CONTENT/SPOILER, 0 = NON-CONTENT/NON-SPOILER)
actions = torch.tensor([1, 0], dtype=torch.long, device=device)

tok, model, name = judges[0]
prompts = []
for d, s, a in zip(descs, sents, actions.tolist()):
    label = "CONTENT" if a == 1 else "NON-CONTENT"
    system = (
        "You are a strict evaluator. Classify the sentence as CONTENT or NON-CONTENT.\n"
        "Return strictly a JSON object: {\"reward\": <float 0..1>}\n"
    )
    user = f"Book description:\n{d}\n\nReview sentence:\n{s}\n\nProposed label: {label}\nRespond with JSON only."
    prompts.append(system + "\n" + user)

# Generate rewards
rewards = judge_batch_json(tok, model, prompts, device=device, max_new_tokens=50, temperature=0.1)
for i, r in enumerate(rewards):
    print(f"Prompt {i} reward: {r.item():.3f}")
