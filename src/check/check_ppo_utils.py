import torch
from ppo_utils import categorical_log_probs, categorical_entropy, categorical_kl

# Create dummy logits and actions
logits_old = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
logits_new = torch.tensor([[1.1, 1.9], [0.4, 1.6]])
actions = torch.tensor([0, 1])

# Test log probabilities
logp = categorical_log_probs(logits_old, actions)
print("Log probabilities:", logp)

# Test entropy
ent = categorical_entropy(logits_old)
print("Entropy:", ent)

# Test KL divergence
kl = categorical_kl(logits_old, logits_new)
print("KL divergence:", kl)
