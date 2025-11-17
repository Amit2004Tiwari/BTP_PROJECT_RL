#!/usr/bin/env python3
"""
Label-free PPO trainer for sentence classification using an LLM judge (RLAIF), with:
- Strict JSON rewards from the judge
- SQLite caching of judge responses
- Advantage normalization + KL penalty + entropy bonus
- Optional class-imbalance reward weights
- Optional judge ensemble averaging
- Diagnostics and optional offline reward data collection

Run (content stage):
  python src/rl_judge_online.py --task content --few_shot --entropy_coef 0.01

Run (spoiler stage):
  python src/rl_judge_online.py --task spoiler --few_shot --class_weight_1 1.5 --entropy_coef 0.01
"""

import os
import re
import math
import json
import time
import sqlite3
import hashlib
import random
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Optional bitsandbytes for 4-bit judge
HAS_BNB = False
try:
    import bitsandbytes as bnb  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rl_judge_online")

# -------------------------
# Data
# -------------------------
class SentenceDataset(Dataset):
    def __init__(self, csv_path: str, max_rows: Optional[int] = None):
        df = pd.read_csv(csv_path)
        for col in ("description_text", "sentence_text"):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")
        if max_rows is not None:
            df = df.iloc[:max_rows].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        desc = str(row["description_text"])
        sent = str(row["sentence_text"])
        text = f"[DESCRIPTION] {desc} [SENTENCE] {sent}"
        return {"text": text, "desc": desc, "sent": sent}

def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch: List[Dict]):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["raw_desc"] = [b["desc"] for b in batch]
        enc["raw_sent"] = [b["sent"] for b in batch]
        return enc
    return collate_fn

# -------------------------
# SQLite Judge Cache
# -------------------------
class JudgeCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, reward REAL, ts REAL)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON cache(ts)")
        self.conn.commit()

    @staticmethod
    def make_key(task: str, desc: str, sent: str, action: int) -> str:
        h = hashlib.sha256()
        norm = lambda s: re.sub(r"\s+", " ", s.strip())
        payload = json.dumps(
            {"task": task, "desc": norm(desc), "sent": norm(sent), "action": int(action)},
            ensure_ascii=False, sort_keys=True,
        )
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[float]:
        c = self.conn.cursor()
        c.execute("SELECT reward FROM cache WHERE key=?", (key,))
        row = c.fetchone()
        return None if row is None else float(row[0])

    def put(self, key: str, reward: float):
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO cache(key, reward, ts) VALUES (?, ?, ?)",
            (key, float(reward), time.time()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

# -------------------------
# PPO helpers
# -------------------------
def categorical_log_probs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(1, actions.view(-1, 1)).squeeze(1)

def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1)

def categorical_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    old_logp = torch.log_softmax(old_logits, dim=-1)
    new_logp = torch.log_softmax(new_logits, dim=-1)
    old_p = torch.softmax(old_logits, dim=-1)
    kl = torch.sum(old_p * (old_logp - new_logp), dim=-1)
    return kl

# -------------------------
# Judge prompts
# -------------------------
def build_rubric(task: str, few_shot: bool = False) -> str:
    if task == "content":
        base = (
            "You are a strict evaluator for classifying review sentences as CONTENT or NON-CONTENT.\n"
            "CONTENT: plot-bearing facts, events, characters, settings, outcomes, or factual critique tied to the book.\n"
            "NON-CONTENT: greetings, meta-comments, general opinions not tied to plot, star ratings, unrelated jokes.\n"
            "Return strictly a JSON object with one key 'reward' and a float value between 0 and 1 indicating correctness of the proposed label.\n"
            "Output format example: {\"reward\": 0.87}\n"
        )
        if few_shot:
            base += (
                "\nExamples:\n"
                "Desc: A detective uncovers a conspiracy in a coastal town.\n"
                "Sent: The twist where the mayor is the culprit shocked me.\n"
                "Proposed: CONTENT -> {\"reward\": 0.9}\n"
                "Desc: A detective uncovers a conspiracy in a coastal town.\n"
                "Sent: I loved the cover art and fast shipping.\n"
                "Proposed: NON-CONTENT -> {\"reward\": 0.95}\n"
            )
        return base
    else:
        base = (
            "You are a strict evaluator for SPOILER detection.\n"
            "SPOILER: reveals key plot points, twists, endings, deaths, or hidden identities not in the public synopsis.\n"
            "NON-SPOILER: general opinions or details obvious from the description.\n"
            "Return strictly a JSON object: {\"reward\": <float 0..1>} for the proposed label's correctness.\n"
            "Output format example: {\"reward\": 0.87}\n"
        )
        if few_shot:
            base += (
                "\nExamples:\n"
                "Desc: A quest to find a lost heirloom.\n"
                "Sent: The heirloom turns out to be fake in the final chapter.\n"
                "Proposed: SPOILER -> {\"reward\": 0.95}\n"
                "Desc: A quest to find a lost heirloom.\n"
                "Sent: The pacing felt slow in the middle.\n"
                "Proposed: NON-SPOILER -> {\"reward\": 0.9}\n"
            )
        return base

def build_prompts(task: str, descs: List[str], sents: List[str], actions: torch.Tensor, few_shot: bool) -> List[str]:
    prompts = []
    for d, s, a in zip(descs, sents, actions.tolist()):
        proposed = ("CONTENT" if a == 1 else "NON-CONTENT") if task == "content" else ("SPOILER" if a == 1 else "NON-SPOILER")
        system = build_rubric(task, few_shot=few_shot)
        user = (
            f"Book description:\n{d}\n\n"
            f"Review sentence:\n{s}\n\n"
            f"Proposed label: {proposed}\n"
            f"Respond with a single line JSON only: {{\"reward\": <float>}}"
        )
        prompts.append(system + "\n" + user)
    return prompts

def parse_reward_json(text: str) -> float:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return 0.5
    obj_txt = m.group(0)
    try:
        obj = json.loads(obj_txt)
        v = float(obj.get("reward", 0.5))
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.5

# -------------------------
# Judge wrappers (ensemble-capable)
# -------------------------
@torch.no_grad()
def judge_batch_json(
    judge_tok,
    judge_model,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> torch.Tensor:
    inputs = judge_tok(prompts, padding=True, truncation=True, max_length=1536, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen = judge_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(1e-5, temperature),
        top_p=0.9,
        eos_token_id=judge_tok.eos_token_id,
        pad_token_id=judge_tok.eos_token_id,
    )
    outputs = gen[:, inputs["input_ids"].shape[1]:]
    texts = judge_tok.batch_decode(outputs, skip_special_tokens=True)
    rewards = [parse_reward_json(t) for t in texts]
    return torch.tensor(rewards, dtype=torch.float32, device=device)

def load_judges(judge_models: List[str], device: torch.device, use_4bit: bool):
    judges = []
    for name in judge_models:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True, use_safetensors=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs = {"torch_dtype": dtype, "trust_remote_code": True, "use_safetensors": True}
        if use_4bit and HAS_BNB:
            kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, device_map="auto"))
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device).eval()
        judges.append((tok, model, name))
    return judges

# -------------------------
# Main PPO training
# -------------------------
def train_judge_online(
    task: str = "content",
    input_csv: str = "data/processed/processed_sentences.csv",
    output_dir: str = "outputs/rl_content",
    policy_model_name: str = "microsoft/deberta-v3-large",
    judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    use_4bit_judge: bool = False,
    judge_ensemble: Optional[str] = None,  # comma-separated list
    judge_max_new_tokens: int = 24,
    judge_temperature: float = 0.1,
    few_shot: bool = True,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-5,
    max_length: int = 256,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 2,
    reward_temp: float = 1.0,
    class_weight_0: float = 1.0,
    class_weight_1: float = 1.0,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = True,
    device_str: Optional[str] = None,
    sample_limit: Optional[int] = None,
    cache_path: str = "outputs/judge_cache/judge_cache.sqlite",
    diagnostics_path: str = "outputs/judge_diagnostics.jsonl",
    collect_judge_rewards: Optional[str] = None,  # jsonl lines (task,text,action,reward)
):
    assert task in ("content", "spoiler"), "task must be 'content' or 'spoiler'"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # Data
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name, use_fast=True, use_safetensors=True)
    collate_fn = make_collate_fn(tokenizer, max_length)
    dataset = SentenceDataset(input_csv, max_rows=sample_limit)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )

    # Policy (force safetensors path; bfloat16 on CUDA)
    pol_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = AutoModelForSequenceClassification.from_pretrained(
        policy_model_name,
        num_labels=2,
        torch_dtype=pol_dtype,
        use_safetensors=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    policy.to(device)
    optimizer = AdamW(policy.parameters(), lr=lr)
    total_steps = len(loader) * epochs * max(1, ppo_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and pol_dtype == torch.float16) else None

    # Judges (support ensemble)
    judge_names = [s.strip() for s in (judge_ensemble.split(",") if judge_ensemble else [judge_model_name])]
    judges = load_judges(judge_names, device, use_4bit_judge and HAS_BNB)

    # Cache and diagnostics
    cache = JudgeCache(cache_path)
    diag_f = Path(diagnostics_path); diag_f.parent.mkdir(parents=True, exist_ok=True)

    # PPO state
    best_avg_reward = -1.0
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        reward_sum = 0.0
        count = 0
        action_hist = [0, 0]
        reward_accum_0 = 0.0
        reward_accum_1 = 0.0

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            descs = batch["raw_desc"]; sents = batch["raw_sent"]

            # Old policy + sample actions
            with torch.no_grad():
                old_out = policy(input_ids=input_ids, attention_mask=attention_mask)
                old_logits = old_out.logits
                old_probs = torch.softmax(old_logits, dim=-1)
                dist = torch.distributions.Categorical(probs=old_probs)
                actions = dist.sample()
                old_logp = categorical_log_probs(old_logits, actions)

            # Class weights (0/1)
            weights = torch.where(actions == 1, torch.tensor(class_weight_1, device=device), torch.tensor(class_weight_0, device=device))

            # Judge rewards with cache and ensemble
            keys = [cache.make_key(task, d, s, int(a)) for d, s, a in zip(descs, sents, actions.tolist())]
            rewards = torch.zeros(len(keys), dtype=torch.float32, device=device)
            cached_mask = []

            for i, k in enumerate(keys):
                val = cache.get(k)
                if val is not None:
                    rewards[i] = float(val); cached_mask.append(True)
                else:
                    cached_mask.append(False)

            need_idx = [i for i, m in enumerate(cached_mask) if not m]
            if need_idx:
                prompts = build_prompts(task, [descs[i] for i in need_idx], [sents[i] for i in need_idx], actions[need_idx], few_shot=few_shot)
                ens_rewards = []
                for tok, mdl, name in judges:
                    r = judge_batch_json(tok, mdl, prompts, device, max_new_tokens=judge_max_new_tokens, temperature=judge_temperature)
                    ens_rewards.append(r)
                r_mean = torch.stack(ens_rewards, dim=0).mean(dim=0)
                if reward_temp != 1.0:
                    r_mean = torch.clamp(r_mean ** (1.0 / max(1e-5, reward_temp)), 0.0, 1.0)
                for j, i_data in enumerate(need_idx):
                    rewards[i_data] = r_mean[j]
                    cache.put(keys[i_data], float(r_mean[j].item()))
                if collect_judge_rewards:
                    Path(collect_judge_rewards).parent.mkdir(parents=True, exist_ok=True)
                    with open(collect_judge_rewards, "a", encoding="utf-8") as fcsv:
                        for j, i_data in enumerate(need_idx):
                            entry = {"task": task, "text": f"[DESCRIPTION] {descs[i_data]} [SENTENCE] {sents[i_data]}", "action": int(actions[i_data].item()), "reward": float(r_mean[j].item())}
                            fcsv.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Weighted rewards
            rewards = rewards * weights

            # Diagnostics
            a_np = actions.detach().cpu().numpy()
            r_np = rewards.detach().cpu().numpy()
            action_hist[0] += int((a_np == 0).sum()); action_hist[1] += int((a_np == 1).sum())
            if (a_np == 0).any(): reward_accum_0 += float(r_np[a_np == 0].mean())
            if (a_np == 1).any(): reward_accum_1 += float(r_np[a_np == 1].mean())

            # Normalize advantages (mean/std)
            advantages = rewards - rewards.mean()
            std = rewards.std().clamp_min(1e-6)
            advantages = advantages / std

            # PPO inner epochs
            for _ in range(max(1, ppo_epochs)):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    new_out = policy(input_ids=input_ids, attention_mask=attention_mask)
                    new_logits = new_out.logits
                    new_logp = categorical_log_probs(new_logits, actions)

                    ratio = torch.exp(new_logp - old_logp)
                    unclipped = ratio * advantages
                    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                    ppo_obj = torch.min(unclipped, clipped).mean()

                    kl = categorical_kl(old_logits.detach(), new_logits).mean()
                    ent = categorical_entropy(new_logits).mean()
                    loss = -(ppo_obj) + kl_coef * kl - entropy_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            reward_sum += rewards.mean().item() * input_ids.size(0)
            count += input_ids.size(0)

            # Periodic diagnostics
            if step % 50 == 0:
                diag_rec = {
                    "epoch": epoch, "step": step,
                    "action_hist": {"0": action_hist[0], "1": action_hist[1]},
                    "batch_reward_mean": float(rewards.mean().item()),
                    "batch_reward_std": float(rewards.std().item()),
                }
                with open(diag_f, "a", encoding="utf-8") as fd:
                    fd.write(json.dumps(diag_rec) + "\n")

        avg_reward = reward_sum / max(1, count)
        per_class_0 = reward_accum_0 / max(1, action_hist[0]) if action_hist[0] else 0.0
        per_class_1 = reward_accum_1 / max(1, action_hist[1]) if action_hist[1] else 0.0
        logger.info(
            f"Epoch {epoch}/{epochs} | PPO Loss: {epoch_loss:.4f} | Avg Reward: {avg_reward:.4f} | "
            f"Actions 0/1: {action_hist[0]}/{action_hist[1]} | ClassReward 0: {per_class_0:.4f} 1: {per_class_1:.4f}"
        )

        # Save best by avg_reward
        save_dir = Path(output_dir) / "best_policy_model"
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            if save_dir.exists():
                try:
                    shutil.rmtree(save_dir)
                except Exception as e:
                    logger.warning(f"Could not clean existing best dir: {e}")
            save_dir.mkdir(parents=True, exist_ok=True)
            try:
                policy.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                with open(Path(output_dir) / "training_meta.json", "w") as f:
                    json.dump({"best_avg_reward": best_avg_reward, "epoch": epoch}, f)
                logger.info(f"Saved best policy to {save_dir} (avg_reward={best_avg_reward:.4f})")
            except Exception as e:
                logger.warning(f"Save failed: {e}")

    cache.close()
    logger.info("Judge-online RL training complete.")
    return {"best_avg_reward": best_avg_reward}

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Label-free PPO with LLM judge for sentence classification.")
    p.add_argument("--task", type=str, choices=["content", "spoiler"], default="content")
    p.add_argument("--input", type=str, default="data/processed/processed_sentences.csv")
    p.add_argument("--output_dir", type=str, default="outputs/rl_content")
    p.add_argument("--policy_model", type=str, default="microsoft/deberta-v3-large")
    p.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--use_4bit_judge", action="store_true")
    p.add_argument("--judge_ensemble", type=str, default=None, help="Comma-separated judge models to ensemble.")
    p.add_argument("--judge_max_new_tokens", type=int, default=24)
    p.add_argument("--judge_temperature", type=float, default=0.1)
    p.add_argument("--few_shot", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--reward_temp", type=float, default=1.0)
    p.add_argument("--class_weight_0", type=float, default=1.0)
    p.add_argument("--class_weight_1", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--cache_path", type=str, default="outputs/judge_cache/judge_cache.sqlite")
    p.add_argument("--diagnostics_path", type=str, default="outputs/judge_diagnostics.jsonl")
    p.add_argument("--collect_judge_rewards", type=str, default=None)
    args = p.parse_args()

    train_judge_online(
        task=args.task,
        input_csv=args.input,
        output_dir=args.output_dir,
        policy_model_name=args.policy_model,
        judge_model_name=args.judge_model,
        use_4bit_judge=args.use_4bit_judge,
        judge_ensemble=args.judge_ensemble,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_temperature=args.judge_temperature,
        few_shot=args.few_shot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        reward_temp=args.reward_temp,
        class_weight_0=args.class_weight_0,
        class_weight_1=args.class_weight_1,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device_str=args.device,
        sample_limit=args.sample_limit,
        cache_path=args.cache_path,
        diagnostics_path=args.diagnostics_path,
        collect_judge_rewards=args.collect_judge_rewards,
    )
