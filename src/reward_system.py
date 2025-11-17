#!/usr/bin/env python3
"""
Ultra-Fast Sophisticated Reward System (Movie Version + Judge Model)
- Uses table base reward for spoiler GT cases (1,3,5) only
- Uses judge-model base reward for non-spoiler GT cases (2,4,6)
- Keeps all other components unchanged
"""

import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import deque


# =========================
# Judge model plug-in
# =========================
class JudgeModel:
    """
    Fast judge model stub.
    Replace `score` with a real LLM/RM inference if desired.
    The method should return a scalar base reward for (sentence, case).
    """

    def __init__(self, min_reward: float = -10.0, max_reward: float = 8.0):
        self.min_reward = min_reward
        self.max_reward = max_reward

    def score(self, sentence: str, case: int, deberta_conf: float, bert_conf: float) -> float:
        """
        Return a base reward for non-spoiler GT cases (2,4,6).
        Current heuristic:
        - Case 2 (FP): penalize more if confidences are high
        - Case 4 (TN): reward with modest positive score that grows with bert_conf
        - Case 6 (Filtered TN): small positive (or small negative) based on deberta_conf

        Feel free to replace with an actual LLM call or a learned reward model later.
        """
        # Clamp helper
        def clamp(x, lo, hi): return max(lo, min(hi, x))

        if case == 2:
            # false positive: higher conf => larger penalty
            penalty = -2.0 - 3.0 * ((deberta_conf + bert_conf) / 2.0 - 0.5)
            return clamp(penalty, self.min_reward, 0.0)
        elif case == 4:
            # correct non-spoiler: modest reward increases with bert_conf
            reward = 1.5 + 3.0 * (bert_conf - 0.5)
            return clamp(reward, 0.0, self.max_reward)
        elif case == 6:
            # filtered non-spoiler: small reward if low conf, small penalty if high conf
            base = 0.5 - 1.5 * (deberta_conf - 0.5)
            return clamp(base, -2.0, 2.0)
        else:
            # Should not be used for spoiler cases; fallback safe
            return 0.0


class UltraFastSophisticatedRewardCalculator:
    """
    Sophisticated multi-component reward system adapted for movie dataset.
    Base reward sourcing:
      - Cases 1,3,5 (GT spoiler): use internal base_rewards table (confident cases)
      - Cases 2,4,6 (GT non-spoiler): use external JudgeModel to get base reward
    All other components remain unchanged.
    """

    def __init__(self, label_dict: Optional[Dict[str, int]] = None, judge: Optional[JudgeModel] = None):
        """
        label_dict: optional mapping of sentence_text -> label (0 or 1)
        judge: optional JudgeModel instance; if None, a default is instantiated
        """
        self.label_dict = label_dict if label_dict else {}
        self.judge = judge if judge is not None else JudgeModel()

        # Core hyperparameters (unchanged)
        self.pos_weight_factor = 2.0
        self.pos_weight_exp = 1.5
        self.conf_alpha = 1.8
        self.info_gamma = 0.6
        self.cons_epsilon = 0.8
        self.adapt_beta = 0.5
        self.balance_lambda = 0.3

        # Enhanced base rewards (unchanged values)
        # IMPORTANT: used ONLY for cases 1,3,5 (GT spoiler)
        self.base_rewards = {
            1: 8.0,   # TP (spoiler)
            2: -3.0,  # FP (non-spoiler) -> now sourced from judge model
            3: -5.0,  # FN (spoiler)
            4: 3.0,   # TN (non-spoiler) -> now sourced from judge model
            5: -10.0, # Filtered FN (spoiler)
            6: 1.0    # Filtered TN (non-spoiler) -> now sourced from judge model
        }

        # Spoiler heuristics and regex (kept for fallback)
        self.spoiler_keywords = {
            'dies', 'death', 'died', 'dead', 'killed', 'murder',
            'suicide', 'ending', 'reveals', 'turns', 'secret',
            'truth', 'real', 'villain', 'killer', 'betrayal', 'surprise'
        }

        self.spoiler_patterns = [
            r'\b(dies?|death|killed) (at|in) (the )?end',
            r'turns? out (that|to be)?',
            r'(real|true|actual) (killer|villain)',
            r'(ending|conclusion) (was|is|reveals)',
            r'(finally|eventually) (dies?|kills?|reveals?)'
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.spoiler_patterns]

        # Tracking buffers (unchanged)
        self.case_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.prediction_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)

        # Class balance tracking
        self.spoiler_count = 0
        self.non_spoiler_count = 0
        self.filter_count = 0

    def get_fast_spoiler_label(self, sentence: str) -> int:
        """
        If label_dict has a label, use it. Otherwise fallback on keyword/pattern heuristics.
        """
        if not sentence or len(sentence.strip()) < 3:
            return 0

        if self.label_dict and sentence in self.label_dict:
            return int(self.label_dict[sentence])

        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        keyword_matches = sum(1 for word in words if word in self.spoiler_keywords)
        if keyword_matches >= 2:
            return 1
        for pattern in self.compiled_patterns:
            if pattern.search(sentence_lower):
                return 1
        return 0

    # ============== unchanged helper components ==============

    def calculate_position_weight(self, sentence_pos: int, total_sentences: int) -> float:
        if total_sentences <= 1:
            return 1.0
        normalized_pos = sentence_pos / total_sentences
        return 1 + self.pos_weight_factor * (normalized_pos ** self.pos_weight_exp)

    def _determine_case(self, deberta_action: int, bert_pred: int, true_label: int) -> int:
        """
        Original case definition, but split out to allow custom base reward sourcing.
        """
        if deberta_action == 1:  # Content
            if bert_pred == true_label:
                case = 1 if true_label == 1 else 4
            else:
                case = 3 if true_label == 1 else 2
        else:
            case = 5 if true_label == 1 else 6
        return case

    def _get_base_reward(self, case: int, true_label: int, sentence: str, deberta_conf: float, bert_conf: float) -> float:
        """
        New logic:
          - If ground truth is spoiler (true_label==1), use internal table (cases 1,3,5).
          - Else (true_label==0), use judge model to decide the base reward (cases 2,4,6).
        """
        if true_label == 1:
            # Spoiler GT: cases 1,3,5 use internal base table
            return self.base_rewards[case]
        else:
            # Non-spoiler GT: cases 2,4,6 use judge model
            return float(self.judge.score(sentence=sentence, case=case, deberta_conf=deberta_conf, bert_conf=bert_conf))

    def calculate_confidence_multiplier(self, case: int, deberta_conf: float, bert_conf: float) -> float:
        if case in [1, 2, 3, 4]:
            correctness_sign = 1 if case in [1, 4] else -1
            conf_mul = 1 + self.conf_alpha * (deberta_conf - 0.5) * (bert_conf - 0.5) * correctness_sign
        else:
            beta = 1.2 if case == 6 else -0.8
            conf_mul = 1 + beta * (deberta_conf - 0.5) ** 2
        return max(0.1, conf_mul)

    def calculate_information_gain(self, case: int, bert_entropy_before: float, bert_entropy_after: float, R_base: float) -> float:
        if case not in [1, 2, 3, 4] or bert_entropy_before <= 0:
            return 0.0
        info_gain = (bert_entropy_before - bert_entropy_after) / bert_entropy_before
        R_info = self.info_gamma * info_gain * abs(R_base)
        return np.clip(R_info, -1.0, 2.0)

    def calculate_adaptive_reward(self, case: int, deberta_conf: float) -> float:
        if not self.case_history:
            return 0.0
        case_perf = [1 if c == case else 0 for c in self.case_history]
        improvement_bonus = 0.0
        if len(case_perf) >= 10:
            recent_success = sum(case_perf[-10:]) / 10
            overall = sum(case_perf) / len(case_perf)
            if overall < 0.6 and recent_success > overall:
                improvement_bonus = self.adapt_beta * (recent_success - overall)
        conf_adj = self.adapt_beta * (deberta_conf - 0.5)
        if case not in [1, 4, 6]:
            conf_adj = -conf_adj
        return np.clip(improvement_bonus + conf_adj, -1.0, 1.5)

    def calculate_class_balance_reward(self, case: int, deberta_action: int) -> float:
        total_samples = self.spoiler_count + self.non_spoiler_count + self.filter_count + 1
        if total_samples < 10:
            return 0.0
        spoiler_ratio = self.spoiler_count / total_samples
        non_spoiler_ratio = self.non_spoiler_count / total_samples
        filter_ratio = self.filter_count / total_samples
        ideal_filter, ideal_spoiler, ideal_non = 0.15, 0.10, 0.75
        if case == 6:
            deviation = abs(filter_ratio - ideal_filter)
            R_balance = self.balance_lambda * (1 - deviation) if filter_ratio < ideal_filter else -self.balance_lambda * deviation
        elif case in [1, 3, 5]:
            deviation = abs(spoiler_ratio - ideal_spoiler)
            R_balance = -self.balance_lambda * deviation * 0.5
        else:
            deviation = abs(non_spoiler_ratio - ideal_non)
            R_balance = -self.balance_lambda * deviation * 0.3
        return np.clip(R_balance, -0.5, 1.0)

    def calculate_consistency_reward(self, case: int, deberta_action: int, deberta_conf: float) -> float:
        if len(self.prediction_history) < 3:
            return 0.0
        recent_actions = list(self.prediction_history)[-3:]
        pattern_consistency = 0.3 if len(set(recent_actions)) == 1 else (0.1 if len(set(recent_actions)) == 2 else -0.2)
        if len(self.confidence_history) >= 2:
            conf_var = np.var(list(self.confidence_history)[-2:] + [deberta_conf])
            conf_consistency = -0.2 * conf_var
        else:
            conf_consistency = 0.0
        R_cons = self.cons_epsilon * (pattern_consistency + conf_consistency)
        if case in [1, 4, 6]:
            R_cons *= 1.2
        return np.clip(R_cons, -1.0, 1.0)

    def update_tracking(self, case: int, deberta_action: int, deberta_conf: float, total_reward: float):
        self.case_history.append(case)
        self.reward_history.append(total_reward)
        self.action_history.append(deberta_action)
        self.prediction_history.append(deberta_action)
        self.confidence_history.append(deberta_conf)
        if case in [1, 3, 5]:
            self.spoiler_count += 1
        elif case in [2, 4]:
            self.non_spoiler_count += 1
        elif case == 6:
            self.filter_count += 1

    # ================== MAIN REWARD ==================

    def calculate_total_reward(
        self,
        deberta_action: int,
        deberta_conf: float,
        bert_pred: int,
        bert_conf: float,
        bert_entropy_before: float,
        bert_entropy_after: float,
        true_label: int,
        gwenn_reward: float,
        sentence_pos: int,
        total_sentences: int,
        sentence_text: str = ""
    ) -> Dict:
        """
        New behavior:
          - Determine case as before
          - Compute base reward:
              * true_label==1 -> internal table (cases 1,3,5)
              * true_label==0 -> judge model (cases 2,4,6)
          - Other components unchanged
        """
        pos_wei = self.calculate_position_weight(sentence_pos, total_sentences)
        case = self._determine_case(deberta_action, bert_pred, true_label)
        R_base = self._get_base_reward(case, true_label, sentence_text, deberta_conf, bert_conf)

        conf_mul = self.calculate_confidence_multiplier(case, deberta_conf, bert_conf)
        R_info = self.calculate_information_gain(case, bert_entropy_before, bert_entropy_after, R_base)
        R_adapt = self.calculate_adaptive_reward(case, deberta_conf)
        R_balance = self.calculate_class_balance_reward(case, deberta_action)
        R_cons = self.calculate_consistency_reward(case, deberta_action, deberta_conf)

        total_reward = gwenn_reward + pos_wei * (R_base * conf_mul + R_info + R_adapt + R_balance + R_cons)
        self.update_tracking(case, deberta_action, deberta_conf, total_reward)

        return {
            "total_reward": float(np.clip(total_reward, -50.0, 50.0)),
            "case": case,
            "base_reward": float(R_base),
            "gwenn_reward": float(gwenn_reward),
            "position_weight": float(pos_wei),
            "confidence_multiplier": float(conf_mul),
            "info_gain": float(R_info),
            "adaptive_reward": float(R_adapt),
            "balance_reward": float(R_balance),
            "consistency_reward": float(R_cons)
        }


def calculate_batch_rewards(
    reward_calculator: UltraFastSophisticatedRewardCalculator,
    deberta_actions: List[int],
    deberta_confs: List[float],
    bert_preds: List[int],
    bert_confs: List[float],
    bert_entropies_before: List[float],
    bert_entropies_after: List[float],
    true_labels: List[int],
    gwenn_rewards: List[float],
    sentences: List[str]
) -> List[Dict]:
    batch_results = []
    for i in range(len(sentences)):
        result = reward_calculator.calculate_total_reward(
            deberta_action=deberta_actions[i],
            deberta_conf=deberta_confs[i],
            bert_pred=bert_preds[i],
            bert_conf=bert_confs[i],
            bert_entropy_before=bert_entropies_before[i],
            bert_entropy_after=bert_entropies_after[i],
            true_label=true_labels[i],
            gwenn_reward=gwenn_rewards[i],
            sentence_pos=i + 1,
            total_sentences=len(sentences),
            sentence_text=sentences[i]
        )
        batch_results.append(result)
    return batch_results


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)
