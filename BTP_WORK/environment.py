"""
Deep Reinforcement Learning Environment for Book Review Content Classification
Ultra-High Accuracy with Fast Precomputed Embeddings
Optimized for Maximum Speed

Author: Research Project  
Python Version: 3.12.11
System: Linux (SSH Compatible)
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque, defaultdict
import random
from tqdm.auto import tqdm
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import sys
sys.path.append(str(Path(__file__).parent))
from config import config
sys.path.append(str(Path(__file__).parent / "utils"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_loader import AdvancedDataLoader
try:
    from utils.similarity_engine import FastSemanticSimilarityEngine
except ImportError:
    logger.warning("FastSemanticSimilarityEngine not found, using fallback")
    FastSemanticSimilarityEngine = None


# Setup logging


class BookReviewContentEnvironment(gym.Env):
    """
    Optimized RL Environment for Book Review Content Classification
    State: sentence + description + genre embeddings + precomputed features
    Action: binary classification
    Rewards: sophisticated, precomputed similarity + features
    """

    def __init__(self, dataset_or_loader, metadata: Optional[Dict] = None, split: str = 'train'):
        """
        dataset_or_loader: either a torch DataLoader, a custom BookReviewDataset (dataset),
                           or a plain list of samples. This constructor will normalize to a dataset-like
                           object that supports __len__ and __getitem__.
        metadata: optional metadata dict
        split: 'train' / 'validation' / 'test' label (for saving)
        """
        super().__init__()

        self.split = split
        self.device = config.DEVICE
        self.config = config
        self.data_loader = dataset_or_loader if hasattr(dataset_or_loader, "batch_size") else None



        # Normalize the incoming object to a dataset object
        # Accept: DataLoader (has attribute .dataset), Dataset (has __len__), or a plain list
        if hasattr(dataset_or_loader, "dataset"):
            # DataLoader -> get underlying dataset
            dataset = dataset_or_loader.dataset
        else:
            dataset = dataset_or_loader

        # Accept lists of samples as well (wrap them in a simple dataset view)
        if isinstance(dataset, list):
            class _ListDataset:
                def __init__(self, samples): self.samples = samples
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx): return self.samples[idx]
            dataset = _ListDataset(dataset)

        # Basic validation: must be sized and subscriptable
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("BookReviewContentEnvironment requires a dataset-like object "
                            "(DataLoader.dataset, Dataset, or list). Received: "
                            f"{type(dataset)}")

        self.dataset = dataset
        self.metadata = metadata or {}

        # Precompute content/non-content indices for faster sampling
        # Note: cast to list() to avoid generators
        self.content_indices = [i for i, s in enumerate(self.dataset) if s.get('label', 0) == 1]
        self.non_content_indices = [i for i, s in enumerate(self.dataset) if s.get('label', 0) == 0]

        # Define action and observation spaces
        state_dim = config.EnvironmentConfig.STATE_DIMENSION + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.EnvironmentConfig.ACTION_SPACE_SIZE)

        # FAST SIMILARITY ENGINE
        logger.info("🚀 Initializing FastSemanticSimilarityEngine...")
        if FastSemanticSimilarityEngine is not None:
            logger.info("🚀 Initializing FastSemanticSimilarityEngine...")
            self.similarity_engine = FastSemanticSimilarityEngine(
                embeddings_dir=config.PROJECT_ROOT / "data" / "embeddings",
                fallback_to_compute=True
            )
            logger.info("✅ FastSemanticSimilarityEngine initialized successfully")
        else:
            logger.warning("⚠️ FastSemanticSimilarityEngine not available, using fallback")
            self.similarity_engine = None


        self.sample_indices = list(range(len(self.dataset)))

        # Environment state defaults
        self.current_sample_idx = 0
        self.current_sample = None
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_accuracies = []

        # Performance tracking
        self.total_episodes = 0
        self.total_correct_predictions = 0
        self.total_predictions = 0
        self.reward_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.content_correct = 0
        self.content_total = 0
        self.non_content_correct = 0
        self.non_content_total = 0
        self.genre_performance = defaultdict(lambda: {'correct': 0, 'total': 0})

        self.dynamic_thresholds = {
            'high': config.EnvironmentConfig.SIMILARITY_THRESHOLD_HIGH,
            'low': config.EnvironmentConfig.SIMILARITY_THRESHOLD_LOW
        }

        # Guard: dataset must have at least 1 element
        ds_len = len(self.dataset)
        if ds_len == 0:
            raise ValueError("Provided dataset is empty. Cannot initialize environment.")
        self.max_steps_per_episode = min(config.DQNConfig.MAX_STEPS_PER_EPISODE, ds_len)

        logger.info(f"✅ Environment initialized: {len(self.dataset)} samples")
    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'overall_accuracy': self.get_overall_accuracy(),
            'content_accuracy': self.get_content_accuracy(),
            'non_content_accuracy': self.get_non_content_accuracy(),
            'total_episodes': self.total_episodes,
            'total_predictions': self.total_predictions,
            'dataset_size': len(self.dataset)
        }
    # ------------------- RESET -------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.episode_step = 0
        self.episode_rewards = []
        self.episode_accuracies = []

        self.current_sample_idx = self._intelligent_sample_selection()
        self.current_sample = self.dataset[self.current_sample_idx]

        observation = self._get_state_representation()
        info = self._create_info_dict()

        self.total_episodes += 1
        return observation, info

    # ------------------- STEP -------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        true_label = self.current_sample['label']
        print(f"🔍 Features type: {type(self.current_sample['features'])}")
        print(f"🔍 Features keys: {list(self.current_sample['features'].keys())}")
        print(f"🔍 Features values: {list(self.current_sample['features'].values())}")
        # Use precomputed similarity stored in features[0]
        similarity_score = self.current_sample['features'].get('cosine_similarity', 0.0)

        reward = self._calculate_advanced_reward(action, true_label, similarity_score)
        self._update_performance_metrics(action, true_label)

        self.episode_step += 1
        terminated = self._check_termination_conditions()
        truncated = self.episode_step >= self.max_steps_per_episode

        if not terminated and not truncated:
            self.current_sample_idx = self._intelligent_sample_selection()
            self.current_sample = self.dataset[self.current_sample_idx]

        next_observation = self._get_state_representation()
        info = self._create_step_info_dict(action, true_label, reward, similarity_score)

        self.episode_rewards.append(reward)
        self.episode_accuracies.append(1.0 if action == true_label else 0.0)

        return next_observation, reward, terminated, truncated, info

    # ------------------- SAMPLING -------------------
    def _intelligent_sample_selection(self) -> int:
        if self.total_episodes < 1000:
            return self._balanced_random_sampling()
        elif self.total_episodes < 3000:
            return self._curriculum_sampling()
        else:
            return self._hard_example_sampling()

    def _balanced_random_sampling(self) -> int:
        if random.random() < 0.5 and self.content_indices:
            return random.choice(self.content_indices)
        elif self.non_content_indices:
            return random.choice(self.non_content_indices)
        return random.randint(0, len(self.dataset) - 1)

    def _curriculum_sampling(self) -> int:
        # ✅ FIXED VERSION - Properly calls __getitem__ to get dictionaries
        confident_samples = []
        uncertain_samples = []
        
        for i in range(min(1000, len(self.dataset))):
            sample = self.dataset[i]  # This calls your __getitem__ method properly
            confidence = sample.get('confidence', 0.0)
            
            # Handle tensor confidence values
            if torch.is_tensor(confidence):
                confidence = confidence.item()
                
            if confidence > 0.7:
                confident_samples.append(i)
            else:
                uncertain_samples.append(i)
        
        # Rest of your logic stays exactly the same
        difficulty_ratio = min(self.total_episodes / 2000, 0.8)
        if random.random() < (1 - difficulty_ratio) and confident_samples:
            return random.choice(confident_samples)
        elif uncertain_samples:
            return random.choice(uncertain_samples)
        else:
            return random.randint(0, len(self.dataset) - 1)

    def _hard_example_sampling(self) -> int:
        """
        Focus on hard examples with low confidence or high uncertainty
        ✅ FIXED VERSION - Properly calls __getitem__ to get dictionaries
        """
        hard_samples = []
        sample_range = min(500, len(self.dataset))
        
        for i in range(sample_range):
            try:
                # Call __getitem__ properly to get the dictionary
                sample = self.dataset[i]  # This calls your __getitem__ method
                
                if isinstance(sample, dict) and 'confidence' in sample:
                    confidence_value = sample['confidence']
                    # Handle tensor confidence values
                    if torch.is_tensor(confidence_value):
                        confidence_value = confidence_value.item()
                    
                    # Low confidence = hard example
                    if confidence_value < 0.5:
                        hard_samples.append(i)
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Return a random hard sample, or fallback to random sampling
        if hard_samples:
            return np.random.choice(hard_samples)
        else:
            return np.random.randint(0, len(self.dataset))


    # ------------------- STATE -------------------
    def _get_state_representation(self) -> torch.Tensor:
        if self.current_sample is None:
            return torch.zeros(self.observation_space.shape[0], dtype=torch.float32, device=self.device)

        # Ensure all tensors are on the same device
        # ✅ FIXED VERSION:
        state = self.current_sample['state'].to(self.device).view(-1)

        additional_features = self.current_sample['features_tensor'].to(self.device)
        
        full_state = torch.cat([state, additional_features], dim=0)  
        return full_state


    # ------------------- REWARD -------------------
    def _calculate_advanced_reward(self, action: int, true_label: int, similarity_score: float) -> float:
        if action == true_label:
            base_reward = config.EnvironmentConfig.REWARD_CONTENT_CORRECT if true_label == 1 else config.EnvironmentConfig.REWARD_NON_CONTENT_CORRECT
        else:
            base_reward = config.EnvironmentConfig.PENALTY_WRONG_CLASSIFICATION

        additional_reward = 0.0
        confidence = self.current_sample['confidence']

        if action == true_label:
            additional_reward += confidence * 2.0
        else:
            additional_reward -= confidence * 3.0

        if action == true_label:
            if true_label == 1 and similarity_score > 0.7:
                additional_reward += 3.0
            elif true_label == 0 and similarity_score < 0.3:
                additional_reward += 2.0

        features = self.current_sample['features']
        genre_relevance = features['genre_relevance']
        plot_keywords = features['plot_keywords']
        character_mentions = features['character_mentions']

        if action == true_label and genre_relevance > 0.5:
            additional_reward += config.EnvironmentConfig.GENRE_MATCH_BONUS
        if action == 1 and true_label == 1 and plot_keywords > 0.5:
            additional_reward += 2.0

        ambiguity = 1 - abs(similarity_score - 0.5) * 2
        if action == true_label and ambiguity > 0.6:
            additional_reward += 5.0

        if self._should_adapt_thresholds():
            additional_reward += self._adaptive_threshold_reward(action, true_label)

        total_reward = base_reward + additional_reward
        return torch.clamp(total_reward, min=-50.0, max=50.0)

    # ------------------- METRICS -------------------
    def _update_performance_metrics(self, action: int, true_label: int):
        self.total_predictions += 1
        if action == true_label:
            self.total_correct_predictions += 1
        if true_label == 1:
            self.content_total += 1
            if action == true_label:
                self.content_correct += 1
        else:
            self.non_content_total += 1
            if action == true_label:
                self.non_content_correct += 1

        self.accuracy_history.append(1.0 if action == true_label else 0.0)

        genres = self.current_sample.get('genres', [])
        for genre in genres:
            self.genre_performance[genre]['total'] += 1
            if action == true_label:
                self.genre_performance[genre]['correct'] += 1

    def _check_termination_conditions(self) -> bool:
        if self.episode_step >= self.max_steps_per_episode:
            return True
        if len(self.episode_accuracies) >= 10:
            recent_accuracy = np.mean(self.episode_accuracies[-10:])
            if recent_accuracy >= 0.95 or (recent_accuracy <= 0.3 and len(self.episode_accuracies) >= 20):
                return True
        return False

    # ------------------- INFO DICTS -------------------
    def _create_info_dict(self) -> Dict[str, Any]:
        return {
            'episode': self.total_episodes,
            'dataset_size': len(self.dataset),
            'sample_index': self.current_sample_idx,
            'ground_truth': self.current_sample['label'],
            'confidence': self.current_sample['confidence'],
            'overall_accuracy': self.get_overall_accuracy(),
            'content_accuracy': self.get_content_accuracy(),
            'non_content_accuracy': self.get_non_content_accuracy()
        }

    def _create_step_info_dict(self, action: int, true_label: int, reward: float, similarity_score: float) -> Dict[str, Any]:
        return {
            'action': action,
            'true_label': true_label,
            'reward': reward,
            'correct': action == true_label,
            'episode_step': self.episode_step,
            'similarity_score': similarity_score,
            'confidence': self.current_sample['confidence'],
            'sample_index': self.current_sample_idx,
            'episode_accuracy': np.mean(self.episode_accuracies) if self.episode_accuracies else 0.0,
            'overall_accuracy': self.get_overall_accuracy(),
            'recent_accuracy': np.mean(list(self.accuracy_history)) if self.accuracy_history else 0.0
        }

    # ------------------- ACCURACY -------------------
    def get_overall_accuracy(self) -> float:
        return self.total_correct_predictions / max(self.total_predictions, 1)

    def get_content_accuracy(self) -> float:
        return self.content_correct / max(self.content_total, 1)

    def get_non_content_accuracy(self) -> float:
        return self.non_content_correct / max(self.non_content_total, 1)

    # ------------------- THRESHOLD ADAPT -------------------
    def _should_adapt_thresholds(self) -> bool:
        return self.total_predictions > 500 and self.total_predictions % 100 == 0

    def _adaptive_threshold_reward(self, action: int, true_label: int) -> float:
        recent_accuracy = sum(self.accuracy_history) / max(len(self.accuracy_history), 1)
        if recent_accuracy < 0.7 and action == 0:
            return 1.0
        if recent_accuracy > 0.9 and action == 1 and true_label == 1:
            return 2.0
        return 0.0

    # ------------------- RENDER & CLOSE -------------------
    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            print(f"Episode {self.total_episodes}, Step {self.episode_step}, Accuracy {self.get_overall_accuracy():.3f}")
            print(f"Sample {self.current_sample_idx}: {self.current_sample.get('sentence', '')[:100]}... Label {self.current_sample['label']}")
        return None

    def close(self):
        performance_data = {
            'overall_accuracy': self.get_overall_accuracy(),
            'content_accuracy': self.get_content_accuracy(),
            'non_content_accuracy': self.get_non_content_accuracy(),
            'total_episodes': self.total_episodes,
            'total_predictions': self.total_predictions,
            'genre_performance': dict(self.genre_performance),
            'fast_similarity_enabled': (
    self.similarity_engine.precomputed_loaded 
    if self.similarity_engine is not None 
    else False
)
        }
        save_path = config.RESULTS_DIR / f"environment_performance_{self.split}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(performance_data, f)
        logger.info(f"Environment performance saved to {save_path}")
        super().close()
# ------------------- ENVIRONMENT FACTORY -------------------
def create_environment(config=None, data_loader=None, dataloaders=None,
                       dataset_or_loader=None, metadata=None, split=None):
    """
    Factory function to create the BookReviewContentEnvironment.

    Accepts dataset in multiple forms for maximum flexibility:
      - data_loader: single PyTorch DataLoader
      - dataloaders: dict with 'train', 'validation', 'test'
      - dataset_or_loader: generic (Dataset, DataLoader, or dict of them)

    Also accepts:
      - config: environment configuration (optional, defaults to Config.EnvironmentConfig)
      - metadata: optional dictionary with dataset metadata
      - split: which split to use ('train', 'validation', 'test')
    """

    # ✅ Ensure config is available
    if config is None:
        try:
            from config import Config
            config = Config.EnvironmentConfig()
            logger.warning("⚠️ No config provided. Using default Config.EnvironmentConfig.")
        except Exception as e:
            logger.error(f"❌ Could not initialize EnvironmentConfig from config.py: {e}")
            config = {}

    # 🔄 Normalize dataset argument
    if dataset_or_loader is not None:
        if isinstance(dataset_or_loader, dict):
            dataloaders = dataset_or_loader
        else:
            data_loader = dataset_or_loader

    # 🔎 Handle split selection
    if split is not None:
        if dataloaders is not None:
            if split not in dataloaders:
                raise ValueError(
                    f"❌ Split '{split}' not found in dataloaders. "
                    f"Available splits: {list(dataloaders.keys())}"
                )
            data_loader = dataloaders[split]
            logger.info(f"📂 Using '{split}' split with {len(data_loader.dataset)} samples.")
        else:
            logger.warning("⚠️ 'split' provided but no dataloaders dict found. Ignoring split.")

    # 🚨 Safety check: must have data
    if data_loader is None and dataloaders is None:
        raise ValueError(
            "❌ No dataset provided to create_environment(). "
            "Pass either `data_loader`, `dataloaders`, or `dataset_or_loader`."
        )

    # ✅ Create environment
    env = BookReviewContentEnvironment(dataset_or_loader=data_loader, metadata=metadata, split=split or "train")


    # 📑 Attach metadata if provided
    if metadata is not None:
        env.metadata = metadata
        logger.info("📑 Metadata attached to environment.")

    logger.info("✅ Environment created successfully with provided dataset(s).")
    return env


# ------------------- ENVIRONMENT EVALUATION -------------------
def evaluate_environment(env: BookReviewContentEnvironment, num_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluate the environment for a number of episodes and return average metrics.
    """
    total_reward = 0.0
    total_accuracy = 0.0

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_accuracy_list = []

        while not done and not truncated:
            # For testing, take a random action
            action = env.action_space.sample()
            observation, reward, done, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_accuracy_list.append(1.0 if action == step_info['true_label'] else 0.0)

        total_reward += episode_reward
        total_accuracy += np.mean(episode_accuracy_list) if episode_accuracy_list else 0.0

    avg_reward = total_reward / num_episodes
    avg_accuracy = total_accuracy / num_episodes

    return {
        'average_reward': avg_reward,
        'average_accuracy': avg_accuracy
    }


if __name__ == "__main__":
    # Test the environment
    logger.info("🧪 Testing BookReviewContentEnvironment...")
    
    # Create environment
    
    data_loader = AdvancedDataLoader(config)
    env = create_environment(data_loader=data_loader, split="train")

    # Run evaluation
    results = evaluate_environment(env, num_episodes=10)
    
    # Display results
    print(f"\n🎯 Test Results:")
    print(f"Average Accuracy: {results['average_accuracy']:.4f}")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Fast Similarity: {'✅ Enabled' if env.similarity_engine.precomputed_loaded else '❌ Disabled'}")
    
    env.close()
    logger.info("✅ Environment test completed!")
