"""
Advanced Deep Q-Network Agent for Book Review Content Classification
Ultra-High Accuracy RL Agent with State-of-the-Art Features

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
Engineered for Maximum Classification Accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pickle
from pathlib import Path
import math
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.monitor_interval = 0

# Import configuration and networks
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from networks import create_dqn_network, AdvancedDQNNetwork

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class PrioritizedReplayBuffer:
    """
    Advanced Prioritized Experience Replay Buffer
    Implements importance sampling for maximum learning efficiency
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """Initialize prioritized replay buffer"""
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Storage
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        # For efficient sampling
        self.max_priority = 1.0
        
        logger.info(f"✅ PrioritizedReplayBuffer initialized: capacity={capacity}")
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, priority: float = None):
        """Add experience to buffer with priority"""
        if priority is None:
            priority = self.max_priority
        
        experience = Experience(state, action, reward, next_state, done, priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
            self.position = (self.position + 1) % self.capacity
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} experiences, need {batch_size}")
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance weights
        total_size = len(self.buffer)
        weights = (total_size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Extract experiences
        experiences = [self.buffer[i] for i in indices]
        
        # ✅ FIXED VERSION - Use torch.tensor() with explicit dtype:
        states = torch.stack([torch.tensor(e.state, dtype=torch.float32) if not torch.is_tensor(e.state) else e.state for e in experiences]).to(config.DEVICE)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(config.DEVICE)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(config.DEVICE)
        next_states = torch.stack([torch.tensor(e.next_state, dtype=torch.float32) if not torch.is_tensor(e.next_state) else e.next_state for e in experiences]).to(config.DEVICE)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool).to(config.DEVICE)
        
        # ✅ CRITICAL FIX - Convert weights to tensor PROPERLY:
        weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)

        
        # Update beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority + 1e-6)
    
    def __len__(self):
        return len(self.buffer)

class AdvancedDQNAgent:
    """
    Ultra-Advanced Deep Q-Network Agent
    Implements cutting-edge RL techniques for maximum accuracy
    """
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = None):
        """Initialize the most advanced DQN agent"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate or config.DQNConfig.LEARNING_RATE
        self.device = config.DEVICE
        
        # Network architecture
        self.q_network = AdvancedDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = AdvancedDQNNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Advanced optimizer with scheduling
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=self.learning_rate * 0.01
        )
        
        # Experience replay buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=config.DQNConfig.MEMORY_SIZE,
            alpha=0.6,
            beta=0.4
        )
        
        # Exploration parameters
        self.epsilon = config.DQNConfig.EPSILON_START
        self.epsilon_min = config.DQNConfig.EPSILON_END
        self.epsilon_decay = config.DQNConfig.EPSILON_DECAY
        
        # Training parameters
        self.batch_size = config.DQNConfig.BATCH_SIZE
        self.gamma = config.DQNConfig.GAMMA
        self.target_update_frequency = config.DQNConfig.TARGET_UPDATE_FREQUENCY
        
        # Performance tracking
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        
        # Advanced features
        self.use_double_dqn = config.DQNConfig.DOUBLE_DQN
        self.use_dueling = config.DQNConfig.DUELING_DQN
        self.use_prioritized_replay = config.DQNConfig.PRIORITIZED_REPLAY
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
        # Performance metrics
        self.best_accuracy = 0.0
        self.consecutive_improvements = 0
        
        logger.info(f"✅ AdvancedDQNAgent initialized")
        logger.info(f"🧠 Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        logger.info(f"⚙️  Features: Double DQN={self.use_double_dqn}, Dueling={self.use_dueling}")
    
    def safe_tensor_to_cpu(self, tensor_or_value):
        """Safely convert tensor to CPU scalar or return original value"""
        if torch.is_tensor(tensor_or_value):
            return tensor_or_value.detach().cpu().item()
        return float(tensor_or_value)
    
    def safe_tensor_list_to_numpy(self, tensor_list):
        """Safely convert list of tensors/values to numpy array"""
        if not tensor_list:
            return np.array([])
        
        converted_list = []
        for item in tensor_list:
            if torch.is_tensor(item):
                converted_list.append(item.detach().cpu().item())
            else:
                converted_list.append(float(item))
        
        return np.array(converted_list)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, Dict[str, float]]:
        """Advanced action selection with epsilon-greedy and uncertainty"""
        # ✅ FIXED VERSION:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Get Q-values and uncertainty
        self.q_network.eval()
        with torch.no_grad():
            if hasattr(self.q_network, 'forward') and len(state_tensor.shape) > 1:
                q_values, uncertainty = self.q_network.forward(state_tensor, return_uncertainty=True)
            else:
                q_values = self.q_network(state_tensor)
                uncertainty = torch.zeros(1, 1).to(self.device)
        
        # Action selection strategy
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            selection_type = "random"
        else:
            action = q_values.argmax().item()
            selection_type = "greedy"
        
        # Compute action probabilities for analysis
        action_probs = F.softmax(q_values, dim=-1).cpu().numpy()[0]
        
        action_info = {
            'q_values': q_values.cpu().numpy()[0],
            'action_probs': action_probs,
            'uncertainty': self.safe_tensor_to_cpu(uncertainty),
            'epsilon': self.epsilon,
            'selection_type': selection_type,
            'confidence': float(action_probs[action])
        }
        
        return action, action_info
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer with computed priority"""
        # Compute TD error for priority
        priority = self._compute_td_error(state, action, reward, next_state, done)
        
        # Store in buffer
        self.memory.add(state, action, reward, next_state, done, priority)
    
    def _compute_td_error(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> float:
        """Compute TD error for experience priority"""
        # ✅ FIXED VERSION:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        self.q_network.eval()
        with torch.no_grad():
            # Current Q-value
            current_q = self.q_network(state_tensor)[0][action]
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                if self.use_double_dqn:
                    next_q_main = self.q_network(next_state_tensor)
                    best_action = next_q_main.argmax(dim=1)[0]
                    next_q_target = self.target_network(next_state_tensor)
                    next_q_value = next_q_target[0][best_action]
                else:
                    next_q_value = self.target_network(next_state_tensor).max(1)[0]
                
                target_q = reward + self.gamma * next_q_value
            
            td_error = abs(self.safe_tensor_to_cpu(target_q - current_q))
        
        return td_error
    
    def train(self) -> Dict[str, float]:
        """Advanced training step with prioritized replay and importance sampling"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'td_error': 0.0, 'learning_rate': self.optimizer.param_groups[0]['lr']}
        
        self.q_network.train()
        
        # Sample batch from prioritized replay buffer
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        
        # Compute current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_q_main = self.q_network(next_states)
                next_actions = next_q_main.argmax(dim=1)
                next_q_target = self.target_network(next_states)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute TD errors for priority updates
        td_errors = torch.abs(target_q - current_q)
        
        # Compute loss with importance sampling weights
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities in replay buffer
        if self.use_prioritized_replay:
            new_priorities = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, new_priorities)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Update exploration
        self.update_epsilon()
        
        # Store metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        training_metrics = {
            'loss': loss_value,
            'td_error': td_errors.mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item()
        }
        
        return training_metrics
    
    def update_target_network(self):
        """Soft update of target network for stability"""
        tau = 0.005
        
        for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
    
    def update_epsilon(self):
        """Advanced epsilon decay with minimum threshold"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """✅ FULLY CORRECTED - Comprehensive evaluation of the agent"""
        logger.info(f"🧪 Evaluating agent over {num_episodes} episodes...")
        
        self.q_network.eval()
        episode_rewards = []
        episode_accuracies = []
        episode_lengths = []
        action_distribution = {0: 0, 1: 0}
        confidence_scores = []
        
        with torch.no_grad():
            for episode in tqdm(range(num_episodes), desc="Evaluation", leave=False):
                state, _ = env.reset()
                episode_reward = 0
                episode_accuracy = []
                episode_length = 0
                done = False
                
                while not done:
                    # Select action without exploration
                    action, action_info = self.select_action(state, training=False)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_accuracy.append(info['correct'])
                    episode_length += 1
                    action_distribution[action] += 1
                    confidence_scores.append(action_info['confidence'])
                    
                    state = next_state
                    done = terminated or truncated
                
                # ✅ SAFE CONVERSION: Handle rewards and accuracies properly
                episode_rewards.append(self.safe_tensor_to_cpu(episode_reward))
                
                # ✅ SAFE ACCURACY COMPUTATION:
                if episode_accuracy:
                    accuracy_values = [self.safe_tensor_to_cpu(acc) for acc in episode_accuracy]
                    episode_accuracies.append(np.mean(accuracy_values))
                else:
                    episode_accuracies.append(0.0)
                
                episode_lengths.append(episode_length)
        
        # ✅ SAFE NUMPY OPERATIONS - All tensors converted to CPU first:
        episode_rewards_np = self.safe_tensor_list_to_numpy(episode_rewards)
        episode_accuracies_np = self.safe_tensor_list_to_numpy(episode_accuracies)
        episode_lengths_np = self.safe_tensor_list_to_numpy(episode_lengths)
        confidence_scores_np = self.safe_tensor_list_to_numpy(confidence_scores)
        
        # Compute comprehensive metrics
        evaluation_metrics = {
            'average_reward': float(np.mean(episode_rewards_np)),
            'std_reward': float(np.std(episode_rewards_np)),
            'average_accuracy': float(np.mean(episode_accuracies_np)),
            'std_accuracy': float(np.std(episode_accuracies_np)),
            'average_episode_length': float(np.mean(episode_lengths_np)),
            'action_distribution': {k: v/sum(action_distribution.values()) for k, v in action_distribution.items()},
            'average_confidence': float(np.mean(confidence_scores_np)),
            'min_accuracy': float(np.min(episode_accuracies_np)) if len(episode_accuracies_np) > 0 else 0.0,
            'max_accuracy': float(np.max(episode_accuracies_np)) if len(episode_accuracies_np) > 0 else 0.0,
            'accuracy_above_90': float(np.sum(episode_accuracies_np >= 0.9) / len(episode_accuracies_np)) if len(episode_accuracies_np) > 0 else 0.0
        }
        
        # Update best accuracy
        current_accuracy = evaluation_metrics['average_accuracy']
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.consecutive_improvements += 1
        else:
            self.consecutive_improvements = 0
        
        evaluation_metrics['best_accuracy'] = self.best_accuracy
        evaluation_metrics['consecutive_improvements'] = self.consecutive_improvements
        
        # Store in history
        self.accuracy_history.append(current_accuracy)
        self.reward_history.append(evaluation_metrics['average_reward'])
        
        logger.info(f"✅ Evaluation completed:")
        logger.info(f"📊 Accuracy: {current_accuracy:.4f} ± {evaluation_metrics['std_accuracy']:.4f}")
        logger.info(f"🎯 Best Accuracy: {self.best_accuracy:.4f}")
        logger.info(f"💰 Average Reward: {evaluation_metrics['average_reward']:.2f}")
        
        return evaluation_metrics
    
    def save_checkpoint(self, filepath: Path, metadata: Dict[str, Any] = None):
        """Save comprehensive agent checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'best_accuracy': self.best_accuracy,
            'consecutive_improvements': self.consecutive_improvements,
            'loss_history': list(self.loss_history),
            'reward_history': list(self.reward_history),
            'accuracy_history': list(self.accuracy_history),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gamma': self.gamma
            }
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.best_accuracy = checkpoint['best_accuracy']
        self.consecutive_improvements = checkpoint['consecutive_improvements']
        
        # Load history
        self.loss_history.extend(checkpoint['loss_history'])
        self.reward_history.extend(checkpoint['reward_history'])
        self.accuracy_history.extend(checkpoint['accuracy_history'])
        
        logger.info(f"📁 Checkpoint loaded: {filepath}")
        logger.info(f"🎯 Loaded accuracy: {self.best_accuracy:.4f}")
        
        return checkpoint.get('metadata', {})
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'best_accuracy': self.best_accuracy,
            'recent_accuracy': np.mean(list(self.accuracy_history)[-10:]) if len(self.accuracy_history) >= 10 else 0.0,
            'recent_reward': np.mean(list(self.reward_history)[-10:]) if len(self.reward_history) >= 10 else 0.0,
            'recent_loss': np.mean(list(self.loss_history)[-100:]) if len(self.loss_history) >= 100 else 0.0,
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'memory_size': len(self.memory),
            'consecutive_improvements': self.consecutive_improvements,
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'device': str(self.device)
        }

# ============================================================================
# AGENT FACTORY AND UTILITIES
# ============================================================================

def create_dqn_agent(state_dim: int, action_dim: int, config_overrides: Dict[str, Any] = None) -> AdvancedDQNAgent:
    """Factory function to create advanced DQN agent"""
    logger.info(f"🏭 Creating AdvancedDQNAgent...")
    
    # Apply config overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config.DQNConfig, key):
                setattr(config.DQNConfig, key, value)
    
    agent = AdvancedDQNAgent(state_dim, action_dim)
    
    logger.info(f"✅ AdvancedDQNAgent created successfully")
    logger.info(f"🧠 State dim: {state_dim}, Action dim: {action_dim}")
    
    return agent

if __name__ == "__main__":
    # Test agent creation and basic functionality
    logger.info("🧪 Testing AdvancedDQNAgent...")
    
    # Test dimensions
    state_dim = config.EnvironmentConfig.STATE_DIMENSION + 7
    action_dim = config.EnvironmentConfig.ACTION_SPACE_SIZE
    
    # Create agent
    agent = create_dqn_agent(state_dim, action_dim)
    
    # Test action selection
    test_state = np.random.randn(state_dim)
    action, action_info = agent.select_action(test_state, training=True)
    
    logger.info(f"✅ Action selection test: action={action}, confidence={action_info['confidence']:.3f}")
    
    # Test experience storage
    next_state = np.random.randn(state_dim)
    agent.store_experience(test_state, action, 1.0, next_state, False)
    
    logger.info(f"✅ Experience storage test: buffer size={len(agent.memory)}")
    logger.info("🎯 All agent tests passed!")
