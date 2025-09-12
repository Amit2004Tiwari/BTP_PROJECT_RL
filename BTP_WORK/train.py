"""
Advanced Deep RL Training Pipeline for Book Review Content Classification
Ultra-High Accuracy Training System with Complete Progress Bar Suppression

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
Engineered for Maximum Classification Accuracy and Research Excellence
"""

import os
import sys
import warnings

# ============================================================================
# COMPLETE TQDM & PROGRESS BAR SUPPRESSION - MUST BE FIRST
# ============================================================================
# Disable ALL tqdm progress bars from all sources

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import and disable transformers progress bars immediately
try:
    import transformers.utils.logging as hf_logging
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
except:
    pass

# Import and patch sentence-transformers to disable progress bars
try:
    import sentence_transformers
    # Monkey patch to disable all sentence-transformers progress bars
    original_tqdm = sentence_transformers.util.tqdm_lib.tqdm
    def disabled_tqdm(*args, **kwargs):
        kwargs['disable'] = True
        kwargs['leave'] = False
        return original_tqdm(*args, **kwargs)
    sentence_transformers.util.tqdm_lib.tqdm = disabled_tqdm
except:
    pass

# Disable all warnings
#warnings.filterwarnings('ignore')

# Standard imports
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import pickle
import matplotlib
matplotlib.use('Agg')
# Core libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
import tqdm
from tqdm import trange
# Import project components
from config import config, validate_config
sys.path.append(str(Path(__file__).parent / "utils"))
sys.path.append(str(Path(__file__).parent / "models"))

from data_loader import AdvancedDataLoader, initialize_data_loader
from environment import BookReviewContentEnvironment, create_environment
from dqn_agent import AdvancedDQNAgent, create_dqn_agent

# Setup comprehensive logging with clean output
class CleanFormatter(logging.Formatter):
    """Custom formatter that removes verbose logging"""
    def format(self, record):
        # Skip batch-related messages
        if 'batch' in record.getMessage().lower() or 'loading' in record.getMessage().lower():
            return ""
        return super().format(record)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    """
    Comprehensive Deep RL Training Pipeline with Clean Output
    Orchestrates all components for maximum accuracy training
    """
    
    def __init__(self, experiment_name: str = None):
        """Initialize the advanced training pipeline"""
        self.experiment_name = experiment_name or f"book_review_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        
        # Create experiment directory
        self.experiment_dir = config.RESULTS_DIR / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.environments = {}
        self.agent = None
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'accuracies': [],
            'losses': [],
            'learning_rates': [],
            'epsilons': [],
            'training_times': []
        }
        
        # Training state
        self.current_episode = 0
        self.best_performance = {
            'accuracy': 0.0,
            'reward': float('-inf'),
            'episode': 0
        }
        
        # Performance tracking
        self.performance_history = []
        self.evaluation_history = []
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.training_interrupted = False
        
        print(f"🚀 Advanced Training Pipeline initialized")
        print(f"📁 Experiment: {self.experiment_name}")
        print(f"💾 Results directory: {self.experiment_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on interrupt"""
        print(f"⚠️  Received signal {signum}, initiating graceful shutdown...")
        self.training_interrupted = True
    
    def setup_components(self) -> bool:
        """
        Initialize and validate all training components:
        - Data loaders
        - RL environments
        - RL agent
        - Monitoring and logging
        """

        print("⚙️  Setting up training components...")
        
        try:
            # -----------------------------
            # 1️⃣ Validate configuration
            # -----------------------------
            validate_config()
            print("✅ Configuration validated")
            
            # -----------------------------
            # 2️⃣ Initialize data loader
            # -----------------------------
            print("📊 Initializing advanced data loader...")

            self.data_loader = initialize_data_loader(config.DATASET_PATH)
            dataloaders, metadata = self.data_loader.get_dataloaders()
            self.dataloaders = dataloaders
            self.dataset_metadata = metadata

            print(f"✅ Dataset loaded: {metadata['total_samples']} samples")
            print(f"📈 Content ratio: {metadata['content_ratio']:.3f}")
            print("DEBUG: dataloaders type:", type(self.dataloaders))
            print("DEBUG: dataloaders keys:", list(self.dataloaders.keys()))
            
            # -----------------------------
            # 3️⃣ Create RL environments
            # -----------------------------
            print("🌍 Creating RL environments...")
            print("🔍 Debug - Checking dataloaders:")
            for split_name, dataloader in self.dataloaders.items():
                print(f"  {split_name}: type={type(dataloader)}")
                if hasattr(dataloader, 'dataset'):
                    dataset = dataloader.dataset
                    print(f"    Dataset: type={type(dataset)}, len={len(dataset)}")
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f"    First sample: type={type(sample)}, keys={sample.keys() if isinstance(sample, dict) else 'N/A'}")
                else:
                    print(f"    No dataset attribute found")
            self.environments = {}
            for split in ['train', 'validation', 'test']:
                env = create_environment(
                    config=config,                    # ✅ Pass config
                    dataloaders=self.dataloaders,     # ✅ Pass entire dataloaders dict
                    metadata=self.dataset_metadata, 
                    split=split                       # ✅ Specify which split to use
                )
                self.environments[split] = env
                print(f"✅ {split.title()} environment created with {len(env.dataset)} samples")

            # -----------------------------
            # 4️⃣ Initialize DQN agent
            # -----------------------------
            print("🤖 Creating advanced DQN agent...")
            state_dim = config.EnvironmentConfig.STATE_DIMENSION + 7
            action_dim = config.EnvironmentConfig.ACTION_SPACE_SIZE
            self.agent = create_dqn_agent(state_dim, action_dim)
            summary = self.agent.get_performance_summary()
            print(f"✅ Agent created: {metadata['total_samples']:,} dataset size, "
                f"{state_dim}-dim state, {action_dim}-dim action space")

            # -----------------------------
            # 5️⃣ Setup monitoring/logging
            # -----------------------------
            self._setup_monitoring()
            
            return True
        
        except Exception as e:
            print(f"❌ Component setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False



    
    def _setup_monitoring(self):
        """Setup comprehensive training monitoring"""
        # Create monitoring directories
        self.plots_dir = self.experiment_dir / "plots"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        
        for directory in [self.plots_dir, self.checkpoints_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Save configuration
        config_dict = {
            'experiment_name': self.experiment_name,
            'dataset_metadata': self.dataset_metadata,
            'agent_config': self.agent.get_performance_summary(),
            'environment_config': {
                'state_dimension': config.EnvironmentConfig.STATE_DIMENSION,
                'action_space_size': config.EnvironmentConfig.ACTION_SPACE_SIZE,
                'reward_thresholds': {
                    'high': config.EnvironmentConfig.SIMILARITY_THRESHOLD_HIGH,
                    'low': config.EnvironmentConfig.SIMILARITY_THRESHOLD_LOW
                }
            }
        }
        
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Initialize progress CSV file
        progress_file = self.experiment_dir / "training_progress.csv"
        with open(progress_file, 'w') as f:
            f.write("episode,accuracy,reward,loss,epsilon,learning_rate,duration,timestamp\n")
        
        print("✅ Monitoring setup completed")
    
    def save_progress_to_csv(self, episode: int, metrics: Dict[str, float]):
        """Save episode progress to CSV file"""
        progress_file = self.experiment_dir / "training_progress.csv"
        with open(progress_file, 'a') as f:
            f.write(f"{episode},{metrics['accuracy']:.6f},{metrics['reward']:.4f},"
                   f"{metrics['loss']:.6f},{metrics['epsilon']:.6f},"
                   f"{metrics['learning_rate']:.8f},{metrics['duration']:.2f},"
                   f"{datetime.now().isoformat()}\n")
    
    def print_detailed_progress_report(self, episode: int, max_episodes: int, episode_metrics: Dict[str, float]):
        """Print comprehensive progress report"""
        elapsed_time = time.time() - self.start_time
        episodes_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0
        remaining_episodes = max_episodes - episode
        eta_hours = remaining_episodes / episodes_per_hour if episodes_per_hour > 0 else 0
        
        # Calculate recent averages
        recent_accuracy = np.mean([acc.cpu().item() if torch.is_tensor(acc) else acc 
                                for acc in self.training_metrics['accuracies'][-10:]]) if len(self.training_metrics['accuracies']) >= 10 else episode_metrics['accuracy']
        recent_reward = np.mean([reward.cpu().item() if torch.is_tensor(reward) else reward 
                                for reward in self.training_metrics['rewards'][-10:]]) if len(self.training_metrics['rewards']) >= 10 else episode_metrics['reward']
        recent_loss = np.mean([loss.cpu().item() if torch.is_tensor(loss) else loss 
                            for loss in self.training_metrics['losses'][-10:]]) if len(self.training_metrics['losses']) >= 10 else episode_metrics['loss']
        
        
        # Performance trend
        if len(self.training_metrics['accuracies']) >= 20:
            trend_recent = np.mean(self.training_metrics['accuracies'][-10:])
            trend_older = np.mean(self.training_metrics['accuracies'][-20:-10])
            trend_direction = "📈 Improving" if trend_recent > trend_older else "📉 Declining" if trend_recent < trend_older else "➡️  Stable"
        else:
            trend_direction = "📊 Learning"
        
        print(f"\n{'='*90}")
        print(f"🎯 EPISODE {episode:4d}/{max_episodes} - DETAILED PROGRESS REPORT")
        print(f"{'='*90}")
        print(f"⏱️  Training Time: {elapsed_time/3600:.1f} hours elapsed | ETA: {eta_hours:.1f} hours remaining")
        print(f"🚀 Training Speed: {episodes_per_hour:.1f} episodes/hour | {episode_metrics['duration']:.1f}s per episode")
        print(f"")
        print(f"📊 CURRENT EPISODE METRICS:")
        print(f"   Accuracy: {episode_metrics['accuracy']:.4f} | Reward: {episode_metrics['reward']:8.2f} | Loss: {episode_metrics['loss']:.6f}")
        print(f"")
        print(f"📈 RECENT 10-EPISODE AVERAGES:")
        print(f"   Accuracy: {recent_accuracy:.4f} | Reward: {recent_reward:8.2f} | Loss: {recent_loss:.6f}")
        print(f"")
        print(f"🏆 BEST PERFORMANCE:")
        print(f"   Best Accuracy: {self.best_performance['accuracy']:.4f} (Episode {self.best_performance['episode']})")
        print(f"")
        print(f"🔧 AGENT PARAMETERS:")
        print(f"   Exploration (ε): {episode_metrics['epsilon']:.6f} | Learning Rate: {episode_metrics['learning_rate']:.8f}")
        print(f"")
        print(f"📊 PERFORMANCE TREND: {trend_direction}")
        print(f"{'='*90}\n")
    
    def train_episode(self, env: BookReviewContentEnvironment, episode_num: int) -> Dict[str, float]:
        """Execute single training episode with comprehensive metrics"""
        episode_start_time = time.time()
        
        # Reset environment
        state, info = env.reset()
        
        # Episode tracking
        episode_reward = 0.0
        episode_loss = 0.0
        episode_steps = 0
        episode_accuracy = []
        step_losses = []
        
        done = False
        
        # Suppress all output during episode execution
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            while not done and not self.training_interrupted:
                # Agent selects action
                action, action_info = self.agent.select_action(state, training=True)
                
                # Execute action in environment
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent (if enough experiences)
                if len(self.agent.memory) >= config.DQNConfig.BATCH_SIZE:
                    training_metrics = self.agent.train()
                    episode_loss += training_metrics['loss']
                    step_losses.append(training_metrics['loss'])
                
                # Update tracking
                episode_reward += reward
                episode_steps += 1
                episode_accuracy.append(step_info['correct'])
                
                # Update state
                state = next_state
        
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start_time
        episode_metrics = {
            'episode': episode_num,
            'reward': episode_reward,
            'accuracy': np.mean([acc.cpu().item() if torch.is_tensor(acc) else acc 
                     for acc in episode_accuracy]) if episode_accuracy else 0.0,
            'loss': episode_loss / max(episode_steps, 1),
            'steps': episode_steps,
            'duration': episode_duration,
            'epsilon': self.agent.epsilon,
            'learning_rate': self.agent.optimizer.param_groups[0]['lr'],
            'final_performance': env.get_performance_summary()
        }
        
        return episode_metrics
    
    def evaluate_agent(self, episode_num: int, detailed: bool = True) -> Dict[str, Any]:
        """Comprehensive agent evaluation across all environments"""
        print(f"\n🧪 Evaluating agent at episode {episode_num}...")
        evaluation_results = {'episode': episode_num, 'timestamp': datetime.now().isoformat()}
        
        # Suppress output during evaluation
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Evaluate on each environment split
            for split_name, env in self.environments.items():
                if split_name == 'train' and not detailed:
                    continue  # Skip training environment for quick evaluations
                    
                eval_episodes = 20 if detailed else 5
                split_results = self.agent.evaluate(env, num_episodes=eval_episodes)
                evaluation_results[split_name] = split_results
        
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        
        # Print results
        for split_name in evaluation_results:
            if isinstance(evaluation_results[split_name], dict) and 'average_accuracy' in evaluation_results[split_name]:
                split_results = evaluation_results[split_name]
                print(f"   📊 {split_name.title()} Results: Accuracy {split_results['average_accuracy']:.4f} ± {split_results['std_accuracy']:.4f}")
        
        # Check for best performance
        val_accuracy = evaluation_results.get('validation', {}).get('average_accuracy', 0.0)
        if val_accuracy > self.best_performance['accuracy']:
            improvement = val_accuracy - self.best_performance['accuracy']
            self.best_performance = {
                'accuracy': val_accuracy,
                'reward': evaluation_results.get('validation', {}).get('average_reward', 0.0),
                'episode': episode_num
            }
            
            # Save best model
            best_model_path = self.checkpoints_dir / "best_model.pth"
            self.agent.save_checkpoint(best_model_path, metadata={
                'performance': self.best_performance,
                'evaluation_results': evaluation_results
            })
            print(f"   💎 NEW BEST MODEL SAVED! Accuracy: {val_accuracy:.4f} (↑{improvement:.4f})")
        
        self.evaluation_history.append(evaluation_results)
        print(f"🧪 Evaluation completed.\n")
        return evaluation_results
    
    def save_training_progress(self):
        """Save comprehensive training progress"""
        progress_data = {
            'experiment_name': self.experiment_name,
            'current_episode': self.current_episode,
            'training_metrics': self.training_metrics,
            'best_performance': self.best_performance,
            'evaluation_history': self.evaluation_history,
            'agent_summary': self.agent.get_performance_summary(),
            'training_duration': time.time() - self.start_time
        }
        
        # Save as JSON
        with open(self.experiment_dir / "training_progress.json", 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
        
        # Save as pickle for detailed analysis
        with open(self.experiment_dir / "training_progress.pkl", 'wb') as f:
            pickle.dump(progress_data, f)
    
    def create_training_visualizations(self):
        """Create comprehensive training visualizations with complete CUDA safety"""
        if not self.training_metrics['episodes']:
            return
        
        # ✅ SAFE CONVERSION UTILITY - Handles all tensor types
        def safe_convert(data):
            """Safely convert tensors to CPU scalars/arrays for plotting"""
            if isinstance(data, list):
                return [x.cpu().item() if torch.is_tensor(x) else x for x in data]
            elif torch.is_tensor(data):
                return data.cpu().numpy()
            else:
                return data
        
        # ✅ CONVERT ALL DATA UPFRONT - No more CUDA errors!
        episodes = safe_convert(self.training_metrics['episodes'])
        rewards = safe_convert(self.training_metrics['rewards'])
        accuracies = safe_convert(self.training_metrics['accuracies'])
        losses = safe_convert(self.training_metrics['losses'])
        learning_rates = safe_convert(self.training_metrics['learning_rates'])
        epsilons = safe_convert(self.training_metrics['epsilons'])
        training_times = safe_convert(self.training_metrics['training_times'])
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
        
        # 1. Reward progression
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Rewards')
        if len(rewards) >= 10:
            axes[0, 0].plot(episodes, 
                        pd.Series(rewards).rolling(10).mean(), 'r-', linewidth=2, label='10-Episode Moving Average')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy progression
        axes[0, 1].plot(episodes, accuracies, 'g-', alpha=0.7, label='Episode Accuracy')
        if len(accuracies) >= 10:
            axes[0, 1].plot(episodes,
                        pd.Series(accuracies).rolling(10).mean(), 'r-', linewidth=2, label='10-Episode Moving Average')
        axes[0, 1].set_title('Episode Accuracy')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss progression
        if len(losses) > 0:
            axes[0, 2].plot(episodes, losses, 'm-', alpha=0.7, label='Episode Loss')
            if len(losses) >= 10:
                axes[0, 2].plot(episodes,
                            pd.Series(losses).rolling(10).mean(), 'r-', linewidth=2, label='10-Episode Moving Average')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning rate schedule
        axes[1, 0].plot(episodes, learning_rates, 'c-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Epsilon decay
        axes[1, 1].plot(episodes, epsilons, 'orange')
        axes[1, 1].set_title('Exploration (Epsilon) Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Training time per episode
        if len(training_times) > 0:
            axes[1, 2].plot(episodes, training_times, 'brown')
            axes[1, 2].set_title('Training Time per Episode')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create evaluation plot if we have evaluation history
        if self.evaluation_history:
            self._plot_evaluation_results()

    
    def _plot_evaluation_results(self):
        """Create detailed evaluation visualizations"""
        if not self.evaluation_history:
            return
            
        episodes = [eval_data['episode'] for eval_data in self.evaluation_history]
        
        # Extract metrics for each split
        splits = ['validation', 'test']
        metrics = ['average_accuracy', 'average_reward']
        
        fig, axes = plt.subplots(len(metrics), len(splits), figsize=(15, 10))
        if len(metrics) == 1:
            axes = axes.reshape(1, -1)
        if len(splits) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(metrics):
            for j, split in enumerate(splits):
                values = []
                errors = []
                
                for eval_data in self.evaluation_history:
                    if split in eval_data:
                        values.append(eval_data[split].get(metric, 0))
                        if metric == 'average_accuracy':
                            errors.append(eval_data[split].get('std_accuracy', 0))
                        else:
                            errors.append(eval_data[split].get('std_reward', 0))
                    else:
                        values.append(0)
                        errors.append(0)
                
                axes[i, j].errorbar(episodes, values, yerr=errors, fmt='o-', capsize=5)
                axes[i, j].set_title(f'{split.title()} {metric.replace("_", " ").title()}')
                axes[i, j].set_xlabel('Episode')
                axes[i, j].set_ylabel(metric.replace('_', ' ').title())
                axes[i, j].grid(True, alpha=0.3)
        
        plt.suptitle('Evaluation Results Over Training')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_training(self, max_episodes: int = None) -> Dict[str, Any]:
        """Execute the complete training pipeline with clean, informative output"""
        max_episodes = max_episodes or config.DQNConfig.TRAINING_EPISODES
        
        print(f"\n🚀 Starting training for {max_episodes} episodes...")
        print(f"🎯 Target: Maximum accuracy book review content classification")
        print(f"📊 Progress will be displayed every episode with detailed reports every 10 episodes")
        print(f"💾 Results will be saved to: {self.experiment_dir}")
        print("="*90)
        
        # Setup components
        if not self.setup_components():
            print("❌ Failed to setup components")
            return {'success': False}
        
        # Training environment
        train_env = self.environments['train']
        
        try:
            for episode in trange(1, max_episodes + 1, desc="Training Episodes"):
                if self.training_interrupted:
                    print("⚠️  Training interrupted by user")
                    break
                
                self.current_episode = episode
                episode_start = time.time()
                
                # Execute training episode
                episode_metrics = self.train_episode(train_env, episode)
                episode_duration = time.time() - episode_start
                
                # Store metrics
                self.training_metrics['episodes'].append(episode)
                self.training_metrics['rewards'].append(episode_metrics['reward'])
                self.training_metrics['accuracies'].append(episode_metrics['accuracy'])
                self.training_metrics['losses'].append(episode_metrics['loss'])
                self.training_metrics['learning_rates'].append(episode_metrics['learning_rate'])
                self.training_metrics['epsilons'].append(episode_metrics['epsilon'])
                self.training_metrics['training_times'].append(episode_duration)
                
                # Save progress to CSV
                self.save_progress_to_csv(episode, episode_metrics)
                
                # Print episode progress
                print(f"Episode {episode:4d}: Acc={episode_metrics['accuracy']:.4f} | "
                      f"Reward={episode_metrics['reward']:7.2f} | "
                      f"Loss={episode_metrics['loss']:.6f} | "
                      f"Best={self.best_performance['accuracy']:.4f} | "
                      f"ε={episode_metrics['epsilon']:.4f} | "
                      f"Time={episode_duration:.1f}s")
                
                # Detailed progress report every 10 episodes
                if episode % 10 == 0:
                    self.print_detailed_progress_report(episode, max_episodes, episode_metrics)
                
                # Periodic evaluation and checkpointing
                if episode % config.TrainingConfig.VALIDATION_FREQUENCY == 0:
                    evaluation_results = self.evaluate_agent(episode, detailed=True)
                
                # Save checkpoint
                if episode % config.TrainingConfig.SAVE_CHECKPOINT_FREQUENCY == 0:
                    checkpoint_path = self.checkpoints_dir / f"checkpoint_episode_{episode}.pth"
                    self.agent.save_checkpoint(checkpoint_path)
                    self.save_training_progress()
                    self.create_training_visualizations()
                    print(f"💾 Checkpoint saved: checkpoint_episode_{episode}.pth")
                
                # Early stopping based on performance
                if len(self.training_metrics['accuracies']) >= 50:
                    recent_50_acc = np.mean(self.training_metrics['accuracies'][-50:])
                    if recent_50_acc >= 0.98:  # Very high accuracy achieved
                        print(f"\n🎯 Excellent accuracy achieved ({recent_50_acc:.4f})!")
                        if episode >= max_episodes * 0.5:  # At least half training completed
                            print("✅ Training completed with excellent performance!")
                            break
        
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        finally:
            # Final evaluation and cleanup
            print("\n🏁 Performing final evaluation...")
            final_evaluation = self.evaluate_agent(self.current_episode, detailed=True)
            
            # Save final results
            self.save_training_progress()
            self.create_training_visualizations()
            
            # Final checkpoint
            final_checkpoint = self.checkpoints_dir / "final_model.pth"
            self.agent.save_checkpoint(final_checkpoint, metadata={
                'final_evaluation': final_evaluation,
                'training_completed': True
            })
            print(f"💾 Final model saved: {final_checkpoint}")
        
        # Compile final results
        training_duration = time.time() - self.start_time
        final_results = {
            'success': True,
            'experiment_name': self.experiment_name,
            'episodes_completed': self.current_episode,
            'training_duration': training_duration,
            'best_performance': self.best_performance,
            'final_evaluation': final_evaluation,
            'agent_summary': self.agent.get_performance_summary(),
            'dataset_stats': self.dataset_metadata
        }
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Duration: {timedelta(seconds=int(training_duration))}")
        print(f"🎯 Best Accuracy: {self.best_performance['accuracy']:.4f}")
        print(f"📁 Results saved in: {self.experiment_dir}")
        print("="*90)
        
        return final_results
    
    def cleanup(self):
        """Clean up resources"""
        for env in self.environments.values():
            if hasattr(env, 'close'):
                env.close()
        print("🧹 Cleanup completed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*90)
    print("🚀 DEEP REINFORCEMENT LEARNING BOOK REVIEW CLASSIFIER")
    print("🎯 High-Accuracy Content Detection Training Pipeline")
    print("📚 Research Project - Maximum Performance Configuration")
    print("="*90)
    
    # Check environment
    print(f"🔧 Python Version: {sys.version}")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print(f"🔧 Device: {config.DEVICE}")
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Validate dataset
    if not config.DATASET_PATH.exists():
        print(f"❌ Dataset not found: {config.DATASET_PATH}")
        print("Please ensure your book_reviews.csv file is in the data/ directory")
        return False
    
    print(f"✅ Dataset found: {config.DATASET_PATH}")
    
    # Create training pipeline
    experiment_name = f"book_review_rl_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedTrainingPipeline(experiment_name)
    
    try:
        # Run training
        results = pipeline.run_training()
        
        if results['success']:
            print("\n🎊 TRAINING COMPLETED SUCCESSFULLY! 🎊")
            print("="*90)
            print("📊 FINAL RESULTS SUMMARY:")
            print(f"🎯 Best Accuracy: {results['best_performance']['accuracy']:.4f}")
            print(f"📈 Episodes Completed: {results['episodes_completed']}")
            print(f"⏱️  Training Duration: {timedelta(seconds=int(results['training_duration']))}")
            print(f"🧠 Model Parameters: {results['agent_summary']['network_parameters']:,}")
            print(f"📁 Results Directory: {pipeline.experiment_dir}")
            print("="*90)
            print("🚀 Your research-grade RL model is ready for publication!")
            return True
        else:
            print("❌ Training failed")
            return False
            
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
