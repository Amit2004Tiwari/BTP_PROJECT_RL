"""
Deep Reinforcement Learning Classifier for Book Review Content Detection
Configuration File - All hyperparameters and settings

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
"""

import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
class Config:
    EMBEDDING_DIMENSION = 384
    """
    Comprehensive configuration class for Deep RL Book Review Classifier
    Optimized for maximum accuracy while maintaining high performance
    """
    
    # ============================================================================
    # PROJECT PATHS & DIRECTORIES
    # ============================================================================
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    LOGS_DIR = LOGS_DIR
    RESULTS_DIR = RESULTS_DIR
    CHECKPOINTS_DIR = CHECKPOINTS_DIR
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Data file path
    DATASET_PATH = DATA_DIR / "book_reviews_clean.csv"
    
    # ============================================================================
    # DEVICE & PERFORMANCE SETTINGS
    # ============================================================================
    # Automatic device detection with fallback
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Performance optimization settings
    NUM_WORKERS = min(8, os.cpu_count())  # For DataLoader
    PIN_MEMORY = True if torch.cuda.is_available() else False
    PERSISTENT_WORKERS = True
    
    # Memory optimization
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = True  # For faster training with minimal accuracy loss
    
    # ============================================================================
    # TEXT ENCODING & PREPROCESSING SETTINGS
    # ============================================================================
    # Primary encoder for semantic similarity (highest accuracy)
    PRIMARY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Backup encoders for ensemble (if needed)
    BACKUP_ENCODERS = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Faster alternative
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]
    
    # Text preprocessing settings
    MAX_SENTENCE_LENGTH = 512  # Optimized for BERT-based models
    MIN_SENTENCE_LENGTH = 5    # Filter out very short sentences
    SENTENCE_OVERLAP_THRESHOLD = 0.85  # Remove near-duplicate sentences
    
    # Encoding settings
    EMBEDDING_DIMENSION = 384  # For all-mpnet-base-v2
    BATCH_SIZE_ENCODING = 32   # Optimized for memory efficiency
    
    # ============================================================================
    # DEEP RL AGENT CONFIGURATION
    # ============================================================================
    # DQN Agent Settings
    class DQNConfig:
        # Network architecture
        HIDDEN_LAYERS = [1024, 512, 256, 128]  # Deep network for high accuracy
        DROPOUT_RATE = 0.2
        ACTIVATION = "relu"
        
        # Training hyperparameters
        LEARNING_RATE = 1e-4
        GAMMA = 0.95  # Discount factor
        EPSILON_START = 1.0
        EPSILON_END = 0.01
        EPSILON_DECAY = 0.995
        
        # Experience replay
        MEMORY_SIZE = 100000
        BATCH_SIZE = 64
        TARGET_UPDATE_FREQUENCY = 1000
        
        # Training schedule
        TRAINING_EPISODES = 5000
        MAX_STEPS_PER_EPISODE = 100
        
        # Performance optimization
        DOUBLE_DQN = True
        DUELING_DQN = True
        PRIORITIZED_REPLAY = True
    
    # ============================================================================
    # ENVIRONMENT CONFIGURATION
    # ============================================================================
    class EnvironmentConfig:
        # State representation
        EMBEDDING_DIMENSION = 384
        STATE_DIMENSION = EMBEDDING_DIMENSION * 3  # review + description + genre
        ACTION_SPACE_SIZE = 2  # content vs non-content
        
        # Reward function parameters
        SIMILARITY_THRESHOLD_HIGH = 0.45    # High similarity = content
        SIMILARITY_THRESHOLD_LOW = 0.3     # Low similarity = non-content
        REWARD_CONTENT_CORRECT = 10.0      # Correct content classification
        REWARD_NON_CONTENT_CORRECT = 5.0   # Correct non-content classification
        PENALTY_WRONG_CLASSIFICATION = -10.0
        REWARD_UNCERTAINTY_PENALTY = -1.0   # For ambiguous cases
        
        # Genre bonus rewards
        GENRE_MATCH_BONUS = 2.0
        FICTION_CONTENT_BONUS = 1.5
        NON_FICTION_CONTENT_BONUS = 2.0
    
    # ============================================================================
    # DATA PROCESSING CONFIGURATION
    # ============================================================================
    class DataConfig:
        # Dataset splitting
        TRAIN_SPLIT = 0.7
        VALIDATION_SPLIT = 0.15
        TEST_SPLIT = 0.15
        
        # Data augmentation
        ENABLE_DATA_AUGMENTATION = True
        AUGMENTATION_PROBABILITY = 0.3
        
        # Similarity calculation
        COSINE_SIMILARITY_THRESHOLD = 0.5
        SEMANTIC_SEARCH_TOP_K = 5
        
        # Genre processing
        GENRE_EMBEDDING_SIZE = 64
        MAX_GENRES_PER_BOOK = 5
        
        # Review processing
        MAX_SENTENCES_PER_REVIEW = 20
        MIN_REVIEWS_PER_BOOK = 1
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    class TrainingConfig:
        # General training settings
        MAX_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        LEARNING_RATE_SCHEDULER = "cosine"
        WARMUP_STEPS = 1000
        
        # Validation settings
        VALIDATION_FREQUENCY = 100  # Every N episodes
        SAVE_BEST_MODEL = True
        SAVE_CHECKPOINT_FREQUENCY = 500
        
        # Logging settings
        LOG_FREQUENCY = 50
        TENSORBOARD_LOGGING = True
        WANDB_LOGGING = False  # Set to True if using Weights & Biases
        
        # Performance monitoring
        TRACK_SIMILARITY_SCORES = True
        TRACK_REWARD_DISTRIBUTION = True
        TRACK_ACTION_DISTRIBUTION = True
    
    # ============================================================================
    # EVALUATION CONFIGURATION
    # ============================================================================
    class EvaluationConfig:
        # Metrics to calculate
        CALCULATE_PRECISION_RECALL = True
        CALCULATE_F1_SCORE = True
        CALCULATE_CONFUSION_MATRIX = True
        CALCULATE_ROC_AUC = True
        
        # Evaluation settings
        EVALUATION_BATCH_SIZE = 128
        SAVE_PREDICTIONS = True
        SAVE_ATTENTION_WEIGHTS = True
        
        # Threshold optimization
        OPTIMIZE_THRESHOLD = True
        THRESHOLD_SEARCH_RANGE = (0.1, 0.9)
        THRESHOLD_SEARCH_STEPS = 50
    
    # ============================================================================
    # REPRODUCIBILITY SETTINGS
    # ============================================================================
    RANDOM_SEED = 42
    TORCH_DETERMINISTIC = True
    CUDNN_BENCHMARK = False  # Set to False for reproducibility
    
    # ============================================================================
    # LOGGING & MONITORING
    # ============================================================================
    class LoggingConfig:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # File logging
        ENABLE_FILE_LOGGING = True
        LOG_FILE = LOGS_DIR / "training.log"
        MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
        BACKUP_COUNT = 5
        
        # Progress tracking
        ENABLE_TQDM = True
        TQDM_DISABLE_ON_SSH = False  # Keep enabled for SSH sessions
        
        # Metrics tracking
        TRACK_GPU_MEMORY = True
        TRACK_TRAINING_TIME = True
        SAVE_TRAINING_CURVES = True

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================
config = Config()

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    """
    Validate configuration settings and environment
    """
    # Check if dataset exists
    if not config.DATASET_PATH.exists():
        print(f"Warning: Dataset not found at {config.DATASET_PATH}")
    
    # Check CUDA availability
    if config.DEVICE.type == "cuda":
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU - consider using GPU for faster training")
    
    # Validate hyperparameters
    assert config.DQNConfig.EPSILON_START >= config.DQNConfig.EPSILON_END
    assert config.DataConfig.TRAIN_SPLIT + config.DataConfig.VALIDATION_SPLIT + config.DataConfig.TEST_SPLIT == 1.0
    assert config.EnvironmentConfig.SIMILARITY_THRESHOLD_HIGH > config.EnvironmentConfig.SIMILARITY_THRESHOLD_LOW
    
    print("✅ Configuration validation passed")

if __name__ == "__main__":
    validate_config()
