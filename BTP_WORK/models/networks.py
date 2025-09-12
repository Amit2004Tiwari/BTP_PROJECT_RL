"""
Advanced Neural Network Architectures for Deep RL Book Review Classification
High-Accuracy Deep Networks with Cutting-Edge Features

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
Optimized for Maximum Classification Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import logging
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAttentionMechanism(nn.Module):
    """
    Sophisticated Multi-Head Self-Attention for Review Content Understanding
    Optimized for semantic similarity and content detection
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize advanced attention mechanism
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Advanced projection layers with Xavier initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Advanced normalization and regularization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1) / math.sqrt(self.head_dim))
        
        # Initialize weights for maximum accuracy
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for optimal convergence"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with advanced attention computation
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            torch.Tensor: Attended features
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with learnable temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        # Advanced attention weights with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output

class AdvancedFeatureFusion(nn.Module):
    """
    Sophisticated feature fusion module for multi-modal inputs
    Combines sentence embeddings, description embeddings, and additional features
    """
    
    def __init__(self, embedding_dim: int, feature_dim: int, hidden_dim: int = 512):
        """
        Initialize feature fusion module
        
        Args:
            embedding_dim: Dimension of embeddings
            feature_dim: Dimension of additional features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Advanced fusion architecture
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),  # Changed from hidden_dim // 4 to hidden_dim
            nn.LayerNorm(hidden_dim),            # Changed from hidden_dim // 4 to hidden_dim
            nn.GELU(),
            nn.Dropout(0.1)
        )
        # Cross-modal attention for embedding-feature interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Now both are hidden_dim
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating mechanism for adaptive feature weighting
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for optimal performance"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Fuse embeddings and features with advanced techniques
        
        Args:
            embeddings: Embedding tensor [batch_size, embedding_dim]
            features: Additional features [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: Fused features
        """
        # Project embeddings and features
        projected_embeddings = self.embedding_projection(embeddings)
        projected_features = self.feature_projection(features)
        
        # Cross-modal attention (embeddings attend to features)
        attended_embeddings, _ = self.cross_attention(
            projected_embeddings.unsqueeze(1),
            projected_features.unsqueeze(1),
            projected_features.unsqueeze(1)
        )
        attended_embeddings = attended_embeddings.squeeze(1)
        
        # Concatenate for fusion
        fused = torch.cat([attended_embeddings, projected_features], dim=-1)
        
        # Apply fusion layers
        fused_features = self.fusion_layers(fused)
        
        # Gating mechanism for adaptive weighting
        gate_weights = self.gate(fused_features)
        gated_features = fused_features * gate_weights
        
        return gated_features

class DuelingDQNNetwork(nn.Module):
    """
    Advanced Dueling DQN Network with state-of-the-art architecture
    Separates value and advantage streams for superior performance
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = None):
        """
        Initialize Dueling DQN Network
        
        Args:
            state_dim: Input state dimension
            action_dim: Number of actions
            hidden_layers: Hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers or config.DQNConfig.HIDDEN_LAYERS
        
        # Feature extraction backbone
        self.feature_extractor = self._build_feature_extractor()
        
        # Dueling streams
        self.value_stream = self._build_value_stream()
        self.advantage_stream = self._build_advantage_stream()
        
        # Advanced normalization
        self.final_layer_norm = nn.LayerNorm(self.hidden_layers[-2])
        
        self._initialize_weights()
        
        logger.info(f"✅ DuelingDQNNetwork initialized: {state_dim} → {action_dim}")
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build sophisticated feature extraction layers"""
        layers = []
        input_dim = self.state_dim
        
        for i, hidden_dim in enumerate(self.hidden_layers[:-1]):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.DQNConfig.DROPOUT_RATE)
            ])
            
            # Add residual connection for deeper networks
            if i > 0 and input_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim))
            
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_value_stream(self) -> nn.Module:
        """Build value function stream"""
        return nn.Sequential(
            nn.Linear(self.hidden_layers[-2], self.hidden_layers[-1]),
            nn.LayerNorm(self.hidden_layers[-1]),
            nn.GELU(),
            nn.Dropout(config.DQNConfig.DROPOUT_RATE),
            nn.Linear(self.hidden_layers[-1], 1)  # Single value output
        )
    
    def _build_advantage_stream(self) -> nn.Module:
        """Build advantage function stream"""
        return nn.Sequential(
            nn.Linear(self.hidden_layers[-2], self.hidden_layers[-1]),
            nn.LayerNorm(self.hidden_layers[-1]),
            nn.GELU(),
            nn.Dropout(config.DQNConfig.DROPOUT_RATE),
            nn.Linear(self.hidden_layers[-1], self.action_dim)  # One per action
        )
    
    def _initialize_weights(self):
        """Advanced weight initialization for optimal performance"""
        def init_module(module):
            if isinstance(module, nn.Linear):
                # He initialization for GELU activation
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        self.apply(init_module)
        
        # Special initialization for final layers
        nn.init.xavier_uniform_(self.value_stream[-1].weight, gain=0.1)
        nn.init.xavier_uniform_(self.advantage_stream[-1].weight, gain=0.1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dueling DQN
        
        Args:
            state: Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Extract features
        features = self.feature_extractor(state)
        features = self.final_layer_norm(features)
        
        # Separate value and advantage streams
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim]
        
        # Combine streams with mean advantage subtraction for identifiability
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values

class ResidualBlock(nn.Module):
    """
    Residual block for deeper networks with better gradient flow
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        x = self.layers(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class AdvancedDQNNetwork(nn.Module):
    """
    Ultra-Advanced DQN Network combining all cutting-edge techniques
    Designed for maximum accuracy in book review content classification
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the most advanced DQN network
        
        Args:
            state_dim: Input state dimension
            action_dim: Number of actions (2 for binary classification)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Calculate embedding and feature dimensions
        self.embedding_dim = config.EMBEDDING_DIMENSION * 3  # sentence + description + genre
        self.feature_dim = 7  # Additional features
        self.hidden_dim = 512
        
        # Advanced feature fusion
        self.feature_fusion = AdvancedFeatureFusion(
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Multi-head attention for sequential processing
        self.attention = AdvancedAttentionMechanism(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Dueling DQN core
        self.dueling_network = DuelingDQNNetwork(
            state_dim=self.hidden_dim,
            action_dim=action_dim,
            hidden_layers=[512, 256, 128]
        )
        
        # Uncertainty estimation head for confidence-aware decisions
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
        # Ensemble components for improved accuracy
        self.ensemble_heads = nn.ModuleList([
            DuelingDQNNetwork(self.hidden_dim, action_dim, [256, 128, 64])
            for _ in range(3)
        ])
        
        # Final aggregation layer
        self.ensemble_aggregator = nn.Sequential(
            nn.Linear(action_dim * 4, action_dim * 2),  # 4 = main + 3 ensemble heads
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim * 2, action_dim)
        )
        
        self._initialize_advanced_weights()
        
        logger.info(f"✅ AdvancedDQNNetwork initialized with {self.count_parameters():,} parameters")
    
    def _initialize_advanced_weights(self):
        """Advanced weight initialization strategy"""
        def init_module(module):
            if isinstance(module, nn.Linear):
                # Use different initializations based on layer type
                if hasattr(module, 'weight'):
                    if module.out_features == self.action_dim:
                        # Final action layer: small weights for stability
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                    else:
                        # Hidden layers: He initialization
                        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        self.apply(init_module)
    
    def forward(self, state: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Advanced forward pass with ensemble and uncertainty estimation
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            torch.Tensor or Tuple: Q-values and optionally uncertainty
        """
        batch_size = state.size(0)
        
        # Split state into embeddings and features
        embeddings = state[:, :self.embedding_dim]
        features = state[:, self.embedding_dim:]
        
        # Advanced feature fusion
        fused_features = self.feature_fusion(embeddings, features)
        
        # Apply attention mechanism (add sequence dimension)
        attended_features = self.attention(fused_features.unsqueeze(1)).squeeze(1)
        
        # Main DQN prediction
        main_q_values = self.dueling_network(attended_features)
        
        # Ensemble predictions for improved accuracy
        ensemble_predictions = []
        for ensemble_head in self.ensemble_heads:
            ensemble_q = ensemble_head(attended_features)
            ensemble_predictions.append(ensemble_q)
        
        # Combine all predictions
        all_predictions = torch.cat([main_q_values] + ensemble_predictions, dim=-1)
        final_q_values = self.ensemble_aggregator(all_predictions)
        
        if return_uncertainty:
            # Estimate uncertainty
            uncertainty = self.uncertainty_head(attended_features)
            return final_q_values, uncertainty
        
        return final_q_values
    
    def get_action_probabilities(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities with temperature scaling
        
        Args:
            state: Input state
            temperature: Temperature for softmax scaling
            
        Returns:
            torch.Tensor: Action probabilities
        """
        q_values = self.forward(state)
        return F.softmax(q_values / temperature, dim=-1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_importance(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using integrated gradients
        
        Args:
            state: Input state
            
        Returns:
            torch.Tensor: Feature importance scores
        """
        state.requires_grad_(True)
        
        # Forward pass
        q_values = self.forward(state)
        
        # Get gradients for the maximum Q-value
        max_q = q_values.max(dim=-1)[0]
        gradients = torch.autograd.grad(max_q.sum(), state, create_graph=True)[0]
        
        # Feature importance as absolute gradients
        importance = torch.abs(gradients)
        
        return importance

class NoiseNetwork(nn.Module):
    """
    Noisy Networks for Exploration
    Implements parameter noise for better exploration in DQN
    """
    
    def __init__(self, input_dim: int, output_dim: int, std_init: float = 0.4):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.empty(output_dim))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_dim))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.output_dim))
    
    def reset_noise(self):
        """Reset noise for exploration"""
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

# ============================================================================
# NETWORK FACTORY FUNCTIONS
# ============================================================================

def create_dqn_network(state_dim: int, action_dim: int, network_type: str = 'advanced') -> nn.Module:
    """
    Factory function to create DQN networks with different configurations
    
    Args:
        state_dim: Input state dimension
        action_dim: Number of actions
        network_type: Type of network ('basic', 'dueling', 'advanced')
        
    Returns:
        nn.Module: Created network
    """
    logger.info(f"🏭 Creating {network_type} DQN network...")
    
    if network_type == 'basic':
        # Simple feedforward network
        hidden_layers = config.DQNConfig.HIDDEN_LAYERS
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.DQNConfig.DROPOUT_RATE)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        network = nn.Sequential(*layers)
        
    elif network_type == 'dueling':
        network = DuelingDQNNetwork(state_dim, action_dim)
        
    elif network_type == 'advanced':
        network = AdvancedDQNNetwork(state_dim, action_dim)
        
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Move to device
    network = network.to(config.DEVICE)
    
    # Count parameters
    param_count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logger.info(f"✅ {network_type.title()} DQN created: {param_count:,} parameters")
    
    return network

def initialize_network_weights(network: nn.Module, init_type: str = 'xavier') -> None:
    """
    Initialize network weights with specified strategy
    
    Args:
        network: Network to initialize
        init_type: Initialization strategy ('xavier', 'he', 'orthogonal')
    """
    def init_weights(module):
        if isinstance(module, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'he':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    network.apply(init_weights)
    logger.info(f"✅ Network weights initialized with {init_type} strategy")

if __name__ == "__main__":
    # Test network creation
    logger.info("🧪 Testing Advanced DQN Networks...")
    
    # Test dimensions based on config
    state_dim = config.EnvironmentConfig.STATE_DIMENSION + 7
    action_dim = config.EnvironmentConfig.ACTION_SPACE_SIZE
    
    # Create networks
    networks = {
        'basic': create_dqn_network(state_dim, action_dim, 'basic'),
        'dueling': create_dqn_network(state_dim, action_dim, 'dueling'),
        'advanced': create_dqn_network(state_dim, action_dim, 'advanced')
    }
    
    # Test forward passes
    test_input = torch.randn(32, state_dim).to(config.DEVICE)
    
    for name, network in networks.items():
        output = network(test_input)
        logger.info(f"✅ {name.title()} network test: {test_input.shape} → {output.shape}")
    
    logger.info("🎯 All network tests passed!")
