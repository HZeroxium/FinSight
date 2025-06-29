# models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List, Tuple, Union

from .interface import ModelInterface
from .components import PositionalEncoding, TransformerBlock
from ..core.config import Config


class FinancialTransformer(ModelInterface):
    """
    Advanced Transformer model specifically designed for financial time series prediction

    Features:
    - Multi-scale temporal attention
    - Financial-specific embeddings
    - Adaptive pooling strategies
    - Robust regularization
    """

    def __init__(self, config: Config):
        """
        Initialize financial transformer model

        Args:
            config: Model configuration containing all hyperparameters
        """
        super().__init__(config)

        # Extract configuration
        self.d_model = config.model.d_model
        self.n_heads = config.model.n_heads
        self.n_layers = config.model.n_layers
        self.d_ff = config.model.d_ff
        self.dropout = config.model.dropout
        self.input_dim = config.model.input_dim
        self.output_dim = config.model.output_dim
        self.sequence_length = config.model.sequence_length
        self.prediction_horizon = config.model.prediction_horizon

        # Advanced configuration
        self.use_relative_position = getattr(
            config.model, "use_relative_position", True
        )
        self.pre_norm = getattr(config.model, "pre_norm", True)
        self.activation = getattr(config.model, "activation", "gelu")
        self.pooling_strategy = getattr(config.model, "pooling_strategy", "adaptive")
        self.residual_scaling = getattr(config.model, "residual_scaling", 1.0)

        # Input projection and embedding
        self.input_projection = nn.Linear(self.input_dim, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_length=self.sequence_length * 2,  # Allow for longer sequences
            learnable=getattr(config.model, "learnable_pos_encoding", False),
            dropout=self.dropout,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    pre_norm=self.pre_norm,
                    use_relative_position=self.use_relative_position,
                    residual_scaling=self.residual_scaling,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Pooling and output layers
        self._setup_output_layers()

        # Initialize weights
        self._init_weights()

        self.logger.info(
            f"Initialized {self.get_model_name()} with {self.get_num_parameters():,} parameters"
        )

    def _setup_output_layers(self):
        """Setup output layers based on pooling strategy"""
        if self.pooling_strategy == "adaptive":
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
            output_input_dim = self.d_model
        elif self.pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.Tanh(),
                nn.Linear(self.d_model // 4, 1),
            )
            output_input_dim = self.d_model
        elif self.pooling_strategy == "last":
            output_input_dim = self.d_model
        elif self.pooling_strategy == "multi_scale":
            # Multi-scale pooling
            self.multi_scale_pools = nn.ModuleList(
                [nn.AdaptiveAvgPool1d(scale) for scale in [1, 2, 4]]
            )
            output_input_dim = self.d_model * 3
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.LayerNorm(output_input_dim),
            nn.Dropout(self.dropout),
            nn.Linear(output_input_dim, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim * self.prediction_horizon),
        )

    def _init_weights(self):
        """Initialize model weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for most layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask for autoregressive prediction

        Args:
            size: Sequence length
            device: Device to create mask on

        Returns:
            torch.Tensor: Causal mask of shape (size, size)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence based on configured strategy

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Pooled tensor
        """
        if self.pooling_strategy == "adaptive":
            # Global adaptive average pooling
            x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            x = self.adaptive_pool(x)  # (batch_size, d_model, 1)
            return x.squeeze(-1)  # (batch_size, d_model)

        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            attention_weights = self.attention_pool(x)  # (batch_size, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            return torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)

        elif self.pooling_strategy == "last":
            # Use last timestep
            return x[:, -1, :]  # (batch_size, d_model)

        elif self.pooling_strategy == "multi_scale":
            # Multi-scale pooling
            x_transposed = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            pooled_features = []

            for pool in self.multi_scale_pools:
                pooled = pool(x_transposed)  # (batch_size, d_model, scale)
                pooled = pooled.mean(dim=-1)  # (batch_size, d_model)
                pooled_features.append(pooled)

            return torch.cat(pooled_features, dim=-1)  # (batch_size, d_model * 3)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            torch.Tensor or tuple: Model predictions, optionally with attention weights
        """
        batch_size, seq_len, _ = x.shape
        attention_weights = []

        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create causal mask for autoregressive prediction
        if self.training:
            mask = self.create_causal_mask(seq_len, x.device)
        else:
            mask = None

        # Pass through transformer blocks
        for i, transformer_block in enumerate(self.transformer_blocks):
            x, attn_weights = transformer_block(x, mask, return_attention)
            if return_attention and attn_weights is not None:
                attention_weights.append(attn_weights)

        # Pool sequence
        pooled = self._pool_sequence(x)

        # Output projection
        output = self.output_projection(pooled)

        # Reshape for multi-step prediction
        if self.prediction_horizon > 1:
            output = output.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                output = output.squeeze(-1)  # (batch_size, prediction_horizon)
        else:
            output = output.view(batch_size, self.output_dim)

        if return_attention:
            return output, attention_weights
        return output

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for visualization

        Args:
            x: Input tensor

        Returns:
            List[torch.Tensor]: Attention weights from each layer
        """
        _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights

    def get_model_name(self) -> str:
        """Get model name for identification"""
        return (
            f"FinancialTransformer_L{self.n_layers}_H{self.n_heads}_"
            f"D{self.d_model}_{self.pooling_strategy}"
        )

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get detailed model configuration

        Returns:
            dict: Complete model configuration
        """
        return {
            "model_name": self.get_model_name(),
            "architecture": "transformer",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "dropout": self.dropout,
            "activation": self.activation,
            "pooling_strategy": self.pooling_strategy,
            "use_relative_position": self.use_relative_position,
            "pre_norm": self.pre_norm,
            "residual_scaling": self.residual_scaling,
            "num_parameters": self.get_num_parameters(),
        }


class LightweightTransformer(ModelInterface):
    """
    Lightweight transformer optimized for faster training and inference
    while maintaining good performance on financial time series.
    """

    def __init__(self, config: Config):
        """
        Initialize lightweight transformer

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Smaller dimensions for efficiency
        self.d_model = min(config.model.d_model, 128)
        self.n_heads = min(config.model.n_heads, 8)
        self.n_layers = min(config.model.n_layers, 4)
        self.d_ff = self.d_model * 2  # Smaller feed-forward dimension
        self.dropout = config.model.dropout
        self.input_dim = config.model.input_dim
        self.output_dim = config.model.output_dim
        self.sequence_length = config.model.sequence_length
        self.prediction_horizon = config.model.prediction_horizon

        # Simplified architecture
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(
            self.d_model, self.sequence_length, dropout=self.dropout
        )

        # Fewer transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation="relu",  # Faster than GELU
                    pre_norm=True,
                    use_relative_position=False,  # Disable for speed
                )
                for _ in range(self.n_layers)
            ]
        )

        # Simplified output
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.output_dim * self.prediction_horizon),
        )

        self._init_weights()
        self.logger.info(
            f"Initialized {self.get_model_name()} with {self.get_num_parameters():,} parameters"
        )

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Model predictions
        """
        batch_size, seq_len, _ = x.shape

        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Transformer layers (no mask for efficiency)
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x, mask=None, return_attention=False)

        # Use last timestep for prediction
        x = x[:, -1, :]  # (batch_size, d_model)

        # Output projection
        output = self.output_projection(x)

        # Reshape for multi-step prediction
        if self.prediction_horizon > 1:
            output = output.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                output = output.squeeze(-1)
        else:
            output = output.view(batch_size, self.output_dim)

        return output

    def get_model_name(self) -> str:
        """Get model name"""
        return (
            f"LightweightTransformer_L{self.n_layers}_H{self.n_heads}_D{self.d_model}"
        )


class HybridTransformer(ModelInterface):
    """
    Hybrid model combining transformer attention with LSTM for
    capturing both local patterns and long-range dependencies.
    """

    def __init__(self, config: Config):
        """
        Initialize hybrid transformer-LSTM model

        Args:
            config: Model configuration
        """
        super().__init__(config)

        self.d_model = config.model.d_model
        self.n_heads = config.model.n_heads
        self.n_layers = max(1, config.model.n_layers // 2)  # Fewer transformer layers
        self.lstm_layers = 2
        self.dropout = config.model.dropout
        self.input_dim = config.model.input_dim
        self.output_dim = config.model.output_dim
        self.sequence_length = config.model.sequence_length
        self.prediction_horizon = config.model.prediction_horizon

        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)

        # LSTM for local pattern extraction
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )

        # Transformer layers for attention-based modeling
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_model * 4,
                    dropout=self.dropout,
                    pre_norm=True,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.output_dim * self.prediction_horizon),
        )

        self._init_weights()
        self.logger.info(
            f"Initialized {self.get_model_name()} with {self.get_num_parameters():,} parameters"
        )

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Model predictions
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # LSTM processing for local patterns
        lstm_out, _ = self.lstm(x)

        # Add residual connection
        x = x + lstm_out

        # Transformer layers for global attention
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x, mask=None, return_attention=False)

        # Use last timestep
        x = x[:, -1, :]

        # Output projection
        output = self.output_projection(x)

        # Reshape for multi-step prediction
        if self.prediction_horizon > 1:
            output = output.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                output = output.squeeze(-1)
        else:
            output = output.view(batch_size, self.output_dim)

        return output

    def get_model_name(self) -> str:
        """Get model name"""
        return (
            f"HybridTransformer_L{self.n_layers}_LSTM{self.lstm_layers}_D{self.d_model}"
        )
