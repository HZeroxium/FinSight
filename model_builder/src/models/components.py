# models/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models

    This implementation provides learnable and fixed positional encodings
    with optional relative position bias for financial time series.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        learnable: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize positional encoding

        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            learnable: Whether to use learnable positional embeddings
            dropout: Dropout rate for positional encodings
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.learnable = learnable
        self.dropout = nn.Dropout(dropout)

        if learnable:
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        else:
            # Create sinusoidal positional encoding
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        seq_len = x.size(1)

        if self.learnable:
            positions = torch.arange(seq_len, device=x.device)
            pos_encoding = self.pos_embedding(positions).unsqueeze(0)
        else:
            pos_encoding = self.pe[:, :seq_len]

        return self.dropout(x + pos_encoding)


class MultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with optional relative position encoding
    and attention dropout for financial time series modeling.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_relative_position: bool = False,
        max_relative_position: int = 32,
    ):
        """
        Initialize multi-head attention

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_relative_position: Whether to use relative position encoding
            max_relative_position: Maximum relative position for encoding
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        # Attention dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Relative position encoding
        if use_relative_position:
            self.relative_position_k = nn.Embedding(
                2 * max_relative_position + 1, self.d_k
            )
            self.relative_position_v = nn.Embedding(
                2 * max_relative_position + 1, self.d_k
            )

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

    def get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position matrix

        Args:
            seq_len: Sequence length

        Returns:
            torch.Tensor: Relative position matrix
        """
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        return distance_mat_clipped + self.max_relative_position

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            tuple: (output, attention_weights) if return_attention else (output, None)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position encoding to attention scores
        if self.use_relative_position:
            relative_positions = self.get_relative_positions(seq_len).to(x.device)
            relative_k = self.relative_position_k(relative_positions)
            relative_scores = torch.matmul(
                Q.permute(2, 0, 1, 3), relative_k.transpose(-2, -1)
            )
            relative_scores = relative_scores.permute(1, 2, 0, 3)
            scores = scores + relative_scores / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Add relative position encoding to values
        if self.use_relative_position:
            relative_v = self.relative_position_v(relative_positions)
            relative_context = torch.matmul(
                attention_weights.permute(2, 0, 1, 3), relative_v
            ).permute(1, 2, 0, 3)
            context = context + relative_context

        # Concatenate heads and apply output projection
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        output = self.output_dropout(self.w_o(context))

        if return_attention:
            return output, attention_weights
        return output, None


class FeedForward(nn.Module):
    """
    Enhanced feed-forward network with multiple activation options
    and optional gating mechanism for financial modeling.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_gating: bool = False,
    ):
        """
        Initialize feed-forward network

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "swish")
            use_gating: Whether to use gating mechanism
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_gating = use_gating

        # Main layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        if use_gating:
            self.gate = nn.Linear(d_model, d_ff)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.use_gating:
            nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if self.use_gating:
            hidden = self.activation(self.linear1(x))
            gate = torch.sigmoid(self.gate(x))
            hidden = hidden * gate
        else:
            hidden = self.activation(self.linear1(x))

        hidden = self.dropout(hidden)
        return self.linear2(hidden)


class TransformerBlock(nn.Module):
    """
    Enhanced transformer block with pre-norm, optional cross-attention,
    and residual scaling for improved training stability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        pre_norm: bool = True,
        use_relative_position: bool = False,
        residual_scaling: float = 1.0,
    ):
        """
        Initialize transformer block

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            pre_norm: Whether to use pre-normalization
            use_relative_position: Whether to use relative position encoding
            residual_scaling: Scaling factor for residual connections
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.residual_scaling = residual_scaling

        # Attention and feed-forward layers
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, dropout, use_relative_position
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block

        Args:
            x: Input tensor
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            tuple: (output, attention_weights) if return_attention else (output, None)
        """
        # Self-attention with residual connection
        if self.pre_norm:
            normed_x = self.norm1(x)
            attn_output, attn_weights = self.self_attention(
                normed_x, mask, return_attention
            )
        else:
            attn_output, attn_weights = self.self_attention(x, mask, return_attention)

        x = x + self.dropout(attn_output) * self.residual_scaling

        if not self.pre_norm:
            x = self.norm1(x)

        # Feed-forward with residual connection
        if self.pre_norm:
            normed_x = self.norm2(x)
            ff_output = self.feed_forward(normed_x)
        else:
            ff_output = self.feed_forward(x)

        x = x + self.dropout(ff_output) * self.residual_scaling

        if not self.pre_norm:
            x = self.norm2(x)

        return x, attn_weights


class FinancialEmbedding(nn.Module):
    """
    Financial-specific embedding layer that handles both continuous
    and categorical features with learned embeddings.
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: Optional[Dict[str, int]] = None,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        """
        Initialize financial embedding layer

        Args:
            continuous_dim: Number of continuous features
            categorical_dims: Dictionary of categorical feature dimensions
            embedding_dim: Embedding dimension for categorical features
            dropout: Dropout rate
        """
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims or {}
        self.embedding_dim = embedding_dim

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feature_name, vocab_size in self.categorical_dims.items():
            embed_dim = min(embedding_dim, (vocab_size + 1) // 2)
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embed_dim)
            total_embedding_dim += embed_dim

        self.total_dim = continuous_dim + total_embedding_dim
        self.dropout = nn.Dropout(dropout)

        # Optional normalization for continuous features
        self.feature_norm = nn.LayerNorm(continuous_dim)

    def forward(
        self,
        continuous_features: torch.Tensor,
        categorical_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through embedding layer

        Args:
            continuous_features: Continuous feature tensor
            categorical_features: Dictionary of categorical feature tensors

        Returns:
            torch.Tensor: Combined embedded features
        """
        # Normalize continuous features
        embedded_continuous = self.feature_norm(continuous_features)

        # Process categorical features
        embedded_categoricals = []
        if categorical_features:
            for feature_name, feature_tensor in categorical_features.items():
                if feature_name in self.embeddings:
                    embedded = self.embeddings[feature_name](feature_tensor)
                    embedded_categoricals.append(embedded)

        # Combine all features
        if embedded_categoricals:
            embedded_categorical = torch.cat(embedded_categoricals, dim=-1)
            combined = torch.cat([embedded_continuous, embedded_categorical], dim=-1)
        else:
            combined = embedded_continuous

        return self.dropout(combined)

    def get_output_dim(self) -> int:
        """Get the output dimension after embedding"""
        return self.total_dim
