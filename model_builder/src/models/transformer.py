# models/transformer.py

import torch
import torch.nn as nn
from typing import List, Tuple, Union

from .interface import ModelInterface
from ..core.config import Config


class FinancialTransformer(ModelInterface):
    """
    Advanced Transformer model specifically designed for financial time series prediction,
    now leveraging torch.nn.TransformerEncoder for simplicity and maintainability.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Extract configuration
        m = config.model
        self.d_model = m.d_model
        self.n_heads = m.n_heads
        self.n_layers = m.n_layers
        self.d_ff = m.d_ff
        self.dropout = m.dropout
        self.input_dim = m.input_dim
        self.output_dim = m.output_dim
        self.sequence_length = m.sequence_length
        self.prediction_horizon = m.prediction_horizon

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # Positional encoding (fixed sinusoidal)
        self.pos_encoder = nn.Embedding(self.sequence_length, self.d_model)
        self._build_sinusoidal_pe()

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=m.activation if hasattr(m, "activation") else "gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model) if getattr(m, "pre_norm", True) else None,
        )

        # Pooling + output head
        self._setup_output_layers()

        # Initialize weights
        self._init_weights()
        self.logger.info(
            f"Initialized {self.get_model_name()} with {self.get_num_parameters():,} parameters"
        )

    def _build_sinusoidal_pe(self):
        """Create fixed sinusoidal positional embeddings"""
        pe = torch.zeros(self.sequence_length, self.d_model)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        with torch.no_grad():
            self.pos_encoder.weight.copy_(pe)
        self.pos_encoder.weight.requires_grad = False

    def _setup_output_layers(self):
        m = self.config.model
        pool = getattr(m, "pooling_strategy", "adaptive")
        if pool == "adaptive":
            self.pool = lambda x: x.mean(dim=1)
            head_input = self.d_model
        elif pool == "last":
            self.pool = lambda x: x[:, -1, :]
            head_input = self.d_model
        else:
            # fallback to mean
            self.pool = lambda x: x.mean(dim=1)
            head_input = self.d_model

        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Dropout(self.dropout),
            nn.Linear(head_input, head_input // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(head_input // 2, self.output_dim * self.prediction_horizon),
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass:
         - project inputs
         - add positional encodings
         - apply TransformerEncoder
         - pool and project to outputs
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)  # (B, L, D)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoder(pos)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, L, D)

        # Pool sequence
        pooled = self.pool(x)  # (B, D)

        # Head
        out = self.head(pooled)  # (B, output_dim * horizon)

        # Reshape
        if self.prediction_horizon > 1:
            out = out.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                out = out.squeeze(-1)
        else:
            out = out.view(batch_size, self.output_dim)

        return out

    def get_model_name(self) -> str:
        return (
            f"FinancialTransformer_L{self.n_layers}_H{self.n_heads}_" f"D{self.d_model}"
        )


class LightweightTransformer(ModelInterface):
    """
    Lightweight transformer optimized for faster training/inference,
    now using torch.nn.TransformerEncoder similarly.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        m = config.model
        self.d_model = min(m.d_model, 128)
        self.n_heads = min(m.n_heads, 8)
        self.n_layers = min(m.n_layers, 4)
        self.d_ff = self.d_model * 2
        self.dropout = m.dropout
        self.input_dim = m.input_dim
        self.output_dim = m.output_dim
        self.prediction_horizon = m.prediction_horizon
        self.sequence_length = m.sequence_length

        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        self.pos_encoder = nn.Embedding(self.sequence_length, self.d_model)
        self._build_sinusoidal_pe()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.head = nn.Sequential(
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

    def _build_sinusoidal_pe(self):
        pe = torch.zeros(self.sequence_length, self.d_model)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        with torch.no_grad():
            self.pos_encoder.weight.copy_(pe)
        self.pos_encoder.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoder(pos)

        x = self.transformer_encoder(x)

        last = x[:, -1, :]
        out = self.head(last)

        if self.prediction_horizon > 1:
            out = out.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                out = out.squeeze(-1)
        else:
            out = out.view(batch_size, self.output_dim)

        return out

    def get_model_name(self) -> str:
        return (
            f"LightweightTransformer_L{self.n_layers}_H{self.n_heads}_D{self.d_model}"
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(param)


class HybridTransformer(ModelInterface):
    """
    Hybrid Transformer + LSTM model simplified to use torch.nn.TransformerEncoder.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        m = config.model
        self.d_model = m.d_model
        self.n_heads = m.n_heads
        self.n_layers = max(1, m.n_layers // 2)
        self.d_ff = m.d_ff
        self.dropout = m.dropout
        self.input_dim = m.input_dim
        self.output_dim = m.output_dim
        self.sequence_length = m.sequence_length
        self.prediction_horizon = m.prediction_horizon

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.head = nn.Sequential(
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
        for name, param in self.named_parameters():
            if param.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        x = x + lstm_out

        x = self.transformer_encoder(x)

        last = x[:, -1, :]
        out = self.head(last)

        if self.prediction_horizon > 1:
            out = out.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                out = out.squeeze(-1)
        else:
            out = out.view(batch_size, self.output_dim)

        return out

    def get_model_name(self) -> str:
        return f"HybridTransformer_L{self.n_layers}_LSTM2_D{self.d_model}"
