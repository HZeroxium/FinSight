# finetune/model_factory.py

"""
Modern model factory for creating and configuring time series models with PEFT support.
"""

import torch
from typing import Optional, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    InformerConfig,
    InformerForPrediction,
    AutoformerConfig,
    AutoformerForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    TimesFmModelForPrediction,
    TimesFmConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType as PeftTaskType,
)

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, ModelType, PeftMethod, TaskType
from pathlib import Path


class ModelFactory:
    """Factory for creating fine-tuning models with PEFT support"""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(
            name="ModelFactory", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
        )

    def create_model_and_tokenizer(
        self,
    ) -> Tuple[torch.nn.Module, Optional[AutoTokenizer]]:
        """
        Create model and tokenizer based on configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Creating model: {self.config.model_name}")

        # Load base model
        model = self._create_base_model()

        # Check and apply gradient checkpointing if supported
        if (
            self.config.gradient_checkpointing
            and self._supports_gradient_checkpointing(model)
        ):
            model.gradient_checkpointing_enable()
            self.logger.info("✓ Gradient checkpointing enabled")
        elif self.config.gradient_checkpointing:
            self.logger.warning(
                "Model does not support gradient checkpointing, skipping"
            )

        # Apply PEFT if enabled and supported
        if self.config.use_peft and self._supports_peft(model):
            model = self._apply_peft(model)

        # Compile model if configured (PyTorch 2.0+)
        if self.config.compile_model:
            try:
                model = torch.compile(model)
                self.logger.info("✓ Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")

        self.logger.info(
            f"✓ Model created with {self._count_parameters(model):,} parameters"
        )

        return model, None  # Time series models don't need tokenizers

    def _create_base_model(self) -> torch.nn.Module:
        """Create the base model with better error handling"""
        try:
            model_name = self.config.model_name

            # Configure model for forecasting with safe defaults
            if model_name == ModelType.PATCH_TSMIXER:
                config = PatchTSMixerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    patch_length=min(
                        8, self.config.sequence_length // 4
                    ),  # Safe patch length
                    num_patches=max(
                        1,
                        self.config.sequence_length
                        // min(8, self.config.sequence_length // 4),
                    ),
                )
                model = PatchTSMixerForPrediction(config)

            elif model_name == ModelType.PATCH_TST:
                config = PatchTSTConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    patch_length=min(8, self.config.sequence_length // 4),
                    num_patches=max(
                        1,
                        self.config.sequence_length
                        // min(8, self.config.sequence_length // 4),
                    ),
                )
                model = PatchTSTForPrediction(config)

            elif model_name == ModelType.INFORMER:
                config = InformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    d_model=64,  # Smaller model for efficiency
                )
                model = InformerForPrediction.from_pretrained(model_name, config=config)

            elif model_name == ModelType.AUTOFORMER:
                config = AutoformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    d_model=64,
                )
                model = AutoformerForPrediction.from_pretrained(
                    model_name, config=config
                )

            elif model_name == ModelType.TIME_SERIES_TRANSFORMER:
                config = TimeSeriesTransformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    d_model=64,
                )
                model = TimeSeriesTransformerForPrediction.from_pretrained(
                    model_name, config=config
                )

            elif model_name == ModelType.TIMESFM:
                config = TimesFmConfig(
                    context_length=self.config.sequence_length,
                    horizon_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    d_model=64,
                )
                model = TimesFmModelForPrediction.from_pretrained(
                    model_name, config=config
                )

            else:
                # Fallback to simple model
                self.logger.warning(
                    f"Unknown model {model_name}, creating simple model"
                )
                model = self._create_simple_timeseries_model()

            return model

        except Exception as e:
            self.logger.error(f"Failed to create {model_name}: {e}")
            self.logger.info("Creating fallback simple time series model...")
            return self._create_simple_timeseries_model()

    def _create_simple_timeseries_model(self) -> torch.nn.Module:
        """Create a simple LSTM-based time series model as fallback"""
        import torch.nn as nn

        class SimpleTimeSeriesModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last output
                last_output = lstm_out[:, -1, :]
                output = self.fc(self.dropout(last_output))
                return output

            def save_pretrained(self, path):
                """Save model for compatibility"""
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

        return SimpleTimeSeriesModel(
            input_size=len(self.config.features),
            hidden_size=64,
            num_layers=2,
            output_size=self.config.prediction_horizon,
        )

    def _supports_gradient_checkpointing(self, model: torch.nn.Module) -> bool:
        """Check if model supports gradient checkpointing"""
        return hasattr(model, "gradient_checkpointing_enable") and callable(
            getattr(model, "gradient_checkpointing_enable")
        )

    def _supports_peft(self, model: torch.nn.Module) -> bool:
        """Check if model supports PEFT"""
        # Most HuggingFace models support PEFT
        return hasattr(model, "named_modules")

    def _apply_peft(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply PEFT to the model"""
        self.logger.info(f"Applying PEFT method: {self.config.peft_method}")

        try:
            # Determine PEFT task type
            peft_task_type = self._get_peft_task_type()

            # Create PEFT config based on method
            if self.config.peft_method == PeftMethod.LORA:
                peft_config = LoraConfig(
                    task_type=peft_task_type,
                    inference_mode=False,
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules
                    or self._get_target_modules(model),
                )
            else:
                # Fallback to LoRA for unsupported methods
                peft_config = LoraConfig(
                    task_type=peft_task_type,
                    inference_mode=False,
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self._get_target_modules(model),
                )

            # Apply PEFT
            model = get_peft_model(model, peft_config)

            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())

            self.logger.info(
                f"✓ PEFT applied: {trainable_params:,} trainable / {total_params:,} total parameters "
                f"({trainable_params/total_params*100:.2f}% trainable)"
            )

            return model

        except Exception as e:
            self.logger.warning(f"Failed to apply PEFT: {e}, using base model")
            return model

    def _get_peft_task_type(self) -> PeftTaskType:
        """Get PEFT task type from config"""
        if self.config.task_type == TaskType.REGRESSION:
            return PeftTaskType.SEQ_CLS
        elif self.config.task_type == TaskType.CLASSIFICATION:
            return PeftTaskType.SEQ_CLS
        else:
            return PeftTaskType.FEATURE_EXTRACTION

    def _get_target_modules(self, model: torch.nn.Module) -> List[str]:
        """Auto-detect target modules for PEFT"""
        target_modules = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.append(name.split(".")[-1])

        # Common target modules for different architectures
        common_targets = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "dense",
            "fc",
            "linear",
            "projection",
        ]

        # Filter to common targets if they exist
        filtered_targets = [
            t for t in common_targets if any(t in name for name in target_modules)
        ]

        if filtered_targets:
            return list(set(filtered_targets))[:4]  # Limit to avoid too many targets

        # Fallback to first few linear layers
        return list(set(target_modules))[:4] if target_modules else ["dense"]

    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in model.parameters())
