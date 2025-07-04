# finetune/model_factory.py

"""
Modern model factory for creating and configuring time series models with PEFT support.
"""

import torch
from typing import Optional, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType as PeftTaskType,
)
import torch.nn as nn

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, ModelType, PeftMethod, TaskType


class SimpleTimeSeriesModel(nn.Module):
    """Simple time series model for financial forecasting"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        return self.fc(output)


class ModelFactory:
    """Factory for creating fine-tuning models with PEFT support"""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(
            name="ModelFactory", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
        )

    def create_model_and_tokenizer(self) -> tuple[torch.nn.Module, AutoTokenizer]:
        """
        Create model and tokenizer based on configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Creating model: {self.config.model_name}")

        # Load tokenizer (None for time series models)
        tokenizer = self._create_tokenizer()

        # Load base model
        model = self._create_base_model()

        # Apply PEFT if enabled and supported
        if self.config.use_peft and self._supports_peft(model):
            model = self._apply_peft(model)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()

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

        return model, tokenizer

    def _create_tokenizer(self) -> Optional[AutoTokenizer]:
        """Create tokenizer for the specified model"""
        # Time series models don't need tokenizers
        if self.config.task_type == TaskType.FORECASTING:
            return None

        try:
            if self.config.model_name in [ModelType.FLAN_T5]:
                tokenizer = T5Tokenizer.from_pretrained(
                    self.config.model_name, cache_dir=self.config.cache_dir
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name, cache_dir=self.config.cache_dir
                )

            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer

        except Exception as e:
            self.logger.warning(f"Failed to create tokenizer: {e}, using None")
            return None

    def _create_base_model(self) -> torch.nn.Module:
        """Create the base model"""
        try:
            # For time series models, create a simple LSTM-based model
            if (
                self.config.task_type == TaskType.FORECASTING
                or "autoformer" in self.config.model_name.lower()
                or "informer" in self.config.model_name.lower()
                or "patchtst" in self.config.model_name.lower()
                or "timesfm" in self.config.model_name.lower()
            ):
                # Create simple time series model
                input_size = len(self.config.features)
                model = SimpleTimeSeriesModel(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    output_size=self.config.prediction_horizon,
                )
                return model

            # For language models
            model_config = AutoConfig.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            )

            # Configure model for task
            if self.config.task_type == TaskType.REGRESSION:
                model_config.num_labels = 1
                model_config.problem_type = "regression"
            elif self.config.task_type == TaskType.CLASSIFICATION:
                model_config.num_labels = 2
                model_config.problem_type = "single_label_classification"

            # Load model based on type
            if self.config.model_name == ModelType.FLAN_T5:
                model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create base model: {e}")
            # Fallback to simple model
            input_size = len(self.config.features)
            return SimpleTimeSeriesModel(input_size=input_size)

    def _supports_peft(self, model: torch.nn.Module) -> bool:
        """Check if model supports PEFT"""
        # Simple models don't support PEFT
        if isinstance(model, SimpleTimeSeriesModel):
            return False
        return True

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
            return PeftTaskType.CAUSAL_LM

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
        ]

        # Filter to common targets if they exist
        filtered_targets = [
            t for t in common_targets if any(t in name for name in target_modules)
        ]

        if filtered_targets:
            return list(set(filtered_targets))[:4]  # Limit to avoid too many targets

        # Fallback to first few linear layers
        return list(set(target_modules))[:4] if target_modules else ["fc"]

    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in model.parameters())
