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
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType as PeftTaskType,
)

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, ModelType, PeftMethod, TaskType


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

        return model, None  # Time series models don't need tokenizers

    def _create_base_model(self) -> torch.nn.Module:
        """Create the base model"""
        try:
            model_name = self.config.model_name

            # Configure model for forecasting
            if model_name == ModelType.PATCH_TSMIXER:
                config = PatchTSMixerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    patch_length=8,
                    num_patches=self.config.sequence_length // 8,
                )
                model = PatchTSMixerForPrediction(config)

            elif model_name == ModelType.PATCH_TST:
                config = PatchTSTConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                    patch_length=8,
                    num_patches=self.config.sequence_length // 8,
                )
                model = PatchTSTForPrediction(config)

            elif model_name == ModelType.INFORMER:
                config = InformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                )
                model = InformerForPrediction(config)

            elif model_name == ModelType.AUTOFORMER:
                config = AutoformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                )
                model = AutoformerForPrediction(config)

            elif model_name == ModelType.TIME_SERIES_TRANSFORMER:
                config = TimeSeriesTransformerConfig(
                    context_length=self.config.sequence_length,
                    prediction_length=self.config.prediction_horizon,
                    num_input_channels=len(self.config.features),
                )
                model = TimeSeriesTransformerForPrediction(config)

            elif model_name == ModelType.TIMESFM:
                # For TimesFM, use AutoModel as fallback
                self.logger.warning("TimesFM not directly supported, using AutoModel")
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                )
            else:
                # Fallback to AutoModel
                self.logger.info(f"Using AutoModel for {model_name}")
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create base model: {e}")
            # Try AutoModel as final fallback
            try:
                self.logger.info("Trying AutoModel as fallback...")
                model = AutoModel.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=(
                        torch.float16 if self.config.use_fp16 else torch.float32
                    ),
                )
                return model
            except Exception as e2:
                self.logger.error(f"All model creation attempts failed: {e2}")
                raise

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
