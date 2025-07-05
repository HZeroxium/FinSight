# peft_config.py

from pydantic import BaseModel, Field
from typing import Literal, Optional


class PEFTConfig(BaseModel):
    """
    Configuration for PEFT (parameter-efficient fine-tuning).
    """

    peft_method: Literal["lora", "adapter", "prefix_tuning"] = Field(
        "lora", description="PEFT method to apply"
    )
    # LoRA-specific
    lora_r: Optional[int] = Field(8, description="LoRA rank")
    lora_alpha: Optional[int] = Field(16, description="LoRA alpha")
    lora_dropout: Optional[float] = Field(0.05, description="LoRA dropout")
    # Adapter-specific
    adapter_reduction_factor: Optional[int] = Field(
        16, description="Adapter bottleneck reduction factor"
    )
    # Prefix Tuning-specific
    prefix_length: Optional[int] = Field(
        30, description="Prefix tuning sequence length"
    )
    prefix_dropout: Optional[float] = Field(0.1, description="Prefix tuning dropout")
    # Common
    target_modules: list[str] = Field(
        ["q_proj", "v_proj"], description="List of module name patterns to apply PEFT"
    )
