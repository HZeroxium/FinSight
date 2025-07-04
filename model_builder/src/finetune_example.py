# finetune.py

"""
Simple API for FinSight fine-tuning module.
Provides an easy-to-use interface for fine-tuning financial AI models.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from .finetune.main import create_default_facade


def finetune_model(
    data_path: str,
    output_dir: Optional[str] = None,
    model_name: str = "ibm/patchtsmixer-forecasting",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    use_wandb: bool = True,
    wandb_project: str = "finsight-finetune",
) -> Dict[str, Any]:
    """
    Fine-tune a financial AI model with minimal configuration.

    Args:
        data_path: Path to the financial dataset (CSV format)
        output_dir: Directory to save the fine-tuned model
        model_name: Hugging Face model name to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name

    Returns:
        Dictionary containing training results and model path

    Example:
        >>> results = finetune_model(
        ...     data_path="data/financial_data.csv",
        ...     model_name="microsoft/DialoGPT-medium",
        ...     num_epochs=5
        ... )
        >>> print(f"Model saved to: {results['model_path']}")
    """
    # Create facade with custom config
    facade = create_default_facade()

    # Update configuration
    facade.config.model_name = model_name
    facade.config.num_epochs = num_epochs
    facade.config.batch_size = batch_size
    facade.config.learning_rate = learning_rate
    facade.config.wandb.enabled = use_wandb
    facade.config.wandb.project = wandb_project

    if output_dir:
        facade.config.output_dir = Path(output_dir)

    return facade.finetune(data_path)


def main():
    """
    Main entry point for quick testing.
    """
    # Example usage with error handling
    # try:
    #     results = finetune_model(
    #         data_path="data/1d.csv",
    #         model_name="ibm/patchtsmixer-forecasting",
    #         num_epochs=3,
    #         batch_size=4,
    #         learning_rate=5e-5,
    #     )

    #     if results["status"] == "success":
    #         print(f"✅ Training completed successfully!")
    #         print(f"Model saved to: {results['model_path']}")
    #         print(
    #             f"Training loss: {results.get('training', {}).get('training_loss', 'N/A')}"
    #         )
    #     else:
    #         print(f"❌ Training failed: {results.get('error', 'Unknown error')}")

    # except Exception as e:
    #     print(f"❌ Unexpected error: {str(e)}")

    # Predict


if __name__ == "__main__":
    main()
