# finetune.py

"""
Simple API for FinSight fine-tuning module.
Provides an easy-to-use interface for fine-tuning financial AI models.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from .finetune.main import create_default_facade, quick_finetune


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


def predict_with_model(
    model_path: str,
    input_data: Any,
) -> Dict[str, Any]:
    """
    Make predictions using a fine-tuned model.

    Args:
        model_path: Path to the saved fine-tuned model
        input_data: Input data for prediction

    Returns:
        Prediction results
    """
    facade = create_default_facade()
    return facade.predict(input_data, model_path)


def evaluate_model(
    model_path: str,
    test_data_path: str,
) -> Dict[str, Any]:
    """
    Evaluate a fine-tuned model on test data.

    Args:
        model_path: Path to the saved model
        test_data_path: Path to test dataset

    Returns:
        Evaluation metrics
    """
    facade = create_default_facade()
    return facade.evaluate_existing_model(model_path, test_data_path)


# Convenience function for quick start
def demo_finetune(
    data_path: str = "data/1d.csv",
) -> Dict[str, Any]:
    """
    Demo function for quick fine-tuning with default settings.

    Args:
        data_path: Path to the dataset

    Returns:
        Training results
    """
    return quick_finetune(
        data_path=data_path,
        output_dir="./demo_finetune_output",
        model_name="ibm/patchtsmixer-forecasting",  # Smaller model for demo
    )


def main():
    """
    Main entry point for quick testing.
    """
    # Example usage with error handling
    try:
        results = finetune_model(
            data_path="data/1d.csv",
            model_name="ibm/patchtsmixer-forecasting",
            num_epochs=3,
            batch_size=4,
            learning_rate=5e-5,
        )

        if results["status"] == "success":
            print(f"✅ Training completed successfully!")
            print(f"Model saved to: {results['model_path']}")
            print(
                f"Training loss: {results.get('training', {}).get('training_loss', 'N/A')}"
            )
        else:
            print(f"❌ Training failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
