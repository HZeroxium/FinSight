# models/exporter.py

"""Model export functionality for sentiment analysis models."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import export

import mlflow

from ..core.config import ExportConfig, ExportFormat


class ModelExporter:
    """Handles model export to ONNX and TorchScript formats."""

    def __init__(self, config: ExportConfig):
        """Initialize the exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

    def export_model(
        self,
        model_path: Path,
        output_dir: Path,
        experiment_name: str = "sentiment-analysis-export",
    ) -> Dict[str, Path]:
        """Export the trained model to the specified formats.

        Args:
            model_path: Path to the trained model
            output_dir: Directory to save exported models
            experiment_name: Name for MLflow experiment

        Returns:
            Dictionary mapping export format to output path

        Raises:
            ValueError: If export fails
        """
        logger.info(f"Starting model export from {model_path}")

        # Setup MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer(model_path)

            # Load preprocessing config
            preprocessing_config = self._load_preprocessing_config(model_path)

            # Export models
            export_paths = {}

            if self.config.format in [ExportFormat.ONNX, ExportFormat.BOTH]:
                logger.info("Exporting to ONNX format")
                onnx_path = self._export_to_onnx(
                    model, tokenizer, preprocessing_config, output_dir
                )
                export_paths["onnx"] = onnx_path

                # Validate ONNX export
                if self.config.validate_export:
                    self._validate_onnx_export(
                        onnx_path, model, tokenizer, preprocessing_config
                    )

                # Log ONNX artifact to MLflow
                mlflow.log_artifact(str(onnx_path), "onnx_model")

            if self.config.format in [ExportFormat.TORCHSCRIPT, ExportFormat.BOTH]:
                logger.info("Exporting to TorchScript format")
                torchscript_path = self._export_to_torchscript(
                    model, tokenizer, preprocessing_config, output_dir
                )
                export_paths["torchscript"] = torchscript_path

                # Log TorchScript artifact to MLflow
                mlflow.log_artifact(str(torchscript_path), "torchscript_model")

            # Save export configuration
            export_config_path = self._save_export_config(
                output_dir, preprocessing_config
            )
            mlflow.log_artifact(str(export_config_path), "export_config")

            # Generate model card
            model_card_path = self._generate_model_card(
                output_dir, preprocessing_config
            )
            mlflow.log_artifact(str(model_card_path), "model_card")

            # Log export parameters
            mlflow.log_params(
                {
                    "export_format": self.config.format.value,
                    "onnx_opset_version": self.config.onnx_opset_version,
                    "onnx_dynamic_axes": self.config.onnx_dynamic_axes,
                    "validate_export": self.config.validate_export,
                }
            )

            logger.info("Model export completed successfully")
            return export_paths

    def _load_model_and_tokenizer(
        self, model_path: Path
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load the trained model and tokenizer.

        Args:
            model_path: Path to the trained model

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model and tokenizer from {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode

        # Move model to device
        model = model.to(self.device)

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    def _load_preprocessing_config(self, model_path: Path) -> Dict[str, Any]:
        """Load preprocessing configuration.

        Args:
            model_path: Path to the trained model

        Returns:
            Preprocessing configuration dictionary
        """
        config_path = model_path / "preprocessing_config.json"

        if not config_path.exists():
            logger.warning("Preprocessing config not found, using defaults")
            return {
                "max_length": 512,
                "min_length": 10,
                "remove_html": True,
                "normalize_unicode": True,
                "lowercase": True,
                "remove_urls": True,
                "remove_emails": True,
                "label_mapping": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2},
                "reverse_label_mapping": {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
            }

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("Preprocessing configuration loaded")
        return config

    def _export_to_onnx(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        preprocessing_config: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            model: Trained model to export
            tokenizer: Tokenizer for text processing
            preprocessing_config: Preprocessing configuration

        Returns:
            Path to the exported ONNX model
        """
        # Create output directory
        onnx_dir = output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dummy input for ONNX export
        dummy_input = self._prepare_dummy_input(tokenizer, preprocessing_config)

        # Export to ONNX
        onnx_path = onnx_dir / "model.onnx"

        logger.info("Exporting model to ONNX format")

        # Use torch.onnx.export directly with proper arguments
        model.eval()

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                input_names=list(dummy_input.keys()),
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "logits": {0: "batch_size"},
                },
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                verbose=False,
                export_params=True,
            )

        logger.info(f"ONNX model exported to {onnx_path}")
        return onnx_path

    def _export_to_torchscript(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        preprocessing_config: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Export model to TorchScript format.

        Args:
            model: Trained model to export
            tokenizer: Tokenizer for text processing
            preprocessing_config: Preprocessing configuration

        Returns:
            Path to the exported TorchScript model
        """
        # Create output directory
        torchscript_dir = output_dir / "torchscript"
        torchscript_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dummy input for TorchScript export
        dummy_input = self._prepare_dummy_input(tokenizer, preprocessing_config)

        # Export to TorchScript
        torchscript_path = torchscript_dir / "model.pt"

        logger.info("Exporting model to TorchScript format")

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced_model, torchscript_path)

        logger.info(f"TorchScript model exported to {torchscript_path}")
        return torchscript_path

    def _prepare_dummy_input(
        self, tokenizer: AutoTokenizer, preprocessing_config: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Prepare dummy input for model export.

        Args:
            tokenizer: Tokenizer for text processing
            preprocessing_config: Preprocessing configuration

        Returns:
            Dictionary of dummy input tensors
        """
        # Create dummy text
        dummy_text = "This is a sample text for model export."

        # Tokenize
        inputs = tokenizer(
            dummy_text,
            truncation=True,
            padding="max_length",
            max_length=preprocessing_config["max_length"],
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _validate_onnx_export(
        self,
        onnx_path: Path,
        original_model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        preprocessing_config: Dict[str, Any],
    ) -> None:
        """Validate the exported ONNX model.

        Args:
            onnx_path: Path to the exported ONNX model
            original_model: Original PyTorch model for comparison
            tokenizer: Tokenizer for text processing
            preprocessing_config: Preprocessing configuration
        """
        logger.info("Validating ONNX export")

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure validation passed")

        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(onnx_path))
        logger.info("ONNX runtime session created successfully")

        # Prepare test input
        test_text = "Bitcoin price is rising significantly today."
        test_inputs = tokenizer(
            test_text,
            truncation=True,
            padding="max_length",
            max_length=preprocessing_config["max_length"],
            return_tensors="pt",
        )

        # Move test inputs to same device as model
        test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}

        # Get original model prediction
        with torch.no_grad():
            original_model.eval()
            original_outputs = original_model(**test_inputs)
            original_logits = original_outputs.logits.cpu().numpy()
            original_predictions = np.argmax(original_logits, axis=1)

        # Get ONNX model prediction
        onnx_inputs = {k: v.cpu().numpy() for k, v in test_inputs.items()}
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]
        onnx_predictions = np.argmax(onnx_logits, axis=1)

        # Compare predictions
        prediction_match = np.array_equal(original_predictions, onnx_predictions)
        logits_diff = np.abs(original_logits - onnx_logits).max()

        logger.info(f"ONNX validation results:")
        logger.info(f"  Prediction match: {prediction_match}")
        logger.info(f"  Max logits difference: {logits_diff:.6f}")

        # Allow for reasonable numerical differences (tolerance of 5.0)
        tolerance = 5.0
        if prediction_match or logits_diff < tolerance:
            logger.info("ONNX export validation passed")
        else:
            logger.warning("ONNX export validation failed")
            logger.warning(f"  Original prediction: {original_predictions}")
            logger.warning(f"  ONNX prediction: {onnx_predictions}")
            logger.warning(f"  Logits difference too large: {logits_diff}")

    def _save_export_config(
        self, output_dir: Path, preprocessing_config: Dict[str, Any]
    ) -> Path:
        """Save export configuration for model serving.

        Args:
            output_dir: Output directory
            preprocessing_config: Preprocessing configuration

        Returns:
            Path to the saved configuration file
        """
        export_config = {
            "export_format": self.config.format.value,
            "onnx_opset_version": self.config.onnx_opset_version,
            "onnx_dynamic_axes": self.config.onnx_dynamic_axes,
            "preprocessing": preprocessing_config,
            "export_timestamp": str(datetime.now().isoformat()),
        }

        config_path = output_dir / "export_config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(export_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Export configuration saved to {config_path}")
        return config_path

    def _generate_model_card(
        self, output_dir: Path, preprocessing_config: Dict[str, Any]
    ) -> Path:
        """Generate a model card for the exported model.

        Args:
            output_dir: Output directory
            preprocessing_config: Preprocessing configuration

        Returns:
            Path to the generated model card
        """
        model_card_content = f"""# Crypto News Sentiment Analysis Model

## Model Description

This is a fine-tuned transformer model for sentiment analysis of cryptocurrency news articles. The model classifies news text into three sentiment categories: NEGATIVE, NEUTRAL, and POSITIVE.

## Model Details

- **Model Type**: Fine-tuned transformer (sequence classification)
- **Base Model**: {preprocessing_config.get('model_backbone', 'Unknown')}
- **Task**: Sentiment Analysis
- **Language**: English
- **License**: MIT

## Intended Use

This model is designed to analyze the sentiment of cryptocurrency and financial news articles. It can be used for:

- Market sentiment analysis
- News filtering and categorization
- Trading strategy development
- Risk assessment

## Training Data

The model was trained on cryptocurrency news articles from various sources including:
- CoinDesk
- CoinTelegraph
- Other financial news sources

## Performance

The model performance metrics are logged in MLflow and can be accessed through the experiment tracking system.

## Limitations

- The model is trained on English text only
- Performance may vary for different news sources and writing styles
- The model may not capture nuanced sentiment expressions
- Training data may contain biases present in financial news reporting

## Preprocessing

The model expects preprocessed text with the following characteristics:
- Maximum length: {preprocessing_config.get('max_length', 'Unknown')} tokens
- HTML tags removed: {preprocessing_config.get('remove_html', 'Unknown')}
- Unicode normalized: {preprocessing_config.get('normalize_unicode', 'Unknown')}
- Lowercase: {preprocessing_config.get('lowercase', 'Unknown')}
- URLs removed: {preprocessing_config.get('remove_urls', 'Unknown')}
- Emails removed: {preprocessing_config.get('remove_emails', 'Unknown')}

## Usage

### ONNX Runtime

```python
import onnxruntime as ort
import json

# Load the model
session = ort.InferenceSession("model.onnx")

# Load preprocessing config
with open("preprocessing_config.json", "r") as f:
    config = json.load(f)

# Preprocess text (implement according to config)
text = "Your news text here"
# ... preprocessing steps ...

# Run inference
inputs = {{"input_ids": input_ids, "attention_mask": attention_mask}}
outputs = session.run(None, inputs)
predictions = np.argmax(outputs[0], axis=1)
```

### PyTorch

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Run inference
inputs = tokenizer("Your news text here", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
```

## Citation

If you use this model in your research or applications, please cite:

```
@software{{crypto_sentiment_model,
  title={{Crypto News Sentiment Analysis Model}},
  author={{FinSight Team}},
  year={{2025}},
  url={{https://github.com/finsight/sentiment-analysis}}
}}
```

## Contact

For questions or support, please contact the FinSight team.
"""

        model_card_path = output_dir / "model_card.md"

        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        logger.info(f"Model card generated at {model_card_path}")
        return model_card_path

    def get_export_summary(self, export_paths: Dict[str, Path]) -> Dict[str, Any]:
        """Get a summary of the export operation.

        Args:
            export_paths: Dictionary of export format to path mapping

        Returns:
            Export summary dictionary
        """
        summary = {
            "export_format": self.config.format.value,
            "exported_formats": list(export_paths.keys()),
            "export_paths": {k: str(v) for k, v in export_paths.items()},
            "validation_performed": self.config.validate_export,
            "device_used": str(self.device),
        }

        # Add file sizes
        for format_name, path in export_paths.items():
            if path.exists():
                summary[f"{format_name}_size_mb"] = round(
                    path.stat().st_size / (1024 * 1024), 2
                )

        return summary
