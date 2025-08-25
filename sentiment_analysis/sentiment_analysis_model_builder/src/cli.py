# cli.py

"""Command-line interface for the sentiment analysis model builder."""

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from .core.config import Config
from .data.data_loader import DataLoader
from .data.dataset import DatasetPreparator
from .models.trainer import SentimentTrainer
from .models.exporter import ModelExporter
from .registry.mlflow_registry import MLflowRegistry
from .utils.file_utils import save_json, ensure_directory
from .utils.model_evaluator import ModelEvaluator

# Create Typer app
app = typer.Typer(
    name="sentiment-analysis",
    help="Crypto News Sentiment Analysis Model Builder",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


def setup_logging(log_level: str) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
    """
    # Remove default handler
    logger.remove()

    # Add console handler with specified level
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


@app.command()
def train(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file (YAML or .env)"
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data", "-d", help="Path to input data file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for training artifacts"
    ),
    experiment_name: str = typer.Option(
        "sentiment-analysis", "--experiment", "-e", help="MLflow experiment name"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Train a sentiment analysis model on crypto news data."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            # Load from specific config file
            config = Config(_env_file=config_file)
        else:
            # Load from default locations
            config = Config()

        # Override config with CLI arguments
        if data_path:
            config.data.input_path = data_path
        if output_dir:
            config.output_dir = output_dir

        # Validate configuration
        if not config.data.input_path or not config.data.input_path.exists():
            console.print(
                f"[red]Error: Input data file not found: {config.data.input_path}[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[green]Configuration loaded successfully[/green]")
        console.print(f"  Data path: {config.data.input_path}")
        console.print(f"  Output directory: {config.output_dir}")
        console.print(f"  Model backbone: {config.training.backbone.value}")
        console.print(f"  Experiment name: {experiment_name}")

        # Create output directory
        ensure_directory(config.output_dir)

        # Initialize components
        data_loader = DataLoader(config.data, config.preprocessing)
        dataset_preparator = DatasetPreparator(config.preprocessing, config.training)
        trainer = SentimentTrainer(config.training, dataset_preparator)

        # Load and validate data
        console.print("\n[yellow]Loading and preprocessing data...[/yellow]")
        articles = data_loader.load_data()

        if not data_loader.validate_data(articles):
            console.print("[red]Error: Data validation failed[/red]")
            raise typer.Exit(1)

        # Prepare datasets
        console.print("[yellow]Preparing datasets...[/yellow]")
        datasets = dataset_preparator.prepare_datasets(articles)

        # Train model
        console.print("[yellow]Starting model training...[/yellow]")
        model, tokenizer, training_metrics = trainer.train(
            datasets=datasets,
            output_dir=config.output_dir,
            experiment_name=experiment_name,
            registry_config=config.registry,
        )

        # Save training summary
        summary_path = config.output_dir / "training_summary.json"
        save_json(training_metrics.model_dump(), summary_path)

        console.print(f"\n[green]Training completed successfully![/green]")
        console.print(f"  Model saved to: {config.output_dir / 'model'}")
        console.print(f"  Training summary: {summary_path}")
        console.print(f"  MLflow run ID: {training_metrics.run_id}")
        console.print(f"  Experiment ID: {training_metrics.experiment_id}")

        # Display final metrics
        if "test" in training_metrics.eval_results:
            test_results = training_metrics.eval_results["test"]
            console.print(f"\n[bold]Test Set Performance:[/bold]")
            console.print(f"  Accuracy: {test_results.eval_accuracy:.4f}")
            console.print(f"  F1 Macro: {test_results.eval_f1_macro:.4f}")
            console.print(f"  F1 Weighted: {test_results.eval_f1_weighted:.4f}")

    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        logger.exception("Training failed")
        raise typer.Exit(1)


@app.command()
def export(
    model_path: Path = typer.Argument(..., help="Path to the trained model directory"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for exported models"
    ),
    format: str = typer.Option(
        "onnx", "--format", "-f", help="Export format: onnx, torchscript, or both"
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate exported models"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Export a trained model to ONNX or TorchScript format."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            config = Config(_env_file=config_file)
        else:
            config = Config()

        # Override config with CLI arguments
        if output_dir:
            config.export.output_dir = output_dir

        # Validate model path
        if not model_path.exists():
            console.print(f"[red]Error: Model path not found: {model_path}[/red]")
            raise typer.Exit(1)

        if not (model_path / "config.json").exists():
            console.print(f"[red]Error: Invalid model directory: {model_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Exporting model from: {model_path}[/green]")
        console.print(f"  Output directory: {config.export.output_dir}")
        console.print(f"  Format: {format}")
        console.print(f"  Validate: {validate}")

        # Initialize exporter
        exporter = ModelExporter(config.export)

        # Export model
        console.print("\n[yellow]Exporting model...[/yellow]")
        export_paths = exporter.export_model(
            model_path=model_path,
            output_dir=config.export.output_dir,
            experiment_name="sentiment-analysis-export",
        )

        # Display export summary
        export_summary = exporter.get_export_summary(export_paths)

        console.print(f"\n[green]Export completed successfully![/green]")
        console.print(
            f"  Exported formats: {', '.join(export_summary['exported_formats'])}"
        )

        for format_name, path in export_paths.items():
            console.print(f"  {format_name.upper()}: {path}")
            if f"{format_name}_size_mb" in export_summary:
                console.print(
                    f"    Size: {export_summary[f'{format_name}_size_mb']} MB"
                )

        # Save export summary
        summary_path = config.export.output_dir / "export_summary.json"
        save_json(export_summary, summary_path)

        console.print(f"  Export summary: {summary_path}")

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        logger.exception("Export failed")
        raise typer.Exit(1)


@app.command()
def register(
    model_path: Path = typer.Argument(..., help="Path to the trained model directory"),
    run_id: str = typer.Option(
        ..., "--run-id", "-r", help="MLflow run ID from training"
    ),
    description: str = typer.Option(
        "Crypto News Sentiment Analysis Model",
        "--description",
        "-d",
        help="Model description",
    ),
    stage: str = typer.Option(
        "Staging",
        "--stage",
        "-s",
        help="Initial model stage (Staging, Production, Archived)",
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Register a trained model in the MLflow model registry."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            config = Config(_env_file=config_file)
        else:
            config = Config()

        # Override config with CLI arguments
        config.registry.model_stage = stage

        # Validate model path
        if not model_path.exists():
            console.print(f"[red]Error: Model path not found: {model_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Registering model from: {model_path}[/green]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Description: {description}")
        console.print(f"  Stage: {stage}")
        console.print(f"  Registry: {config.registry.tracking_uri}")

        # Initialize registry
        registry = MLflowRegistry(config.registry)

        # Register model
        console.print("\n[yellow]Registering model in MLflow...[/yellow]")
        model_uri = registry.register_model(
            model_path=model_path, run_id=run_id, description=description
        )

        console.print(f"\n[green]Model registered successfully![/green]")
        console.print(f"  Model URI: {model_uri}")

        # Display registry summary
        summary = registry.get_registry_summary()
        if summary:
            console.print(f"\n[bold]Registry Summary:[/bold]")
            console.print(f"  Model: {summary['model_name']}")
            console.print(f"  Total versions: {summary['total_versions']}")
            console.print(f"  Stage counts: {summary['stage_counts']}")
            console.print(f"  Latest versions: {summary['latest_versions']}")

    except Exception as e:
        console.print(f"[red]Registration failed: {e}[/red]")
        logger.exception("Registration failed")
        raise typer.Exit(1)


@app.command()
def list_models(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """List all models in the MLflow registry."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            config = Config(_env_file=config_file)
        else:
            config = Config()

        # Initialize registry
        registry = MLflowRegistry(config.registry)

        # Get model versions
        versions = registry.list_model_versions()

        if not versions:
            console.print("[yellow]No model versions found in registry[/yellow]")
            return

        # Create table
        table = Table(title="Model Versions")
        table.add_column("Version", style="cyan")
        table.add_column("Stage", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Run ID", style="blue")
        table.add_column("Created", style="magenta")

        for version in versions:
            table.add_row(
                str(version["version"]),
                version["stage"],
                version["status"],
                version["run_id"][:8] + "...",
                str(version["created_at"])[:19],
            )

        console.print(table)

        # Display summary
        summary = registry.get_registry_summary()
        if summary:
            console.print(f"\n[bold]Registry Summary:[/bold]")
            console.print(f"  Model: {summary['model_name']}")
            console.print(f"  Total versions: {summary['total_versions']}")
            console.print(f"  Stage counts: {summary['stage_counts']}")
            console.print(f"  Latest versions: {summary['latest_versions']}")

    except Exception as e:
        console.print(f"[red]Failed to list models: {e}[/red]")
        logger.exception("Failed to list models")
        raise typer.Exit(1)


@app.command()
def info(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Display configuration and system information."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            config = Config(_env_file=config_file)
        else:
            config = Config()

        # Display configuration
        console.print("[bold green]Configuration Information[/bold green]")

        # Data configuration
        console.print("\n[bold yellow]Data Configuration[/bold yellow]")
        console.print(f"  Input path: {config.data.input_path}")
        console.print(f"  Input format: {config.data.input_format.value}")
        console.print(f"  Text column: {config.data.text_column}")
        console.print(f"  Label column: {config.data.label_column}")

        # Preprocessing configuration
        console.print("\n[bold yellow]Preprocessing Configuration[/bold yellow]")
        console.print(f"  Max length: {config.preprocessing.max_length}")
        console.print(f"  Min length: {config.preprocessing.min_length}")
        console.print(f"  Remove HTML: {config.preprocessing.remove_html}")
        console.print(f"  Lowercase: {config.preprocessing.lowercase}")
        console.print(f"  Label mapping: {config.preprocessing.label_mapping}")

        # Training configuration
        console.print("\n[bold yellow]Training Configuration[/bold yellow]")
        console.print(f"  Backbone: {config.training.backbone.value}")
        console.print(f"  Batch size: {config.training.batch_size}")
        console.print(f"  Learning rate: {config.training.learning_rate}")
        console.print(f"  Epochs: {config.training.num_epochs}")
        console.print(f"  Random seed: {config.training.random_seed}")

        # Export configuration
        console.print("\n[bold yellow]Export Configuration[/bold yellow]")
        console.print(f"  Format: {config.export.format.value}")
        console.print(f"  ONNX opset: {config.export.onnx_opset_version}")
        console.print(f"  Validate export: {config.export.validate_export}")

        # Registry configuration
        console.print("\n[bold yellow]Registry Configuration[/bold yellow]")
        console.print(f"  Tracking URI: {config.registry.tracking_uri}")
        console.print(f"  Model name: {config.registry.model_name}")
        console.print(f"  Model stage: {config.registry.model_stage.value}")
        console.print(
            f"  Artifact storage: {'S3/MinIO' if config.registry.aws_access_key_id else 'Local'}"
        )

        # System information
        console.print("\n[bold yellow]System Information[/bold yellow]")
        import torch

        console.print(f"  PyTorch version: {torch.__version__}")
        console.print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"  CUDA version: {torch.version.cuda}")
            console.print(f"  GPU count: {torch.cuda.device_count()}")

        import mlflow

        console.print(f"  MLflow version: {mlflow.__version__}")

    except Exception as e:
        console.print(f"[red]Failed to display info: {e}[/red]")
        logger.exception("Failed to display info")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", "-m", help="Path to trained model"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Start the sentiment analysis API server."""
    setup_logging(log_level)

    try:
        import uvicorn
        import os
        import sys
        from pathlib import Path

        # Set environment variables for API configuration
        os.environ["API_HOST"] = host
        os.environ["API_PORT"] = str(port)
        os.environ["API_DEBUG"] = str(debug).lower()
        os.environ["API_RELOAD"] = str(reload).lower()

        # Set model paths if provided
        if model_path:
            os.environ["API_MODEL_PATH"] = str(model_path)
        else:
            # Use default paths
            current_dir = Path.cwd()
            os.environ.setdefault(
                "API_MODEL_PATH", str(current_dir / "outputs" / "model")
            )
            os.environ.setdefault(
                "API_PREPROCESSING_CONFIG_PATH",
                str(current_dir / "outputs" / "preprocessing_config.json"),
            )
            os.environ.setdefault(
                "API_LABEL_MAPPING_PATH", str(current_dir / "outputs" / "id2label.json")
            )

        console.print(f"🚀 Starting FinBERT Sentiment Analysis API server...")
        console.print(f"📍 Server: http://{host}:{port}")
        console.print(f"📚 Documentation: http://{host}:{port}/docs")
        console.print(f"🔍 Health check: http://{host}:{port}/api/v1/health")

        # Import the app
        from .api.app import app as fastapi_app

        # Run the server
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            workers=1,  # Always use 1 worker for model loading
            log_level=log_level.lower(),
            access_log=True,
        )

    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}", style="red")
        console.print(
            "Install API dependencies with: pip install fastapi uvicorn", style="yellow"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start API server: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def predict(
    text: str = typer.Argument(..., help="Text to analyze for sentiment"),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", "-m", help="Path to trained model"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Predict sentiment for a single text using the trained model."""
    setup_logging(log_level)

    try:
        import asyncio
        from .services.inference_service import SentimentInferenceService
        from .core.config import Config, APIConfig

        # Load configuration
        config = Config()
        if model_path:
            config.api.model_path = model_path

        console.print(f"🔮 Analyzing sentiment for: {text[:100]}...", style="bold blue")

        async def run_prediction():
            # Initialize inference service
            inference_service = SentimentInferenceService(config.api)
            await inference_service.initialize()

            # Predict sentiment
            result = await inference_service.predict_sentiment(text)

            # Display results
            table = Table(title="Sentiment Analysis Result")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Text", text[:50] + "..." if len(text) > 50 else text)
            table.add_row("Sentiment", str(result.label.value))
            table.add_row("Confidence", f"{result.confidence:.4f}")
            table.add_row("Positive Score", f"{result.scores.positive:.4f}")
            table.add_row("Negative Score", f"{result.scores.negative:.4f}")
            table.add_row("Neutral Score", f"{result.scores.neutral:.4f}")
            if result.processing_time_ms:
                table.add_row("Processing Time", f"{result.processing_time_ms:.2f}ms")

            console.print(table)

            # Cleanup
            await inference_service.cleanup()

        # Run the prediction
        asyncio.run(run_prediction())

    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Prediction failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_name_or_path: str = typer.Argument(
        ..., help="Model name (for pretrained) or path (for fine-tuned model)"
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data", "-d", help="Path to evaluation data file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for evaluation results"
    ),
    pretrained: bool = typer.Option(
        False, "--pretrained", help="Evaluate pretrained model (vs fine-tuned)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Evaluate a sentiment analysis model on the given dataset."""
    try:
        # Setup logging
        setup_logging(log_level)

        # Load configuration
        if config_file:
            config = Config(_env_file=config_file)
        else:
            config = Config()

        # Override config with CLI arguments
        if data_path:
            config.data.input_path = data_path
        if output_dir:
            config.output_dir = output_dir
        else:
            # Set default output directory for evaluation
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_safe_name = model_name_or_path.replace("/", "_").replace("\\", "_")
            config.output_dir = Path(f"evaluation_{model_safe_name}_{timestamp}")

        # Validate data path
        if not config.data.input_path or not config.data.input_path.exists():
            console.print(
                f"[red]Error: Evaluation data file not found: {config.data.input_path}[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[green]Starting model evaluation[/green]")
        console.print(f"  Model: {model_name_or_path}")
        console.print(f"  Model type: {'Pretrained' if pretrained else 'Fine-tuned'}")
        console.print(f"  Data path: {config.data.input_path}")
        console.print(f"  Output directory: {config.output_dir}")

        # Create output directory
        ensure_directory(config.output_dir)

        # Initialize components
        data_loader = DataLoader(config.data, config.preprocessing)
        evaluator = ModelEvaluator(config.preprocessing, config.training)

        # Load and validate data
        console.print("\n[yellow]Loading evaluation data...[/yellow]")
        articles = data_loader.load_data()

        if not data_loader.validate_data(articles):
            console.print("[red]Error: Data validation failed[/red]")
            raise typer.Exit(1)

        console.print(f"Loaded {len(articles)} articles for evaluation")

        # Run evaluation
        console.print(
            f"\n[yellow]Evaluating {'pretrained' if pretrained else 'fine-tuned'} model...[/yellow]"
        )

        if pretrained:
            metrics = evaluator.evaluate_pretrained_model(
                model_name_or_path=model_name_or_path,
                articles=articles,
                output_dir=config.output_dir,
            )
        else:
            model_path = Path(model_name_or_path)
            if not model_path.exists():
                console.print(f"[red]Error: Model path not found: {model_path}[/red]")
                raise typer.Exit(1)

            metrics = evaluator.evaluate_finetuned_model(
                model_path=model_path,
                articles=articles,
                output_dir=config.output_dir,
            )

        # Display results
        console.print(f"\n[green]Evaluation completed successfully![/green]")
        console.print(f"  Results saved to: {config.output_dir}")

        # Create and display results table
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
        table.add_row("F1 Macro", f"{metrics.f1_macro:.4f}")
        table.add_row("F1 Weighted", f"{metrics.f1_weighted:.4f}")
        table.add_row("Precision Macro", f"{metrics.precision_macro:.4f}")
        table.add_row("Recall Macro", f"{metrics.recall_macro:.4f}")
        table.add_row("Runtime (seconds)", f"{metrics.runtime_seconds:.2f}")
        table.add_row("Samples/second", f"{metrics.samples_per_second:.2f}")

        console.print(table)

        # Display per-class metrics
        if metrics.per_class_metrics:
            console.print(f"\n[bold]Per-Class Metrics:[/bold]")
            class_table = Table()
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Precision", style="green")
            class_table.add_column("Recall", style="green")
            class_table.add_column("F1-Score", style="green")
            class_table.add_column("Support", style="yellow")

            for class_name, class_metrics in metrics.per_class_metrics.items():
                class_table.add_row(
                    class_name,
                    f"{class_metrics.get('precision', 0.0):.4f}",
                    f"{class_metrics.get('recall', 0.0):.4f}",
                    f"{class_metrics.get('f1-score', 0.0):.4f}",
                    str(int(class_metrics.get("support", 0))),
                )

            console.print(class_table)

    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        logger.exception("Evaluation failed")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
