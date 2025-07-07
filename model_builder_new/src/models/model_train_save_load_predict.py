# models/model_train_save_load_predict.py

"""
Debug script to test train -> save -> load -> predict pipeline for all model types.
This script reproduces the service behavior to identify scaling/prediction issues.
"""

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ..logger.logger_factory import LoggerFactory, LogLevel
from .model_facade import ModelFacade
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import ModelConfig
from ..data.data_loader import CSVDataLoader
from ..data.feature_engineering import BasicFeatureEngineering


class ModelDebugger:
    """Debug class to test model training, saving, loading, and prediction pipeline"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelDebugger")
        self.facade = ModelFacade()
        self.data_loader = CSVDataLoader()

        # Test configuration
        self.symbol = "BTCUSDT"
        self.timeframe = TimeFrame.DAY_1
        self.models_to_test = [
            ModelType.PATCHTST,
            ModelType.PATCHTSMIXER,
            ModelType.PYTORCH_TRANSFORMER,
        ]

    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        try:
            # Try to load from finetuning data directory first
            data_path = Path("../../finetuning/data/BTCUSDT_1d.csv")
            if data_path.exists():
                self.logger.info(f"Loading data from {data_path}")
                df = pd.read_csv(data_path)
            else:
                # Fallback to data loader
                df = self.data_loader.load_data(self.symbol, self.timeframe)

            # Ensure timestamp column is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            self.logger.info(f"Loaded {len(df)} rows of data")
            self.logger.info(f"Data shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            self.logger.info(
                f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
            )
            self.logger.info(
                f"Close price range: {df['close'].min():.2f} to {df['close'].max():.2f}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise

    def prepare_data_splits(self, df: pd.DataFrame) -> tuple:
        """Prepare train/val/test splits"""
        try:
            # Use chronological split
            total_samples = len(df)
            train_end = int(total_samples * 0.7)
            val_end = int(total_samples * 0.85)

            train_data = df.iloc[:train_end].copy()
            val_data = df.iloc[train_end:val_end].copy()
            test_data = df.iloc[val_end:].copy()

            self.logger.info(
                f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
            )

            # Print some sample data for debugging
            self.logger.info(
                f"Last few training samples close prices: {train_data['close'].tail().tolist()}"
            )
            self.logger.info(
                f"First few test samples close prices: {test_data['close'].head().tolist()}"
            )

            return train_data, val_data, test_data

        except Exception as e:
            self.logger.error(f"Error preparing data splits: {e}")
            raise

    def create_model_config(self, model_type: ModelType) -> ModelConfig:
        """Create model configuration based on model type"""
        base_config = {
            "context_length": 32,
            "prediction_length": 1,
            "target_column": "close",
            "feature_columns": ["open", "high", "low", "close", "volume"],
            "batch_size": 16,
            "num_epochs": 2,  # Quick training for testing
            "learning_rate": 0.001,
        }

        # Model-specific adjustments
        if model_type == ModelType.PYTORCH_TRANSFORMER:
            base_config.update(
                {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1}
            )
        elif model_type in [ModelType.PATCHTST, ModelType.PATCHTSMIXER]:
            base_config.update(
                {
                    "d_model": 64,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 2,
                    "dropout": 0.1,
                }
            )

        return ModelConfig(**base_config)

    def test_single_model(
        self,
        model_type: ModelType,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Test a single model through the complete pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Testing model: {model_type}")
        self.logger.info(f"{'='*60}")

        results = {
            "model_type": model_type,
            "success": False,
            "error": None,
            "train_results": None,
            "evaluation_results": None,
            "prediction_results": None,
            "scaling_debug": {},
        }

        try:
            # 1. Create feature engineering
            feature_engineering = BasicFeatureEngineering(
                feature_columns=["open", "high", "low", "close", "volume"],
                add_technical_indicators=False,  # Keep simple for debugging
                normalize_features=True,
            )

            # 2. Create model config
            config = self.create_model_config(model_type)

            # 3. Train model
            self.logger.info("Starting model training...")
            train_results = self.facade.train_model(
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_type=model_type,
                train_data=train_data,
                val_data=val_data,
                feature_engineering=feature_engineering,
                config=config,
            )

            results["train_results"] = train_results
            self.logger.info(f"Training results: {train_results}")

            if not train_results.get("success", False):
                results["error"] = (
                    f"Training failed: {train_results.get('error', 'Unknown error')}"
                )
                return results

            # 4. Check if model exists
            model_exists = self.facade.model_exists(
                self.symbol, self.timeframe, model_type
            )
            self.logger.info(f"Model exists after training: {model_exists}")

            if not model_exists:
                results["error"] = "Model doesn't exist after training"
                return results

            # 5. Evaluate model on test data
            self.logger.info("Starting model evaluation...")
            eval_results = self.facade.evaluate_model(
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_type=model_type,
                test_data=test_data,
            )

            results["evaluation_results"] = eval_results
            # self.logger.info(f"Evaluation results: {eval_results}")

            # 6. Test prediction with recent data (last 50 samples for context)
            self.logger.info("Testing prediction...")
            recent_data = test_data.tail(50).copy()  # Use more data for context

            # Debug: Print recent data info
            self.logger.info(f"Recent data shape: {recent_data.shape}")
            self.logger.info(
                f"Recent close prices: {recent_data['close'].tail(10).tolist()}"
            )

            # Single step prediction
            prediction_results = self.facade.predict(
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_type=model_type,
                recent_data=recent_data,
                n_steps=1,
            )

            results["prediction_results"] = prediction_results
            self.logger.info(f"Prediction results: {prediction_results}")

            # 7. Multi-step prediction for comparison
            multistep_results = self.facade.predict(
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_type=model_type,
                recent_data=recent_data,
                n_steps=5,
            )

            results["multistep_prediction"] = multistep_results
            self.logger.info(f"Multi-step prediction: {multistep_results}")

            # 8. Scaling debug - manually load model to inspect scalers
            try:
                model_instance = self.facade._load_model(
                    self.symbol, self.timeframe, model_type
                )
                if (
                    hasattr(model_instance, "target_scaler")
                    and model_instance.target_scaler is not None
                ):
                    scaler = model_instance.target_scaler

                    # Test scaler with some sample values
                    test_values = np.array([100000, 110000, 120000]).reshape(-1, 1)
                    scaled_values = scaler.transform(test_values)
                    inverse_values = scaler.inverse_transform(scaled_values)

                    results["scaling_debug"] = {
                        "scaler_mean": (
                            float(scaler.mean_[0]) if hasattr(scaler, "mean_") else None
                        ),
                        "scaler_scale": (
                            float(scaler.scale_[0])
                            if hasattr(scaler, "scale_")
                            else None
                        ),
                        "test_transform": {
                            "original": test_values.flatten().tolist(),
                            "scaled": scaled_values.flatten().tolist(),
                            "inverse": inverse_values.flatten().tolist(),
                        },
                    }

                    self.logger.info(f"Scaler debug info: {results['scaling_debug']}")

            except Exception as e:
                self.logger.warning(f"Could not debug scaler: {e}")

            results["success"] = True

        except Exception as e:
            error_msg = f"Error testing {model_type}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            results["error"] = error_msg

        return results

    def analyze_results(self, all_results: list):
        """Analyze and compare results across all models"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("RESULTS ANALYSIS")
        self.logger.info(f"{'='*60}")

        for result in all_results:
            model_type = result["model_type"]
            success = result["success"]

            self.logger.info(f"\nModel: {model_type}")
            self.logger.info(f"Success: {success}")

            if success:
                # Training info
                if result.get("train_results"):
                    train_loss = result["train_results"].get("eval_loss", "N/A")
                    self.logger.info(f"Training loss: {train_loss}")

                # Evaluation info
                if result.get("evaluation_results"):
                    eval_metrics = result["evaluation_results"].get("metrics", {})
                    mae = eval_metrics.get("mae", "N/A")
                    rmse = eval_metrics.get("rmse", "N/A")
                    self.logger.info(f"Evaluation MAE: {mae}, RMSE: {rmse}")

                # Prediction info
                if result.get("prediction_results"):
                    pred_result = result["prediction_results"]
                    if pred_result.get("success"):
                        predictions = pred_result.get("predictions", [])
                        current_price = pred_result.get("current_price", 0)
                        self.logger.info(f"Current price: {current_price}")
                        self.logger.info(f"Prediction: {predictions}")

                        if predictions:
                            pred_change_pct = (
                                (predictions[0] - current_price) / current_price
                            ) * 100
                            self.logger.info(
                                f"Predicted change: {pred_change_pct:.2f}%"
                            )

                # Scaling info
                if result.get("scaling_debug"):
                    scale_info = result["scaling_debug"]
                    self.logger.info(f"Scaler mean: {scale_info.get('scaler_mean')}")
                    self.logger.info(f"Scaler scale: {scale_info.get('scaler_scale')}")

                # Multi-step prediction info
                if result.get("multistep_prediction"):
                    multi_pred = result["multistep_prediction"]
                    if multi_pred.get("success"):
                        multi_predictions = multi_pred.get("predictions", [])
                        self.logger.info(f"Multi-step predictions: {multi_predictions}")

            else:
                self.logger.error(f"Error: {result.get('error', 'Unknown error')}")

    def run_debug_test(self):
        """Run the complete debug test"""
        self.logger.info("Starting model debug test...")

        try:
            # Load data
            df = self.load_test_data()

            # Prepare splits
            train_data, val_data, test_data = self.prepare_data_splits(df)

            # Test each model
            all_results = []
            for model_type in self.models_to_test:
                result = self.test_single_model(
                    model_type, train_data, val_data, test_data
                )
                all_results.append(result)

            # Analyze results
            self.analyze_results(all_results)

            return all_results

        except Exception as e:
            self.logger.error(f"Debug test failed: {e}")
            self.logger.error(traceback.format_exc())
            raise


def main():
    """Main function to run the debug test"""
    # Set up logging
    logger = LoggerFactory.get_logger("ModelDebugMain")
    logger.info("Starting model train/save/load/predict debug test")

    try:
        debugger = ModelDebugger()
        results = debugger.run_debug_test()

        logger.info("Debug test completed successfully")
        return results

    except Exception as e:
        logger.error(f"Debug test failed: {e}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()
