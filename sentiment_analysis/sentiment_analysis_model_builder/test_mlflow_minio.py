#!/usr/bin/env python3
"""
Test script for MLflow + MinIO integration.

This script verifies that the MLflow tracking server and MinIO artifact storage
are properly configured and working together.
"""

import os
import json
import time
from pathlib import Path

import mlflow
import requests
from loguru import logger


def test_mlflow_connection():
    """Test MLflow tracking server connection."""
    try:
        # Set up environment variables first (like the trainer does)
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "s3://mlflow-artifacts/"

        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")

        # Try to create a test experiment
        experiment_name = "test-connection"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log some test metrics
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_param("test_param", "test_value")

            # Create and log a test artifact
            test_file = Path("test_artifact.txt")
            test_file.write_text("This is a test artifact for MLflow+MinIO integration")
            mlflow.log_artifact(str(test_file))
            test_file.unlink()  # Clean up local file

        logger.info(f"✅ MLflow connection test passed. Run ID: {run.info.run_id}")
        return True

    except Exception as e:
        logger.error(f"❌ MLflow connection test failed: {e}")
        return False


def test_minio_connection():
    """Test MinIO server connection."""
    try:
        # Test MinIO health endpoint
        response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
        response.raise_for_status()

        logger.info("✅ MinIO health check passed")
        return True

    except Exception as e:
        logger.error(f"❌ MinIO connection test failed: {e}")
        return False


def test_s3_integration():
    """Test S3/MinIO integration with MLflow."""
    try:
        # Set environment variables for S3 integration
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

        # Set MLflow to use S3 artifact store
        mlflow.set_tracking_uri("http://localhost:5000")

        experiment_name = "test-s3-integration"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Create a test artifact
            test_data = {
                "model_name": "test-model",
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.97,
            }

            test_file = Path("test_model_metrics.json")
            test_file.write_text(json.dumps(test_data, indent=2))

            # Log artifact to S3/MinIO
            mlflow.log_artifact(str(test_file), "model_artifacts")
            test_file.unlink()  # Clean up local file

        logger.info(f"✅ S3/MinIO integration test passed. Run ID: {run.info.run_id}")
        return True

    except Exception as e:
        logger.error(f"❌ S3/MinIO integration test failed: {e}")
        return False


def check_services_health():
    """Check if required services are running."""
    services = {
        "MLflow": "http://localhost:5000/health",
        "MinIO": "http://localhost:9000/minio/health/live",
    }

    all_healthy = True

    for service, endpoint in services.items():
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service} is healthy")
            else:
                logger.warning(f"⚠️ {service} returned status {response.status_code}")
                all_healthy = False
        except Exception as e:
            logger.error(f"❌ {service} is not accessible: {e}")
            all_healthy = False

    return all_healthy


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting MLflow + MinIO integration tests...")

    # Wait a moment for services to be ready
    logger.info("⏳ Waiting for services to be ready...")
    time.sleep(5)

    # Check services health
    if not check_services_health():
        logger.error(
            "❌ Some services are not healthy. Please check Docker containers."
        )
        return False

    # Run tests
    tests = [
        ("MLflow Connection", test_mlflow_connection),
        ("MinIO Connection", test_minio_connection),
        ("S3 Integration", test_s3_integration),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"🧪 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))

        if result:
            logger.info(f"✅ {test_name} test passed")
        else:
            logger.error(f"❌ {test_name} test failed")

        time.sleep(1)  # Brief pause between tests

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    logger.info(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        logger.info(
            "🎉 All tests passed! MLflow + MinIO integration is working correctly."
        )
        return True
    else:
        logger.error(
            "❌ Some tests failed. Please check the configuration and services."
        )
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
