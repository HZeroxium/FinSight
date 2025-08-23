# services/triton_manager.py

"""Triton Inference Server management with Docker automation (CPU-friendly for Docker Desktop)."""

import asyncio
import os
import platform
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any

import docker
import httpx
import requests
from docker.errors import DockerException, NotFound

from ..core.config import TritonConfig
from ..core.enums import ServerStatus, ModelStatus
from common.logger.logger_factory import LoggerFactory, LoggerType

logger = LoggerFactory.get_logger(
    name="triton_manager",
    logger_type=LoggerType.STANDARD,
    log_file="logs/triton_manager.log",
)


class TritonServerError(Exception):
    """Base exception for Triton server operations."""

    pass


class TritonServerManager:
    """Manages Triton Inference Server Docker container lifecycle."""

    def __init__(self, config: TritonConfig):
        """Initialize Triton server manager.

        Args:
            config: Triton configuration
        """
        self.config = config
        self.docker_client = None
        self.container = None
        self.startup_time = None
        self._status = ServerStatus.STOPPED

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_server()

    @property
    def status(self) -> ServerStatus:
        """Get current server status."""
        return self._status

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        if self.startup_time is None:
            return 0.0
        return time.time() - self.startup_time

    async def start_server(self) -> None:
        """Start Triton server in Docker container."""
        try:
            logger.info("Starting Triton Inference Server...")
            self._status = ServerStatus.STARTING

            # Initialize Docker client
            self._init_docker_client()

            # Stop existing container if running
            await self._stop_existing_container()

            # Validate model repository (+ force CPU if needed)
            self._validate_model_repository()

            # Create and start container
            await self._create_container()
            await self._start_container()

            # Wait for server to be ready
            await self._wait_for_server_ready()

            # Validate model is loaded
            await self._wait_for_model_ready()

            self._status = ServerStatus.RUNNING
            self.startup_time = time.time()

            logger.info(
                f"Triton server started successfully on ports {self.config.http_port}/{self.config.grpc_port}"
            )

        except Exception as e:
            self._status = ServerStatus.ERROR
            logger.error(f"Failed to start Triton server: {e}")
            await self._cleanup_on_error()
            raise TritonServerError(f"Failed to start Triton server: {e}")

    async def stop_server(self) -> None:
        """Stop Triton server container."""
        try:
            logger.info("Stopping Triton Inference Server...")
            # FIX: don't call enum like a function
            self._status = ServerStatus.STOPPING

            if self.container:
                try:
                    self.container.stop(timeout=10)
                    self.container.remove()
                    logger.info("Triton container stopped and removed")
                except Exception as e:
                    logger.warning(f"Error stopping container: {e}")

            if self.docker_client:
                try:
                    self.docker_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Docker client: {e}")

            self.container = None
            self.docker_client = None
            self.startup_time = None
            self._status = ServerStatus.STOPPED

            logger.info("Triton server stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping Triton server: {e}")
            self._status = ServerStatus.ERROR
            raise TritonServerError(f"Failed to stop Triton server: {e}")

    async def restart_server(self) -> None:
        """Restart Triton server."""
        logger.info("Restarting Triton server...")
        await self.stop_server()
        await asyncio.sleep(2)  # Brief pause between stop and start
        await self.start_server()

    async def get_server_health(self) -> Dict[str, Any]:
        """Get server health status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{self.config.host}:{self.config.http_port}/v2/health/ready",
                    timeout=5.0,
                )
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code,
                    "uptime_seconds": self.uptime_seconds,
                }
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "uptime_seconds": self.uptime_seconds,
            }

    async def get_model_status(self, model_name: str) -> ModelStatus:
        """Get model status from Triton.

        Uses /v2/models/{name}/ready per Triton API.
        200 -> READY, otherwise -> UNAVAILABLE
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://{self.config.host}:{self.config.http_port}/v2/models/{model_name}/ready",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    return ModelStatus.READY
                return ModelStatus.UNAVAILABLE
        except Exception as e:
            logger.warning(f"Failed to get model status: {e}")
            return ModelStatus.UNAVAILABLE

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information.

        Use /v2/models/{name}/config for full configuration.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://{self.config.host}:{self.config.http_port}/v2/models/{model_name}/config",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    return resp.json()
                return None
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None

    def _init_docker_client(self) -> None:
        """Initialize Docker client with Windows-friendly fallback."""
        try:
            base_url = os.environ.get("DOCKER_HOST")
            is_windows = platform.system().lower().startswith("win")

            # On Windows, prefer named pipe. If DOCKER_HOST is mis-set to http+docker
            # (symptom of requests 2.32 incompatibility), coerce to npipe.
            if is_windows:
                if not base_url or base_url.startswith("http+docker://"):
                    base_url = "npipe:////./pipe/docker_engine"

            if base_url:
                self.docker_client = docker.DockerClient(base_url=base_url)
            else:
                self.docker_client = docker.from_env()

            # Test Docker connection
            self.docker_client.ping()

            try:
                docker_ver = getattr(docker, "__version__", "unknown")
                requests_ver = getattr(requests, "__version__", "unknown")
                logger.debug(
                    f"Docker SDK={docker_ver}, requests={requests_ver}, base_url={base_url or 'from_env'}"
                )
            except Exception:
                pass

            logger.debug("Docker client initialized successfully")

        except DockerException as e:
            # Give precise hint for the well-known http+docker scheme issue
            msg = str(e)
            if "http+docker" in msg:
                raise TritonServerError(
                    "Failed to connect to Docker (unsupported 'http+docker' scheme). "
                    "This is commonly caused by an incompatibility between docker SDK and requests 2.32. "
                    "Please upgrade the Docker SDK: `pip install -U docker>=7.1.0` "
                    "or pin requests to `<2.32`. See docker-py issue #3256 for details."
                )
            raise TritonServerError(f"Failed to connect to Docker: {e}")

        except Exception as e:
            raise TritonServerError(f"Failed to connect to Docker: {e}")

    async def _stop_existing_container(self) -> None:
        """Stop and remove existing container with the same name."""
        try:
            existing = self.docker_client.containers.get(self.config.container_name)
            logger.info(f"Stopping existing container: {self.config.container_name}")
            existing.stop(timeout=10)
            existing.remove()

        except NotFound:
            # Container doesn't exist, which is fine
            pass
        except Exception as e:
            logger.warning(f"Error stopping existing container: {e}")

    def _validate_model_repository(self) -> None:
        """Validate model repository exists and has required files.

        Also ensure the model config is CPU-friendly for Docker Desktop.
        """
        if not self.config.model_repository.exists():
            raise TritonServerError(
                f"Model repository not found: {self.config.model_repository}"
            )

        model_path = self.config.model_repository / self.config.model_name
        if not model_path.exists():
            raise TritonServerError(f"Model directory not found: {model_path}")

        config_file = model_path / "config.pbtxt"
        if not config_file.exists():
            raise TritonServerError(f"Model config not found: {config_file}")

        # Ensure CPU instance group and remove GPU accelerators if present
        try:
            self._ensure_cpu_model_config(config_file)
        except Exception as e:
            logger.warning(f"Could not normalize config.pbtxt to CPU: {e}")

        logger.debug(f"Model repository validated: {self.config.model_repository}")

    def _ensure_cpu_model_config(self, config_path: Path) -> None:
        """Force model config to use CPU instance group and strip GPU accelerators."""
        text = config_path.read_text(encoding="utf-8")
        original = text

        # Remove any gpu_execution_accelerator blocks
        text = re.sub(
            r"gpu_execution_accelerator\s*\{[^}]*\}",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Replace any instance_group {...} to KIND_CPU
        def _repl_instance_group(match: re.Match) -> str:
            return "instance_group {\n  count: 1\n  kind: KIND_CPU\n}"

        text = re.sub(
            r"instance_group\s*\{[^}]*\}",
            _repl_instance_group,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # If no instance_group at all, append one
        if "instance_group" not in text:
            text = (
                text.rstrip()
                + "\n\ninstance_group {\n  count: 1\n  kind: KIND_CPU\n}\n"
            )

        # Ensure backend/platform is ONNX Runtime or compatible
        if "backend:" not in text and "platform:" not in text:
            # default to ONNX Runtime platform if missing
            text = f'platform: "onnxruntime_onnx"\n' + text

        if text != original:
            config_path.write_text(text, encoding="utf-8")
            logger.info(
                f"Rewrote {config_path} to enforce CPU instance group for Docker Desktop."
            )

    async def _create_container(self) -> None:
        """Create Triton Docker container."""
        try:
            # Prepare environment variables (disable GPU if not enabled)
            environment = {
                "CUDA_VISIBLE_DEVICES": "0" if self.config.gpu_enabled else "",
                "TRITON_LOG_VERBOSE": "1",
            }

            # Prepare port mappings
            ports = {
                "8000/tcp": self.config.http_port,
                "8001/tcp": self.config.grpc_port,
                "8002/tcp": self.config.metrics_port,
            }

            # Prepare volumes
            volumes = {
                str(self.config.model_repository.absolute()): {
                    "bind": "/models",
                    "mode": "ro",
                }
            }

            # Docker run arguments
            docker_args = {
                "image": self.config.docker_image,
                "name": self.config.container_name,
                "ports": ports,
                "volumes": volumes,
                "environment": environment,
                "detach": True,
                "command": [
                    "tritonserver",
                    "--model-repository=/models",
                    "--log-verbose=1",
                    # Keep strict readiness (default true) so server_ready==200 means models ready
                    # "--strict-readiness=true",
                ],
            }

            # Add GPU support if enabled
            if self.config.gpu_enabled:
                docker_args["device_requests"] = [
                    docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])
                ]

            logger.debug(f"Creating container with args: {docker_args}")

            self.container = self.docker_client.containers.create(**docker_args)
            logger.debug(f"Container created: {self.container.id}")

        except Exception as e:
            raise TritonServerError(f"Failed to create container: {e}")

    async def _start_container(self) -> None:
        """Start the created container."""
        try:
            self.container.start()
            logger.debug(f"Container started: {self.container.id}")

        except Exception as e:
            raise TritonServerError(f"Failed to start container: {e}")

    async def _wait_for_server_ready(self) -> None:
        """Wait for Triton server to be ready."""
        logger.info("Waiting for Triton server to be ready...")

        start_time = time.time()
        timeout = self.config.startup_timeout

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://{self.config.host}:{self.config.http_port}/v2/health/ready",
                        timeout=5.0,
                    )
                    # By default strict_readiness=true -> 200 only if all models loaded
                    if response.status_code == 200:
                        logger.info("Triton server is ready")
                        return
            except Exception:
                pass  # Server not ready yet

            await asyncio.sleep(self.config.health_check_interval)

        raise TritonServerError(
            f"Triton server failed to start within {timeout} seconds"
        )

    async def _wait_for_model_ready(self) -> None:
        """Wait for model to be loaded and ready."""
        logger.info(f"Waiting for model '{self.config.model_name}' to be ready...")

        start_time = time.time()
        timeout = 60  # Model loading timeout

        while time.time() - start_time < timeout:
            status = await self.get_model_status(self.config.model_name)

            if status == ModelStatus.READY:
                logger.info(f"Model '{self.config.model_name}' is ready")
                return
            elif status == ModelStatus.UNAVAILABLE:
                logger.warning(f"Model '{self.config.model_name}' is unavailable")

            await asyncio.sleep(2)

        raise TritonServerError(
            f"Model '{self.config.model_name}' failed to load within {timeout} seconds"
        )

    async def _cleanup_on_error(self) -> None:
        """Clean up resources when an error occurs during startup."""
        try:
            if self.container:
                try:
                    self.container.stop(timeout=5)
                    self.container.remove()
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")

            if self.docker_client:
                try:
                    self.docker_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Docker client during cleanup: {e}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        finally:
            self.container = None
            self.docker_client = None
