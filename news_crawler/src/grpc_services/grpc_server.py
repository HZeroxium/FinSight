# grpc_services/grpc_server.py

"""
gRPC server implementation for news service.

This module provides the gRPC server setup and configuration,
running alongside the FastAPI server to provide dual protocol support.
"""

from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from grpc import aio

from ..services.news_service import NewsService
from ..grpc_services.news_grpc_service import create_news_servicer
from ..core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize logger
logger = LoggerFactory.get_logger(
    name="grpc-server",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/grpc_server.log",
)


class GrpcServer:
    """gRPC server for news service."""

    def __init__(
        self,
        news_service: NewsService,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
        max_receive_message_length: int = 4 * 1024 * 1024,  # 4MB
        max_send_message_length: int = 4 * 1024 * 1024,  # 4MB
    ):
        """
        Initialize gRPC server.

        Args:
            news_service: Core news service instance
            host: Server host address
            port: Server port
            max_workers: Maximum worker threads
            max_receive_message_length: Maximum receive message size
            max_send_message_length: Maximum send message size
        """
        self.news_service = news_service
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_receive_message_length = max_receive_message_length
        self.max_send_message_length = max_send_message_length

        self.server: Optional[aio.Server] = None
        self._running = False

        logger.info(f"GrpcServer initialized on {host}:{port}")

    async def start(self) -> None:
        """Start the gRPC server."""
        try:
            logger.info("Starting gRPC server...")

            # Import generated gRPC code
            try:
                from ..grpc_generated.news_service_pb2_grpc import (
                    add_NewsServiceServicer_to_server,
                )
            except ImportError:
                logger.error(
                    "Failed to import generated gRPC code. Please run generate_grpc_code.py first."
                )
                raise

            # Create server with options
            options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ("grpc.max_receive_message_length", self.max_receive_message_length),
                ("grpc.max_send_message_length", self.max_send_message_length),
            ]

            self.server = aio.server(
                ThreadPoolExecutor(max_workers=self.max_workers), options=options
            )

            # Create and add servicer
            servicer = create_news_servicer(self.news_service)
            add_NewsServiceServicer_to_server(servicer, self.server)

            # Add server port
            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)

            # Start server
            await self.server.start()
            self._running = True

            logger.info(f"gRPC server started successfully on {listen_addr}")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server."""
        try:
            if self.server and self._running:
                logger.info("Stopping gRPC server...")

                await self.server.stop(grace_period)
                self._running = False

                logger.info("gRPC server stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server and self._running:
            await self.server.wait_for_termination()

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    async def health_check(self) -> bool:
        """Perform health check on gRPC server."""
        try:
            return self.is_running()
        except Exception as e:
            logger.error(f"gRPC server health check failed: {e}")
            return False


async def create_grpc_server(
    news_service: NewsService,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> GrpcServer:
    """
    Create and configure gRPC server.

    Args:
        news_service: Core news service instance
        host: Optional server host (defaults to config)
        port: Optional server port (defaults to config)

    Returns:
        Configured GrpcServer instance
    """
    try:
        # Use provided values or fallback to config
        server_host = host or getattr(settings, "grpc_host", "0.0.0.0")
        server_port = port or getattr(settings, "grpc_port", 50051)

        # Create server instance
        grpc_server = GrpcServer(
            news_service=news_service,
            host=server_host,
            port=server_port,
            max_workers=getattr(settings, "grpc_max_workers", 10),
        )

        logger.info(f"gRPC server created for {server_host}:{server_port}")
        return grpc_server

    except Exception as e:
        logger.error(f"Failed to create gRPC server: {e}")
        raise


async def run_grpc_server_standalone(news_service: NewsService) -> None:
    """
    Run gRPC server as standalone service.

    Args:
        news_service: Core news service instance
    """
    try:
        logger.info("Starting standalone gRPC server...")

        # Create and start server
        grpc_server = await create_grpc_server(news_service)
        await grpc_server.start()

        # Wait for termination
        try:
            await grpc_server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            await grpc_server.stop()

    except Exception as e:
        logger.error(f"Standalone gRPC server error: {e}")
        raise
