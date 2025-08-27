# grpc_services/__init__.py

"""
gRPC services module for news crawler.

This module provides gRPC service implementations and server setup
for high-performance, strongly-typed API access to news data.
"""

from .grpc_server import (GrpcServer, create_grpc_server,
                          run_grpc_server_standalone)
from .news_grpc_service import NewsGrpcService, create_news_servicer

__all__ = [
    "GrpcServer",
    "create_grpc_server",
    "run_grpc_server_standalone",
    "NewsGrpcService",
    "create_news_servicer",
]
