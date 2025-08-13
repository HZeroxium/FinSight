# interfaces/message_broker.py

"""
Message broker interface for async communication.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable


class MessageBrokerError(Exception):
    """Base exception for message broker operations."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MessageBroker(ABC):
    """Abstract base class for message brokers."""

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to message broker.

        Raises:
            MessageBrokerError: When connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to message broker.
        """
        pass

    @abstractmethod
    async def publish(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        persistent: bool = True,
    ) -> bool:
        """
        Publish message to exchange with routing key.

        Args:
            exchange: Exchange name
            routing_key: Routing key
            message: Message payload
            persistent: Make message persistent

        Returns:
            bool: True if successful

        Raises:
            MessageBrokerError: When publish fails
        """
        pass

    @abstractmethod
    async def consume(
        self,
        queue: str,
        callback: Callable[[Dict[str, Any]], None],
        auto_ack: bool = False,
    ) -> None:
        """
        Consume messages from queue.

        Args:
            queue: Queue name
            callback: Message handler function
            auto_ack: Auto-acknowledge messages

        Raises:
            MessageBrokerError: When consume setup fails
        """
        pass

    @abstractmethod
    async def declare_exchange(
        self, exchange: str, exchange_type: str = "topic", durable: bool = True
    ) -> None:
        """
        Declare exchange.

        Args:
            exchange: Exchange name
            exchange_type: Exchange type (topic, direct, fanout, headers)
            durable: Make exchange durable
        """
        pass

    @abstractmethod
    async def declare_queue(
        self,
        queue: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
    ) -> None:
        """
        Declare queue.

        Args:
            queue: Queue name
            durable: Make queue durable
            exclusive: Make queue exclusive
            auto_delete: Auto-delete queue when not used
        """
        pass

    @abstractmethod
    async def bind_queue(self, queue: str, exchange: str, routing_key: str) -> None:
        """
        Bind queue to exchange with routing key.

        Args:
            queue: Queue name
            exchange: Exchange name
            routing_key: Routing key pattern
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check broker health.

        Returns:
            bool: True if healthy
        """
        pass
