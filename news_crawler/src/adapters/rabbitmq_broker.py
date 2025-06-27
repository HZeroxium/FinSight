# adapters/rabbitmq_broker.py

"""
RabbitMQ message broker implementation.
"""

import json
import asyncio
from typing import Dict, Any, Optional, Callable

import aio_pika
from aio_pika import (
    connect_robust,
    RobustConnection,
    RobustChannel,
    Message,
    ExchangeType,
)

from ..interfaces.message_broker import MessageBroker, MessageBrokerError
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="rabbitmq-broker", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class RabbitMQBroker(MessageBroker):
    """
    RabbitMQ message broker implementation using aio-pika.
    """

    def __init__(self, connection_url: str):
        """
        Initialize RabbitMQ broker.

        Args:
            connection_url: RabbitMQ connection URL
        """
        self.connection_url = connection_url
        self._connection: Optional[RobustConnection] = None
        self._channel: Optional[RobustChannel] = None
        self._consumers = {}
        logger.info("RabbitMQ broker initialized")

    async def connect(self) -> None:
        """Establish robust connection and channel."""
        try:
            logger.info("Connecting to RabbitMQ...")
            self._connection = await connect_robust(self.connection_url)
            self._channel = await self._connection.channel()

            # Set QoS for fair message distribution
            await self._channel.set_qos(prefetch_count=10)

            logger.info("Successfully connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise MessageBrokerError(f"Connection failed: {str(e)}")

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        try:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")

    async def publish(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        persistent: bool = True,
    ) -> bool:
        """Publish message to exchange with routing key."""
        try:
            if not self._channel:
                await self.connect()

            # Serialize message
            message_body = json.dumps(message, default=str).encode("utf-8")

            # Create message
            aio_message = Message(
                message_body,
                content_type="application/json",
                delivery_mode=2 if persistent else 1,  # 2 = persistent
            )

            # Get exchange - if it doesn't exist, create it first
            try:
                exchange_obj = await self._channel.get_exchange(exchange)
            except Exception:
                logger.info(f"Exchange {exchange} not found, creating it...")
                await self.create_exchange(exchange)
                exchange_obj = await self._channel.get_exchange(exchange)

            # Publish message
            await exchange_obj.publish(aio_message, routing_key=routing_key)

            logger.debug(f"Published message to {exchange}/{routing_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            raise MessageBrokerError(f"Publish failed: {str(e)}")

    async def consume(
        self,
        queue: str,
        callback: Callable[[Dict[str, Any]], None],
        auto_ack: bool = False,
    ) -> None:
        """Consume messages from queue."""
        try:
            if not self._channel:
                await self.connect()

            # Get queue
            queue_obj = await self._channel.get_queue(queue)

            async def message_handler(message: aio_pika.abc.AbstractIncomingMessage):
                async with message.process(ignore_processed=True):
                    try:
                        # Deserialize message
                        message_data = json.loads(message.body.decode("utf-8"))

                        # Call callback - handle both sync and async callbacks
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message_data)
                        else:
                            callback(message_data)

                        if not auto_ack:
                            await message.ack()

                        logger.debug(f"Processed message from {queue}")

                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        if not auto_ack:
                            message.nack(requeue=True)

            # Start consuming
            consumer_tag = await queue_obj.consume(message_handler, no_ack=auto_ack)
            self._consumers[queue] = consumer_tag

            logger.info(f"Started consuming from queue: {queue}")

        except Exception as e:
            logger.error(f"Failed to start consumer: {str(e)}")
            raise MessageBrokerError(f"Consumer setup failed: {str(e)}")

    async def declare_exchange(
        self, exchange: str, exchange_type: str = "topic", durable: bool = True
    ) -> None:
        """Declare exchange."""
        try:
            if not self._channel:
                await self.connect()

            exchange_type_map = {
                "topic": ExchangeType.TOPIC,
                "direct": ExchangeType.DIRECT,
                "fanout": ExchangeType.FANOUT,
                "headers": ExchangeType.HEADERS,
            }

            await self._channel.declare_exchange(
                exchange,
                exchange_type_map.get(exchange_type, ExchangeType.TOPIC),
                durable=durable,
            )

            logger.debug(f"Declared exchange: {exchange} ({exchange_type})")

        except Exception as e:
            logger.error(f"Failed to declare exchange: {str(e)}")
            raise MessageBrokerError(f"Exchange declaration failed: {str(e)}")

    async def declare_queue(
        self,
        queue: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
    ) -> None:
        """Declare queue."""
        try:
            if not self._channel:
                await self.connect()

            await self._channel.declare_queue(
                queue, durable=durable, exclusive=exclusive, auto_delete=auto_delete
            )

            logger.debug(f"Declared queue: {queue}")

        except Exception as e:
            logger.error(f"Failed to declare queue: {str(e)}")
            raise MessageBrokerError(f"Queue declaration failed: {str(e)}")

    async def bind_queue(self, queue: str, exchange: str, routing_key: str) -> None:
        """Bind queue to exchange with routing key."""
        try:
            if not self._channel:
                await self.connect()

            queue_obj = await self._channel.get_queue(queue)
            exchange_obj = await self._channel.get_exchange(exchange)

            await queue_obj.bind(exchange_obj, routing_key)

            logger.debug(
                f"Bound queue {queue} to exchange {exchange} with key {routing_key}"
            )

        except Exception as e:
            logger.error(f"Failed to bind queue: {str(e)}")
            raise MessageBrokerError(f"Queue binding failed: {str(e)}")

    async def health_check(self) -> bool:
        """Check RabbitMQ health."""
        try:
            if not self._connection or self._connection.is_closed:
                await self.connect()

            # Simple health check - declare a temporary queue
            temp_queue = await self._channel.declare_queue(
                exclusive=True, auto_delete=True
            )
            await temp_queue.delete()

            logger.debug("RabbitMQ health check passed")
            return True

        except Exception as e:
            logger.error(f"RabbitMQ health check failed: {str(e)}")
            return False

    async def create_exchange(
        self, exchange_name: str, exchange_type: str = "topic"
    ) -> None:
        """
        Create an exchange if it doesn't exist.

        Args:
            exchange_name: Name of the exchange
            exchange_type: Type of exchange (topic, direct, fanout, headers)
        """
        try:
            if not self._channel:
                await self.connect()

            exchange_type_map = {
                "topic": ExchangeType.TOPIC,
                "direct": ExchangeType.DIRECT,
                "fanout": ExchangeType.FANOUT,
                "headers": ExchangeType.HEADERS,
            }

            await self._channel.declare_exchange(
                exchange_name,
                exchange_type_map.get(exchange_type, ExchangeType.TOPIC),
                durable=True,
            )

            logger.info(f"Exchange '{exchange_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to create exchange '{exchange_name}': {str(e)}")
            raise MessageBrokerError(f"Exchange creation failed: {str(e)}")
