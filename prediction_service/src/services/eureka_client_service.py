# service/eureka_client_service.py

"""
Eureka Client Service for service discovery registration.
Handles registration and deregistration with Eureka server.
"""

import asyncio
import random
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional

import requests
from common.logger import LoggerFactory, LoggerType, LogLevel

from ..core.config import settings


class EurekaClientService:
    """
    Eureka Client Service for managing service registration with Eureka server.

    This service handles:
    - Service registration with Eureka server
    - Heartbeat management with retry logic
    - Service deregistration on shutdown
    - Health check URL management
    - Automatic re-registration after server restart
    """

    def __init__(self):
        """Initialize the Eureka client service."""
        self.logger = LoggerFactory.get_logger(
            name="eureka-client-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            console_level=LogLevel.INFO,
            use_colors=True,
            log_file=f"{settings.logs_dir}eureka_client.log",
        )

        self._is_registered: bool = False
        self._instance_id: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat: bool = False
        self._consecutive_heartbeat_failures: int = 0
        self._last_registration_time: float = 0
        self._registration_lock = threading.Lock()

    async def start(self) -> bool:
        """
        Start the Eureka client service and register with Eureka server.

        Returns:
            bool: True if registration successful, False otherwise
        """
        if not settings.enable_eureka_client:
            self.logger.info("🔄 Eureka client is disabled in configuration")
            return False

        try:
            self.logger.info("🚀 Starting Eureka client service...")

            # Generate new instance ID for fresh registration
            self._generate_new_instance_id()

            # Get local IP address if not provided
            ip_address = settings.eureka_ip_address or self._get_local_ip_address()

            # Register with Eureka server with retry logic
            success = await self._register_with_eureka_with_retry(ip_address)

            if success:
                # Start heartbeat thread
                self._start_heartbeat_thread(ip_address)
                self.logger.info("✅ Eureka client service started successfully")
                return True
            else:
                self.logger.warning(
                    "⚠️ Failed to register with Eureka server after all retry attempts"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to start Eureka client service: {e}")
            return False

    async def stop(self) -> None:
        """
        Stop the Eureka client service and deregister from Eureka server.
        """
        if not settings.enable_eureka_client or not self._is_registered:
            self.logger.info("🔄 Eureka client is not running or not registered")
            return

        try:
            self.logger.info("🛑 Stopping Eureka client service...")

            # Stop heartbeat thread
            self._stop_heartbeat = True
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5)

            # Deregister from Eureka server
            await self._deregister_from_eureka()

            self.logger.info("✅ Eureka client service stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Eureka client service: {e}")

    def is_registered(self) -> bool:
        """
        Check if the service is registered with Eureka server.

        Returns:
            bool: True if registered, False otherwise
        """
        return self._is_registered

    def get_instance_id(self) -> Optional[str]:
        """
        Get the current instance ID.

        Returns:
            Optional[str]: The instance ID if registered, None otherwise
        """
        return self._instance_id

    def _generate_new_instance_id(self) -> None:
        """
        Generate a new instance ID for fresh registration.
        This ensures proper re-registration after server restart.
        """
        with self._registration_lock:
            # Generate new instance ID with timestamp to ensure uniqueness
            timestamp = int(time.time())
            random_suffix = uuid.uuid4().hex[:8]
            self._instance_id = f"{settings.eureka_app_name}:{settings.host}:{settings.port}:{timestamp}_{random_suffix}"
            self.logger.info(f"🆔 Generated new instance ID: {self._instance_id}")

    def _get_local_ip_address(self) -> str:
        """
        Get the local IP address of the machine.

        Returns:
            str: Local IP address
        """
        try:
            # Create a socket to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Doesn't actually connect, just gets local IP
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            # Fallback to localhost
            return "127.0.0.1"

    def _calculate_retry_delay(self, attempt: int, base_delay: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current retry attempt (1-based)
            base_delay: Base delay in seconds

        Returns:
            float: Delay in seconds
        """
        # Exponential backoff: base_delay * (multiplier ^ (attempt - 1))
        delay = base_delay * (settings.eureka_retry_backoff_multiplier ** (attempt - 1))

        # Cap at maximum delay
        delay = min(delay, settings.eureka_max_retry_delay_seconds)

        # Add jitter (±25% random variation)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay += jitter

        return max(0.1, delay)  # Minimum 0.1 seconds

    async def _execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        max_attempts: int,
        base_delay: int,
        *args,
        **kwargs,
    ) -> bool:
        """
        Execute an operation with retry logic and exponential backoff.

        Args:
            operation: Function to execute
            operation_name: Name of operation for logging
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            bool: True if operation succeeded, False otherwise
        """
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

                if result:
                    if attempt > 1:
                        self.logger.info(
                            f"✅ {operation_name} succeeded on attempt {attempt}"
                        )
                    return True
                else:
                    raise Exception(f"{operation_name} returned False")

            except Exception as e:
                last_exception = e

                if attempt < max_attempts:
                    delay = self._calculate_retry_delay(attempt, base_delay)
                    self.logger.warning(
                        f"⚠️ {operation_name} failed on attempt {attempt}/{max_attempts}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"❌ {operation_name} failed after {max_attempts} attempts. "
                        f"Last error: {e}"
                    )

        return False

    async def _register_with_eureka_with_retry(self, ip_address: str) -> bool:
        """
        Register with Eureka server using retry logic.

        Args:
            ip_address: IP address for registration

        Returns:
            bool: True if registration successful, False otherwise
        """
        return await self._execute_with_retry(
            self._register_with_eureka,
            "Eureka registration",
            settings.eureka_registration_retry_attempts,
            settings.eureka_registration_retry_delay_seconds,
            ip_address,
        )

    async def _register_with_eureka(self, ip_address: str) -> bool:
        """
        Register the service with Eureka server using REST API.

        Args:
            ip_address: IP address for registration

        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            self.logger.info(
                f"📝 Registering with Eureka server: {settings.eureka_server_url}"
            )

            # Build registration payload
            registration_data = {
                "instance": {
                    "instanceId": self._instance_id,
                    "hostName": ip_address,
                    "app": settings.eureka_app_name,
                    "ipAddr": ip_address,
                    "status": "UP",
                    "overriddenstatus": "UNKNOWN",
                    "port": {"$": settings.port, "@enabled": "true"},
                    "securePort": {
                        "$": settings.eureka_secure_port,
                        "@enabled": str(settings.eureka_secure_port_enabled).lower(),
                    },
                    "countryId": 1,
                    "dataCenterInfo": {
                        "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                        "name": "MyOwn",
                    },
                    "leaseInfo": {
                        "renewalIntervalInSecs": settings.eureka_lease_renewal_interval_in_seconds,
                        "durationInSecs": settings.eureka_lease_expiration_duration_in_seconds,
                        "registrationTimestamp": 0,
                        "lastRenewalTimestamp": 0,
                        "evictionTimestamp": 0,
                        "serviceUpTimestamp": 0,
                    },
                    "metadata": {"@class": "java.util.Collections$EmptyMap"},
                    "homePageUrl": f"http://{ip_address}:{settings.port}/",
                    "statusPageUrl": f"http://{ip_address}:{settings.port}/health",
                    "healthCheckUrl": f"http://{ip_address}:{settings.port}/health",
                    "vipAddress": settings.eureka_vip_address
                    or settings.eureka_app_name,
                    "secureVipAddress": settings.eureka_secure_vip_address
                    or settings.eureka_app_name,
                    "isCoordinatingDiscoveryServer": "false",
                    "lastUpdatedTimestamp": int(time.time() * 1000),
                    "lastDirtyTimestamp": int(time.time() * 1000),
                    "actionType": "ADDED",
                }
            }

            # Send registration request
            response = requests.post(
                f"{settings.eureka_server_url}/eureka/apps/{settings.eureka_app_name}",
                json=registration_data,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code in [200, 204]:
                self._is_registered = True
                self._last_registration_time = time.time()
                self.logger.info(
                    f"✅ Successfully registered with Eureka server. Instance ID: {self._instance_id}"
                )
                return True
            else:
                self.logger.error(
                    f"❌ Failed to register with Eureka server. Status: {response.status_code}, Response: {response.text}"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to register with Eureka server: {e}")
            return False

    async def _deregister_from_eureka(self) -> None:
        """
        Deregister the service from Eureka server using REST API.
        """
        try:
            self.logger.info("📝 Deregistering from Eureka server...")

            # Send deregistration request
            response = requests.delete(
                f"{settings.eureka_server_url}/eureka/apps/{settings.eureka_app_name}/{self._instance_id}",
                timeout=10,
            )

            if response.status_code in [200, 204]:
                self.logger.info("✅ Successfully deregistered from Eureka server")
            else:
                self.logger.warning(
                    f"⚠️ Deregistration response: {response.status_code}"
                )

            self._is_registered = False

        except Exception as e:
            self.logger.error(f"❌ Failed to deregister from Eureka server: {e}")

    async def _re_register_with_eureka(self, ip_address: str) -> bool:
        """
        Re-register with Eureka server after connection issues.

        Args:
            ip_address: IP address for registration

        Returns:
            bool: True if re-registration successful, False otherwise
        """
        try:
            self.logger.info("🔄 Attempting to re-register with Eureka server...")

            # Generate new instance ID for re-registration
            self._generate_new_instance_id()

            # Reset registration state
            self._is_registered = False
            self._consecutive_heartbeat_failures = 0

            # Attempt re-registration with retry
            success = await self._register_with_eureka_with_retry(ip_address)

            if success:
                self.logger.info("✅ Successfully re-registered with Eureka server")
                return True
            else:
                self.logger.error("❌ Failed to re-register with Eureka server")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error during re-registration: {e}")
            return False

    def _send_heartbeat_with_retry(self, ip_address: str) -> bool:
        """
        Send heartbeat to Eureka server with retry logic.

        Args:
            ip_address: IP address for heartbeat

        Returns:
            bool: True if heartbeat successful, False otherwise
        """
        try:
            # Send heartbeat
            response = requests.put(
                f"{settings.eureka_server_url}/eureka/apps/{settings.eureka_app_name}/{self._instance_id}",
                headers={"Content-Type": "application/json"},
                timeout=5,
            )

            if response.status_code in [200, 204]:
                if self._consecutive_heartbeat_failures > 0:
                    self.logger.info(
                        f"✅ Heartbeat restored after {self._consecutive_heartbeat_failures} failures"
                    )
                    self._consecutive_heartbeat_failures = 0
                return True
            else:
                self._consecutive_heartbeat_failures += 1
                self.logger.warning(
                    f"⚠️ Heartbeat failed. Status: {response.status_code}. "
                    f"Consecutive failures: {self._consecutive_heartbeat_failures}"
                )
                return False

        except Exception as e:
            self._consecutive_heartbeat_failures += 1
            self.logger.error(
                f"❌ Error in heartbeat: {e}. "
                f"Consecutive failures: {self._consecutive_heartbeat_failures}"
            )
            return False

    def _start_heartbeat_thread(self, ip_address: str) -> None:
        """
        Start the heartbeat thread to maintain registration.

        Args:
            ip_address: IP address for heartbeat
        """
        if not self._is_registered:
            return

        def heartbeat_loop():
            """Heartbeat loop to maintain Eureka registration with retry logic."""
            while not self._stop_heartbeat and self._is_registered:
                try:
                    # Send heartbeat with retry logic
                    success = self._send_heartbeat_with_retry(ip_address)

                    if (
                        not success
                        and self._consecutive_heartbeat_failures
                        >= settings.eureka_heartbeat_retry_attempts
                    ):
                        # Try to re-register if too many consecutive failures
                        self.logger.warning(
                            f"⚠️ Too many consecutive heartbeat failures ({self._consecutive_heartbeat_failures}). "
                            "Attempting to re-register..."
                        )

                        # Attempt re-registration if enabled
                        if settings.eureka_enable_auto_re_registration:
                            self.logger.info(
                                f"⏳ Waiting {settings.eureka_re_registration_delay_seconds}s before re-registration..."
                            )
                            time.sleep(settings.eureka_re_registration_delay_seconds)

                            # Attempt re-registration in a separate thread to avoid blocking
                            def re_register_task():
                                try:
                                    # Create new event loop for async operation
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    loop.run_until_complete(
                                        self._re_register_with_eureka(ip_address)
                                    )
                                    loop.close()
                                except Exception as e:
                                    self.logger.error(f"❌ Re-registration failed: {e}")

                            re_register_thread = threading.Thread(
                                target=re_register_task, daemon=True
                            )
                            re_register_thread.start()
                        else:
                            self.logger.warning("🔄 Auto re-registration is disabled")

                        # Reset failure counter after attempting re-registration
                        self._consecutive_heartbeat_failures = 0

                    # Wait for next heartbeat
                    time.sleep(settings.eureka_heartbeat_interval_seconds)

                except Exception as e:
                    self.logger.error(f"❌ Error in heartbeat loop: {e}")
                    time.sleep(5)  # Wait before retrying

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info("💓 Heartbeat thread started with retry logic")

    @asynccontextmanager
    async def lifecycle(self):
        """
        Context manager for Eureka client lifecycle management.

        Usage:
            async with eureka_client_service.lifecycle():
                # Service is registered and running
                pass
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


# Global Eureka client service instance
eureka_client_service = EurekaClientService()
