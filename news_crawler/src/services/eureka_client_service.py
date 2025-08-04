"""
Eureka Client Service for service discovery registration.
Handles registration and deregistration with Eureka server.
"""

import asyncio
import socket
import uuid
import threading
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import requests
from common.logger import LoggerFactory, LoggerType, LogLevel
from ..core.config import settings


class EurekaClientService:
    """
    Eureka Client Service for managing service registration with Eureka server.

    This service handles:
    - Service registration with Eureka server
    - Heartbeat management
    - Service deregistration on shutdown
    - Health check URL management
    """

    def __init__(self):
        """Initialize the Eureka client service."""
        self.logger = LoggerFactory.get_logger(
            name="eureka-client-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            console_level=LogLevel.INFO,
            use_colors=True,
            log_file=f"{settings.log_file_path}eureka_client.log",
        )

        self._is_registered: bool = False
        self._instance_id: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat: bool = False

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

            # Generate instance ID if not provided
            if not settings.eureka_instance_id:
                self._instance_id = f"{settings.eureka_app_name}:{settings.host}:{settings.port}:{uuid.uuid4().hex[:8]}"
            else:
                self._instance_id = settings.eureka_instance_id

            # Get local IP address if not provided
            ip_address = settings.eureka_ip_address or self._get_local_ip_address()

            # Register with Eureka server
            success = await self._register_with_eureka(ip_address)

            if success:
                # Start heartbeat thread
                self._start_heartbeat_thread(ip_address)
                self.logger.info("✅ Eureka client service started successfully")
                return True
            else:
                self.logger.warning("⚠️ Failed to register with Eureka server")
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

    def _start_heartbeat_thread(self, ip_address: str) -> None:
        """
        Start the heartbeat thread to maintain registration.

        Args:
            ip_address: IP address for heartbeat
        """
        if not self._is_registered:
            return

        def heartbeat_loop():
            """Heartbeat loop to maintain Eureka registration."""
            while not self._stop_heartbeat and self._is_registered:
                try:
                    # Send heartbeat
                    response = requests.put(
                        f"{settings.eureka_server_url}/eureka/apps/{settings.eureka_app_name}/{self._instance_id}",
                        headers={"Content-Type": "application/json"},
                        timeout=5,
                    )

                    if response.status_code not in [200, 204]:
                        self.logger.warning(
                            f"⚠️ Heartbeat failed. Status: {response.status_code}"
                        )

                    # Wait for next heartbeat
                    time.sleep(settings.eureka_heartbeat_interval_seconds)

                except Exception as e:
                    self.logger.error(f"❌ Error in heartbeat loop: {e}")
                    time.sleep(5)  # Wait before retrying

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info("💓 Heartbeat thread started")

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
