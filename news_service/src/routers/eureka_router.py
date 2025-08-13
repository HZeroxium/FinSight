"""
Eureka Client Router for managing service discovery registration.
Provides endpoints for Eureka client status and management.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from ..utils.dependencies import get_eureka_client_service, require_admin_access
from ..core.config import settings
from common.logger import LoggerFactory, LoggerType, LogLevel

# Setup router logger
logger = LoggerFactory.get_logger(
    name="eureka-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    use_colors=True,
)

router = APIRouter(prefix="/eureka", tags=["Eureka Client Management"])


@router.get("/status")
async def get_eureka_status() -> Dict[str, Any]:
    """
    Get Eureka client status and configuration.

    Returns:
        Dict[str, Any]: Eureka client status information
    """
    try:
        eureka_service = get_eureka_client_service()

        status = {
            "enabled": settings.enable_eureka_client,
            "server_url": settings.eureka_server_url,
            "app_name": settings.eureka_app_name,
            "instance_id": eureka_service.get_instance_id(),
            "registered": eureka_service.is_registered(),
            "configuration": {
                "host": settings.host,
                "port": settings.port,
                "secure_port": settings.eureka_secure_port,
                "secure_port_enabled": settings.eureka_secure_port_enabled,
                "prefer_ip_address": settings.eureka_prefer_ip_address,
                "lease_renewal_interval_in_seconds": settings.eureka_lease_renewal_interval_in_seconds,
                "lease_expiration_duration_in_seconds": settings.eureka_lease_expiration_duration_in_seconds,
                "heartbeat_interval_seconds": settings.eureka_heartbeat_interval_seconds,
                "retry_config": {
                    "registration_retry_attempts": settings.eureka_registration_retry_attempts,
                    "registration_retry_delay_seconds": settings.eureka_registration_retry_delay_seconds,
                    "heartbeat_retry_attempts": settings.eureka_heartbeat_retry_attempts,
                    "heartbeat_retry_delay_seconds": settings.eureka_heartbeat_retry_delay_seconds,
                    "retry_backoff_multiplier": settings.eureka_retry_backoff_multiplier,
                    "max_retry_delay_seconds": settings.eureka_max_retry_delay_seconds,
                    "enable_auto_re_registration": settings.eureka_enable_auto_re_registration,
                    "re_registration_delay_seconds": settings.eureka_re_registration_delay_seconds,
                },
            },
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get Eureka status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Eureka status: {str(e)}"
        )


@router.post("/register")
async def register_with_eureka(
    admin_access: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Manually register the service with Eureka server.

    Args:
        admin_access: Admin access verification

    Returns:
        Dict[str, Any]: Registration result
    """
    if not settings.enable_eureka_client:
        raise HTTPException(status_code=400, detail="Eureka client is disabled")

    try:
        eureka_service = get_eureka_client_service()

        if eureka_service.is_registered():
            return {
                "success": True,
                "message": "Service is already registered with Eureka",
                "instance_id": eureka_service.get_instance_id(),
            }

        success = await eureka_service.start()

        if success:
            return {
                "success": True,
                "message": "Service registered successfully with Eureka",
                "instance_id": eureka_service.get_instance_id(),
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to register with Eureka server"
            )

    except Exception as e:
        logger.error(f"Failed to register with Eureka: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to register with Eureka: {str(e)}"
        )


@router.post("/deregister")
async def deregister_from_eureka(
    admin_access: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Manually deregister the service from Eureka server.

    Args:
        admin_access: Admin access verification

    Returns:
        Dict[str, Any]: Deregistration result
    """
    if not settings.enable_eureka_client:
        raise HTTPException(status_code=400, detail="Eureka client is disabled")

    try:
        eureka_service = get_eureka_client_service()

        if not eureka_service.is_registered():
            return {"success": True, "message": "Service is not registered with Eureka"}

        await eureka_service.stop()

        return {
            "success": True,
            "message": "Service deregistered successfully from Eureka",
        }

    except Exception as e:
        logger.error(f"Failed to deregister from Eureka: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to deregister from Eureka: {str(e)}"
        )


@router.post("/re-register")
async def re_register_with_eureka(
    admin_access: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Manually trigger re-registration with Eureka server.

    Args:
        admin_access: Admin access verification

    Returns:
        Dict[str, Any]: Re-registration result
    """
    if not settings.enable_eureka_client:
        raise HTTPException(status_code=400, detail="Eureka client is disabled")

    try:
        eureka_service = get_eureka_client_service()

        # Get local IP address
        ip_address = (
            settings.eureka_ip_address or eureka_service._get_local_ip_address()
        )

        # Trigger re-registration
        success = await eureka_service._re_register_with_eureka(ip_address)

        if success:
            return {
                "success": True,
                "message": "Service re-registered successfully with Eureka",
                "instance_id": eureka_service.get_instance_id(),
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to re-register with Eureka server"
            )

    except Exception as e:
        logger.error(f"Failed to re-register with Eureka: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to re-register with Eureka: {str(e)}"
        )


@router.get("/config")
async def get_eureka_config(
    admin_access: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Get Eureka client configuration.

    Args:
        admin_access: Admin access verification

    Returns:
        Dict[str, Any]: Eureka configuration
    """
    try:
        return settings.eureka_config

    except Exception as e:
        logger.error(f"Failed to get Eureka config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Eureka config: {str(e)}"
        )
