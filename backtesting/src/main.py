# main.py

"""
FastAPI Application Entry Point

Clean entry point for starting the FinSight Backtesting API server.
Data collection and cron jobs are handled by separate scripts.
"""

import uvicorn
from .app import app
from .core.config import settings
from common.logger import LoggerFactory


def main():
    """Main entry point for the FastAPI application"""

    # Load settings
    # No need to initialize settings, use the global instance

    # Initialize logger
    logger = LoggerFactory.get_logger(name="main")

    logger.info("🚀 Starting FinSight Backtesting API server")
    logger.info(f"📊 Environment: {settings.environment}")
    logger.info(f"🌐 Server will run at: http://{settings.host}:{settings.port}")
    logger.info(f"📚 API Documentation: http://{settings.host}:{settings.port}/docs")

    # Start the FastAPI server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=settings.debug,
    )


if __name__ == "__main__":
    main()
