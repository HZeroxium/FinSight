# main.py

"""
FinSight Model Builder - Main Entry Point

A robust FastAPI server for time series model training, prediction, and backtesting.
Supports HuggingFace models (PatchTST, PatchTSMixer) with flexible feature engineering.
"""

import uvicorn

from .app import app
from .core.config import get_settings


def main():
    """Main entry point for the application"""
    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    main()
