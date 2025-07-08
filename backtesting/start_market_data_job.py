#!/usr/bin/env python3
# start_market_data_job.py

"""
Convenience script for starting the market data cron job service
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.market_data_job import main as job_main


def main():
    """Start the market data cron job service"""
    print("🚀 Starting Market Data Cron Job Service...")
    print("📊 This will collect market data automatically on schedule")
    print("🔄 Use Ctrl+C to stop the service")
    print("📝 Logs will be written to: logs/market_data_job.log")
    print("-" * 60)

    # Set default arguments for start command
    sys.argv = ["market_data_job", "start"]

    try:
        asyncio.run(job_main())
    except KeyboardInterrupt:
        print("\n👋 Market Data Job Service stopped by user")
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
