#!/usr/bin/env python3
# check_status.py

"""
Convenience script for checking system status and health
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.market_data_job import MarketDataJobService
from src.services.market_data_service import MarketDataService
from src.factories import create_repository


async def check_job_status():
    """Check cron job service status"""
    print("🔍 Checking Market Data Cron Job Status")
    print("=" * 50)

    try:
        job_service = MarketDataJobService()
        status = job_service.get_status()

        print(f"Service Running: {'✅' if status['is_running'] else '❌'}")
        print(f"Current Job: {status['current_job_id'] or 'None'}")
        print(f"Scheduler State: {status['scheduler_state']}")

        if status.get("next_run_time"):
            print(f"Next Run: {status['next_run_time']}")

        stats = status.get("stats", {})
        print(f"\n📊 Statistics:")
        print(f"  Total Runs: {stats.get('total_runs', 0)}")
        print(f"  Successful: {stats.get('successful_runs', 0)}")
        print(f"  Failed: {stats.get('failed_runs', 0)}")

        if stats.get("last_run_time"):
            print(f"  Last Run: {stats['last_run_time']}")

        if stats.get("last_error_message"):
            print(f"  Last Error: {stats['last_error_message']}")

    except Exception as e:
        print(f"❌ Error checking job status: {e}")


async def check_data_status():
    """Check market data availability"""
    print("\n🗃️ Checking Market Data Status")
    print("=" * 50)

    try:
        # Try MongoDB first, fallback to CSV
        repository_configs = [
            (
                "mongodb",
                {
                    "connection_string": "mongodb://localhost:27017/",
                    "database_name": "finsight_market_data",
                },
            ),
            ("csv", {"base_directory": "data"}),
        ]

        for repo_type, config in repository_configs:
            try:
                repository = create_repository(repo_type, config)
                service = MarketDataService(repository)

                print(f"\n📦 {repo_type.upper()} Repository:")

                # Check available exchanges
                exchanges = await service.get_available_exchanges()
                print(f"  Exchanges: {', '.join(exchanges) if exchanges else 'None'}")

                if exchanges:
                    for exchange in exchanges[:3]:  # Limit to first 3
                        symbols = await service.get_available_symbols(exchange)
                        print(f"  {exchange} Symbols: {len(symbols)} available")

                        if symbols:
                            # Check first symbol's timeframes
                            timeframes = await service.get_available_timeframes(
                                exchange, symbols[0]
                            )
                            print(
                                f"    {symbols[0]} Timeframes: {', '.join(timeframes)}"
                            )

                # Use the first working repository
                return

            except Exception as e:
                print(f"  ❌ {repo_type.upper()}: {str(e)[:100]}...")
                continue

        print("❌ No accessible repositories found")

    except Exception as e:
        print(f"❌ Error checking data status: {e}")


def check_log_files():
    """Check log file status"""
    print("\n📝 Checking Log Files")
    print("=" * 50)

    log_files = [
        "logs/market_data_job.log",
        "logs/market_data_collector.log",
        "logs/fastapi_app.log",
    ]

    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            stat = log_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime)
            print(
                f"  ✅ {log_file}: {size_mb:.2f}MB (modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})"
            )
        else:
            print(f"  ❌ {log_file}: Not found")


def check_config_files():
    """Check configuration files"""
    print("\n⚙️ Checking Configuration Files")
    print("=" * 50)

    config_files = [
        "market_data_job_config.json",
        ".env",
        "requirements.txt",
    ]

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            if config_file.endswith(".json"):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    print(f"  ✅ {config_file}: Valid JSON with {len(config)} keys")
                except json.JSONDecodeError:
                    print(f"  ⚠️ {config_file}: Invalid JSON format")
            else:
                print(f"  ✅ {config_file}: Present")
        else:
            print(f"  ❌ {config_file}: Not found")


async def main():
    """Main status check function"""
    print("🔎 FinSight Backtesting System Status Check")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    await check_job_status()
    await check_data_status()
    check_log_files()
    check_config_files()

    print("\n✅ Status check complete!")
    print("\nQuick Commands:")
    print("  python start_market_data_job.py    # Start cron job service")
    print("  python collect_data.py BTCUSDT     # Collect data for BTCUSDT")
    print("  python -m src.main                 # Start API server")


if __name__ == "__main__":
    asyncio.run(main())
