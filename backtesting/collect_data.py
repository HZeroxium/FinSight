#!/usr/bin/env python3
# collect_data.py

"""
Convenience script for collecting market data for new symbols
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.collect_new_symbol_data import main as collect_main


def main():
    """Run the new symbol data collection"""
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("🔍 FinSight Market Data Collector")
        print("=" * 50)
        print()
        print("Quick Commands:")
        print(
            "  python collect_data.py BTCUSDT          # Collect BTCUSDT (all timeframes)"
        )
        print("  python collect_data.py BTCUSDT 1h       # Collect BTCUSDT 1h data")
        print("  python collect_data.py --help           # Show full help")
        print()
        print("Examples:")
        print("  python collect_data.py ETHUSDT --force-full")
        print("  python collect_data.py BNBUSDT --start-date 2024-01-01")
        print()
        return

    # Parse simple arguments
    args = sys.argv[1:]

    # Handle simple symbol argument
    if len(args) >= 1 and not args[0].startswith("--"):
        symbol = args[0]

        # Handle optional timeframe
        if len(args) >= 2 and not args[1].startswith("--"):
            timeframe = args[1]
            remaining_args = args[2:]
        else:
            timeframe = None
            remaining_args = args[1:]

        # Build new sys.argv
        new_argv = ["collect_new_symbol_data"]
        new_argv.extend(["--symbol", symbol])
        if timeframe:
            new_argv.extend(["--timeframe", timeframe])
        new_argv.extend(remaining_args)

        sys.argv = new_argv

        print(
            f"🎯 Collecting data for {symbol}"
            + (f" ({timeframe})" if timeframe else " (all timeframes)")
        )
        print("📊 Starting intelligent data collection...")
        print("-" * 60)

    try:
        asyncio.run(collect_main())
    except KeyboardInterrupt:
        print("\n👋 Data collection stopped by user")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
