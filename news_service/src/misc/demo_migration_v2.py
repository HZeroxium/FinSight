"""
Enhanced Demo script for Database Migration Service v2.0
Shows how to use the migration service with date range support and new features
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from misc.database_migration import (DatabaseMigrationError,
                                     DatabaseMigrationService,
                                     MigrationDateRange)


async def demo_database_stats():
    """Demo: Show enhanced database statistics including date boundaries"""
    print("\n" + "=" * 50)
    print("ENHANCED DATABASE STATISTICS DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        for env in ["local", "cloud"]:
            try:
                print(f"\n{env.upper()} Database Statistics:")
                stats = await migration_service.get_database_stats(env)

                print(f"  Environment: {stats['environment']}")
                print(f"  Articles: {stats['articles_count']:,}")
                print(f"  Database Size: {stats['database_size_mb']:.2f} MB")
                print(f"  Storage Size: {stats['storage_size_mb']:.2f} MB")
                print(f"  Collections: {stats['collections']}")
                print(f"  Indexes: {stats['indexes']}")

                date_info = stats["date_range"]
                if date_info["earliest_article"]:
                    print(f"  Date Range:")
                    print(f"    Earliest: {date_info['earliest_article']}")
                    print(f"    Latest: {date_info['latest_article']}")
                    print(f"    Span: {date_info['span_days']:,} days")
                else:
                    print("  No articles found")

            except DatabaseMigrationError as e:
                print(f"  Error: {e}")


async def demo_date_boundaries():
    """Demo: Show database date boundaries"""
    print("\n" + "=" * 50)
    print("DATABASE DATE BOUNDARIES DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        for env in ["local", "cloud"]:
            try:
                print(f"\n{env.upper()} Database Date Boundaries:")
                min_date, max_date = (
                    await migration_service.get_database_date_boundaries(env)
                )

                if min_date and max_date:
                    print(f"  Earliest Article: {min_date.isoformat()}")
                    print(f"  Latest Article: {max_date.isoformat()}")
                    print(f"  Total Span: {(max_date - min_date).days:,} days")

                    # Show some example date ranges
                    last_week = max_date - timedelta(days=7)
                    last_month = max_date - timedelta(days=30)

                    print(f"  Example Ranges:")
                    print(
                        f"    Last Week: {last_week.isoformat()} to {max_date.isoformat()}"
                    )
                    print(
                        f"    Last Month: {last_month.isoformat()} to {max_date.isoformat()}"
                    )
                else:
                    print("  No articles found")

            except DatabaseMigrationError as e:
                print(f"  Error: {e}")


async def demo_date_range_migration():
    """Demo: Perform migration with specific date range"""
    print("\n" + "=" * 50)
    print("DATE RANGE MIGRATION DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        try:
            # Get source database boundaries to create a meaningful demo
            min_date, max_date = await migration_service.get_database_date_boundaries(
                "local"
            )

            if not min_date or not max_date:
                print("No data found in local database for demo")
                return

            # Create a date range for the last 7 days of available data
            demo_start = max_date - timedelta(days=7)
            demo_end = max_date

            date_range = MigrationDateRange(start_date=demo_start, end_date=demo_end)

            print(
                f"Demo: Migrating articles from {demo_start.isoformat()} to {demo_end.isoformat()}"
            )
            print("Performing dry run migration from local to cloud...")

            stats = await migration_service.migrate_articles(
                source_env="local",
                target_env="cloud",
                date_range=date_range,
                dropout_ratio=0.1,  # Skip 10% for demo
                batch_size=50,
                dry_run=True,
            )

            print(f"\nDate Range Migration Results:")
            print(f"  Duration: {stats.duration}")
            print(f"  Total articles in range: {stats.total_articles:,}")
            print(f"  Would migrate: {stats.migrated_articles:,}")
            print(f"  Would skip: {stats.skipped_articles:,}")
            print(f"  Success rate: {stats.success_rate:.1f}%")

            if stats.actual_date_range:
                actual = stats.actual_date_range
                print(f"  Actual date range used:")
                if actual.start_date:
                    print(f"    Start: {actual.start_date.isoformat()}")
                if actual.end_date:
                    print(f"    End: {actual.end_date.isoformat()}")

        except DatabaseMigrationError as e:
            print(f"Date range migration demo failed: {e}")


async def demo_boundary_validation():
    """Demo: Show how the system validates and adjusts date boundaries"""
    print("\n" + "=" * 50)
    print("BOUNDARY VALIDATION DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        try:
            # Get actual boundaries
            min_date, max_date = await migration_service.get_database_date_boundaries(
                "local"
            )

            if not min_date or not max_date:
                print("No data found in local database for demo")
                return

            print(
                f"Database boundaries: {min_date.isoformat()} to {max_date.isoformat()}"
            )

            # Test with dates outside boundaries
            too_early = min_date - timedelta(days=30)
            too_late = max_date + timedelta(days=30)

            print(f"\nTesting with dates outside boundaries:")
            print(f"  Requested: {too_early.isoformat()} to {too_late.isoformat()}")

            requested_range = MigrationDateRange(
                start_date=too_early, end_date=too_late
            )
            adjusted_range = await migration_service.validate_and_adjust_date_range(
                "local", requested_range
            )

            print(
                f"  Adjusted to: {adjusted_range.start_date.isoformat()} to {adjusted_range.end_date.isoformat()}"
            )
            print("  ✓ System automatically adjusted to database boundaries")

        except DatabaseMigrationError as e:
            print(f"Boundary validation demo failed: {e}")


async def demo_config_switching():
    """Demo: Show configuration switching"""
    print("\n" + "=" * 50)
    print("CONFIGURATION SWITCHING DEMO")
    print("=" * 50)

    from core.config import settings

    print("Current Configuration:")
    print(f"  Environment: {settings.database_environment}")
    print(f"  Active URL: {settings.mongodb_url}")
    print(f"  Active Database: {settings.mongodb_database}")

    print(f"\nDatabase Info:")
    db_info = settings.database_info
    for key, value in db_info.items():
        print(f"  {key}: {value}")

    print(f"\nConnection Options:")
    conn_options = settings.get_mongodb_connection_options()
    for key, value in conn_options.items():
        print(f"  {key}: {value}")


async def demo_migration_verification():
    """Demo: Verify migration results"""
    print("\n" + "=" * 50)
    print("MIGRATION VERIFICATION DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        try:
            print("Verifying migration between local and cloud...")

            verification = await migration_service.verify_migration("local", "cloud")

            print(f"\nVerification Results:")
            print(
                f"  Source (local): {verification['source']['articles_count']:,} articles"
            )
            print(
                f"  Target (cloud): {verification['target']['articles_count']:,} articles"
            )
            print(f"  Match: {'✓' if verification['articles_match'] else '✗'}")

            if not verification["articles_match"]:
                print(f"  Difference: {verification['difference']:,} articles")

        except DatabaseMigrationError as e:
            print(f"Verification failed: {e}")


async def demo_interactive_date_migration():
    """Demo: Interactive date range migration with user input"""
    print("\n" + "=" * 50)
    print("INTERACTIVE DATE RANGE MIGRATION")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        try:
            # Show available date range first
            min_date, max_date = await migration_service.get_database_date_boundaries(
                "local"
            )

            if not min_date or not max_date:
                print("No data found in local database")
                return

            print(f"Available date range: {min_date.date()} to {max_date.date()}")
            print("\nEnter your desired migration date range:")

            while True:
                start_input = input(
                    "Start date (YYYY-MM-DD) or press Enter for database minimum: "
                ).strip()
                if not start_input:
                    start_date = None
                    break

                try:
                    start_date = datetime.fromisoformat(
                        start_input + "T00:00:00Z"
                    ).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    print("Invalid date format. Use YYYY-MM-DD")
                    continue

            while True:
                end_input = input(
                    "End date (YYYY-MM-DD) or press Enter for database maximum: "
                ).strip()
                if not end_input:
                    end_date = None
                    break

                try:
                    end_date = datetime.fromisoformat(end_input + "T23:59:59Z").replace(
                        tzinfo=timezone.utc
                    )
                    break
                except ValueError:
                    print("Invalid date format. Use YYYY-MM-DD")
                    continue

            date_range = MigrationDateRange(start_date=start_date, end_date=end_date)

            # Validate and show what would be migrated
            adjusted_range = await migration_service.validate_and_adjust_date_range(
                "local", date_range
            )

            print(
                f"\nAdjusted date range: {adjusted_range.start_date.date()} to {adjusted_range.end_date.date()}"
            )

            confirm = input("Proceed with dry run migration? (y/N): ").strip().lower()
            if confirm == "y":
                stats = await migration_service.migrate_articles(
                    source_env="local",
                    target_env="cloud",
                    date_range=date_range,
                    dry_run=True,
                )

                print(f"\nMigration Results:")
                print(f"  Articles found: {stats.total_articles:,}")
                print(f"  Would migrate: {stats.migrated_articles:,}")
                print(f"  Success rate: {stats.success_rate:.1f}%")

        except Exception as e:
            print(f"Interactive migration failed: {e}")


async def main():
    """Main demo function with enhanced options"""
    print("FinSight Database Migration Service Demo v2.0")
    print("Enhanced with Date Range Support")
    print("=" * 60)

    demos = [
        ("1", "Enhanced Database Statistics", demo_database_stats),
        ("2", "Database Date Boundaries", demo_date_boundaries),
        ("3", "Date Range Migration Demo", demo_date_range_migration),
        ("4", "Boundary Validation Demo", demo_boundary_validation),
        ("5", "Configuration Switching Demo", demo_config_switching),
        ("6", "Interactive Date Range Migration", demo_interactive_date_migration),
        ("7", "Migration Verification Demo", demo_migration_verification),
    ]

    while True:
        print("\nAvailable demos:")
        for key, description, _ in demos:
            print(f"  {key}. {description}")
        print("  q. Quit")

        choice = input("\nSelect a demo (1-7) or 'q' to quit: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        # Find and run the selected demo
        demo_found = False
        for key, description, demo_func in demos:
            if choice == key:
                try:
                    await demo_func()
                    demo_found = True
                    break
                except Exception as e:
                    print(f"Demo failed: {e}")
                    demo_found = True
                    break

        if not demo_found:
            print("Invalid choice. Please select 1-7 or 'q'")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
