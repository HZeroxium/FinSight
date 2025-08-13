"""
Demo script for Database Migration Service
Shows how to use the migration service with different configurations
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from misc.database_migration import DatabaseMigrationService, DatabaseMigrationError


async def demo_database_stats():
    """Demo: Show database statistics"""
    print("\n" + "=" * 50)
    print("DATABASE STATISTICS DEMO")
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

            except DatabaseMigrationError as e:
                print(f"  Error: {e}")


async def demo_dry_run_migration():
    """Demo: Perform a dry run migration"""
    print("\n" + "=" * 50)
    print("DRY RUN MIGRATION DEMO")
    print("=" * 50)

    async with DatabaseMigrationService() as migration_service:
        try:
            print("Performing dry run migration from local to cloud...")

            stats = await migration_service.migrate_articles(
                source_env="local",
                target_env="cloud",
                time_range_days=7,  # Last 7 days only
                dropout_ratio=0.1,  # Skip 10% of articles
                batch_size=50,
                dry_run=True,
            )

            print(f"\nDry Run Results:")
            print(f"  Duration: {stats.duration}")
            print(f"  Total articles: {stats.total_articles:,}")
            print(f"  Would migrate: {stats.migrated_articles:,}")
            print(f"  Would skip: {stats.skipped_articles:,}")
            print(f"  Would fail: {stats.failed_articles:,}")

        except DatabaseMigrationError as e:
            print(f"Dry run failed: {e}")


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


async def demo_interactive_migration():
    """Demo: Interactive migration with user input"""
    print("\n" + "=" * 50)
    print("INTERACTIVE MIGRATION DEMO")
    print("=" * 50)

    print("Available environments: local, cloud")

    while True:
        source = (
            input("\nEnter source environment (local/cloud) or 'quit': ")
            .strip()
            .lower()
        )
        if source == "quit":
            break

        if source not in ["local", "cloud"]:
            print("Invalid environment. Please use 'local' or 'cloud'")
            continue

        target = input("Enter target environment (local/cloud): ").strip().lower()
        if target not in ["local", "cloud"]:
            print("Invalid environment. Please use 'local' or 'cloud'")
            continue

        if source == target:
            print("Source and target cannot be the same")
            continue

        # Get optional parameters
        try:
            days_str = input(
                "Enter time range in days (or press Enter for all): "
            ).strip()
            time_range_days = int(days_str) if days_str else None

            dropout_str = input(
                "Enter dropout ratio 0.0-1.0 (or press Enter for 0.0): "
            ).strip()
            dropout_ratio = float(dropout_str) if dropout_str else 0.0

            dry_run_str = input("Perform dry run? (y/N): ").strip().lower()
            dry_run = dry_run_str == "y"

        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

        # Perform migration
        async with DatabaseMigrationService() as migration_service:
            try:
                print(f"\nStarting migration from {source} to {target}...")

                stats = await migration_service.migrate_articles(
                    source_env=source,
                    target_env=target,
                    time_range_days=time_range_days,
                    dropout_ratio=dropout_ratio,
                    dry_run=dry_run,
                )

                print(f"\nMigration Results:")
                print(f"  Duration: {stats.duration}")
                print(f"  Total articles: {stats.total_articles:,}")
                print(f"  Migrated: {stats.migrated_articles:,}")
                print(f"  Skipped: {stats.skipped_articles:,}")
                print(f"  Failed: {stats.failed_articles:,}")

            except DatabaseMigrationError as e:
                print(f"Migration failed: {e}")


async def main():
    """Main demo function"""
    print("FinSight Database Migration Service Demo")
    print("=" * 60)

    demos = [
        ("1", "Show Database Statistics", demo_database_stats),
        ("2", "Configuration Switching Demo", demo_config_switching),
        ("3", "Dry Run Migration Demo", demo_dry_run_migration),
        ("4", "Migration Verification Demo", demo_migration_verification),
        ("5", "Interactive Migration", demo_interactive_migration),
    ]

    while True:
        print("\nAvailable demos:")
        for key, description, _ in demos:
            print(f"  {key}. {description}")
        print("  q. Quit")

        choice = input("\nSelect a demo (1-5) or 'q' to quit: ").strip().lower()

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
            print("Invalid choice. Please select 1-5 or 'q'")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
