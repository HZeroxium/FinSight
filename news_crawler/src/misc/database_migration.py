# misc/database_migration.py

"""
Database Migration Service for FinSight News Crawler
Provides functionality to migrate data between local and cloud MongoDB databases
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import sys

from common.logger.logger_factory import LoggerFactory
from src.core.config import settings


@dataclass
class MigrationStats:
    """Statistics for migration operation"""

    total_articles: int = 0
    migrated_articles: int = 0
    skipped_articles: int = 0
    failed_articles: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DatabaseMigrationError(Exception):
    """Custom exception for database migration errors"""

    pass


class DatabaseMigrationService:
    """Service for migrating data between local and cloud MongoDB databases"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.source_client: Optional[AsyncIOMotorClient] = None
        self.target_client: Optional[AsyncIOMotorClient] = None
        self.source_db: Optional[AsyncIOMotorDatabase] = None
        self.target_db: Optional[AsyncIOMotorDatabase] = None

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.close_connections()

    async def connect_databases(self, source_env: str, target_env: str) -> None:
        """
        Connect to source and target databases

        Args:
            source_env: Source environment ('local' or 'cloud')
            target_env: Target environment ('local' or 'cloud')
        """
        try:
            # Configure source connection
            if source_env == "local":
                source_url = settings.mongodb_local_url
                source_db_name = settings.mongodb_local_database
            else:
                source_url = settings.mongodb_cloud_url
                source_db_name = settings.mongodb_cloud_database

            # Configure target connection
            if target_env == "local":
                target_url = settings.mongodb_local_url
                target_db_name = settings.mongodb_local_database
            else:
                target_url = settings.mongodb_cloud_url
                target_db_name = settings.mongodb_cloud_database

            # Create connections with options
            connection_options = settings.get_mongodb_connection_options()

            self.source_client = AsyncIOMotorClient(source_url, **connection_options)
            self.target_client = AsyncIOMotorClient(target_url, **connection_options)

            # Get database references
            self.source_db = self.source_client[source_db_name]
            self.target_db = self.target_client[target_db_name]

            # Test connections
            await self.source_client.admin.command("ismaster")
            await self.target_client.admin.command("ismaster")

            self.logger.info(
                f"Connected to source database: {source_env} ({source_db_name})"
            )
            self.logger.info(
                f"Connected to target database: {target_env} ({target_db_name})"
            )

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            error_msg = f"Failed to connect to databases: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)

    async def close_connections(self) -> None:
        """Close database connections"""
        if self.source_client:
            self.source_client.close()
            self.logger.debug("Closed source database connection")
        if self.target_client:
            self.target_client.close()
            self.logger.debug("Closed target database connection")

    async def get_database_stats(self, env: str) -> Dict[str, Any]:
        """
        Get statistics for a database environment

        Args:
            env: Environment ('local' or 'cloud')

        Returns:
            Dictionary with database statistics
        """
        try:
            if env == "local":
                client = AsyncIOMotorClient(settings.mongodb_local_url)
                db = client[settings.mongodb_local_database]
            else:
                client = AsyncIOMotorClient(settings.mongodb_cloud_url)
                db = client[settings.mongodb_cloud_database]

            # Test connection
            await client.admin.command("ismaster")

            # Get collection stats
            articles_count = await db[settings.mongodb_collection_news].count_documents(
                {}
            )

            # Get database stats
            db_stats = await db.command("dbStats")

            stats = {
                "environment": env,
                "articles_count": articles_count,
                "database_size_mb": round(
                    db_stats.get("dataSize", 0) / (1024 * 1024), 2
                ),
                "storage_size_mb": round(
                    db_stats.get("storageSize", 0) / (1024 * 1024), 2
                ),
                "indexes": db_stats.get("indexes", 0),
                "collections": db_stats.get("collections", 0),
            }

            client.close()
            return stats

        except Exception as e:
            error_msg = f"Failed to get database stats for {env}: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)

    async def migrate_articles(
        self,
        source_env: str,
        target_env: str,
        time_range_days: Optional[int] = None,
        dropout_ratio: float = 0.0,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> MigrationStats:
        """
        Migrate articles between databases

        Args:
            source_env: Source environment ('local' or 'cloud')
            target_env: Target environment ('local' or 'cloud')
            time_range_days: Number of days to migrate (None for all)
            dropout_ratio: Ratio of articles to skip (0.0 to 1.0)
            batch_size: Number of articles to process in each batch
            dry_run: If True, only simulate migration without actual changes

        Returns:
            MigrationStats object with operation statistics
        """
        stats = MigrationStats()
        stats.start_time = datetime.now(timezone.utc)

        try:
            await self.connect_databases(source_env, target_env)

            # Build query filter
            query_filter = {}
            if time_range_days:
                cutoff_date = datetime.now(timezone.utc) - timedelta(
                    days=time_range_days
                )
                query_filter["published_at"] = {"$gte": cutoff_date}

            # Get total count
            stats.total_articles = await self.source_db[
                settings.mongodb_collection_news
            ].count_documents(query_filter)

            if stats.total_articles == 0:
                self.logger.info("No articles found to migrate")
                return stats

            self.logger.info(
                f"Starting migration: {stats.total_articles} articles from {source_env} to {target_env}"
            )

            if dry_run:
                self.logger.info("DRY RUN MODE - No actual changes will be made")

            # Calculate dropout count
            dropout_count = 0
            if dropout_ratio > 0:
                dropout_count = int(stats.total_articles * dropout_ratio)
                self.logger.info(
                    f"Dropout ratio: {dropout_ratio:.2f} - Will skip {dropout_count} articles"
                )

            # Process articles in batches
            cursor = (
                self.source_db[settings.mongodb_collection_news]
                .find(query_filter)
                .batch_size(batch_size)
            )

            batch_number = 0
            articles_processed = 0

            async for article in cursor:
                articles_processed += 1

                # Apply dropout ratio
                if dropout_count > 0 and articles_processed <= dropout_count:
                    stats.skipped_articles += 1
                    continue

                try:
                    if not dry_run:
                        # Check if article already exists in target
                        existing = await self.target_db[
                            settings.mongodb_collection_news
                        ].find_one({"_id": article["_id"]})

                        if existing:
                            self.logger.debug(
                                f"Article {article['_id']} already exists in target, skipping"
                            )
                            stats.skipped_articles += 1
                            continue

                        # Insert article into target database
                        await self.target_db[
                            settings.mongodb_collection_news
                        ].insert_one(article)

                    stats.migrated_articles += 1

                    # Log progress every batch
                    if stats.migrated_articles % batch_size == 0:
                        batch_number += 1
                        progress = (articles_processed / stats.total_articles) * 100
                        self.logger.info(
                            f"Batch {batch_number}: Processed {articles_processed}/{stats.total_articles} "
                            f"({progress:.1f}%) - Migrated: {stats.migrated_articles}, "
                            f"Skipped: {stats.skipped_articles}, Failed: {stats.failed_articles}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to migrate article {article.get('_id', 'unknown')}: {e}"
                    )
                    stats.failed_articles += 1

            stats.end_time = datetime.now(timezone.utc)

            # Log final statistics
            self.logger.info(f"Migration completed in {stats.duration}")
            self.logger.info(f"Total articles: {stats.total_articles}")
            self.logger.info(f"Migrated: {stats.migrated_articles}")
            self.logger.info(f"Skipped: {stats.skipped_articles}")
            self.logger.info(f"Failed: {stats.failed_articles}")

            if dry_run:
                self.logger.info("DRY RUN COMPLETED - No actual changes were made")

            return stats

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)

    async def verify_migration(
        self, source_env: str, target_env: str
    ) -> Dict[str, Any]:
        """
        Verify migration results by comparing database statistics

        Args:
            source_env: Source environment
            target_env: Target environment

        Returns:
            Dictionary with verification results
        """
        try:
            source_stats = await self.get_database_stats(source_env)
            target_stats = await self.get_database_stats(target_env)

            verification = {
                "source": source_stats,
                "target": target_stats,
                "articles_match": source_stats["articles_count"]
                == target_stats["articles_count"],
                "difference": target_stats["articles_count"]
                - source_stats["articles_count"],
            }

            return verification

        except Exception as e:
            error_msg = f"Migration verification failed: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)


async def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Database Migration Tool for FinSight News Crawler"
    )
    parser.add_argument(
        "--source",
        choices=["local", "cloud"],
        required=True,
        help="Source database environment",
    )
    parser.add_argument(
        "--target",
        choices=["local", "cloud"],
        required=True,
        help="Target database environment",
    )
    parser.add_argument(
        "--time-range-days",
        type=int,
        help="Number of recent days to migrate (default: all)",
    )
    parser.add_argument(
        "--dropout-ratio",
        type=float,
        default=0.0,
        help="Ratio of articles to skip (0.0-1.0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes",
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Show database statistics only"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify migration results"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == args.target:
        print("Error: Source and target environments cannot be the same")
        return

    if not (0.0 <= args.dropout_ratio <= 1.0):
        print("Error: Dropout ratio must be between 0.0 and 1.0")
        return

    try:
        async with DatabaseMigrationService() as migration_service:

            if args.stats_only:
                # Show database statistics
                print("\n=== Database Statistics ===")
                for env in [args.source, args.target]:
                    try:
                        stats = await migration_service.get_database_stats(env)
                        print(f"\n{env.upper()} Database:")
                        print(f"  News: {stats['articles_count']:,}")
                        print(f"  Database Size: {stats['database_size_mb']:.2f} MB")
                        print(f"  Storage Size: {stats['storage_size_mb']:.2f} MB")
                        print(f"  Collections: {stats['collections']}")
                        print(f"  Indexes: {stats['indexes']}")
                    except Exception as e:
                        print(f"  Error getting {env} stats: {e}")
                return

            if args.verify:
                # Verify migration results
                print("\n=== Migration Verification ===")
                verification = await migration_service.verify_migration(
                    args.source, args.target
                )

                print(
                    f"Source ({args.source}): {verification['source']['articles_count']:,} articles"
                )
                print(
                    f"Target ({args.target}): {verification['target']['articles_count']:,} articles"
                )
                print(f"Match: {'✓' if verification['articles_match'] else '✗'}")

                if not verification["articles_match"]:
                    print(f"Difference: {verification['difference']:,} articles")

                return

            # Perform migration
            print(
                f"\n=== Migrating from {args.source.upper()} to {args.target.upper()} ==="
            )

            if args.time_range_days:
                print(f"Time range: Last {args.time_range_days} days")

            if args.dropout_ratio > 0:
                print(f"Dropout ratio: {args.dropout_ratio:.2f}")

            if args.dry_run:
                print("DRY RUN MODE - No actual changes will be made")

            stats = await migration_service.migrate_articles(
                source_env=args.source,
                target_env=args.target,
                time_range_days=args.time_range_days,
                dropout_ratio=args.dropout_ratio,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )

            print(f"\n=== Migration Results ===")
            print(f"Duration: {stats.duration}")
            print(f"Total articles: {stats.total_articles:,}")
            print(f"Migrated: {stats.migrated_articles:,}")
            print(f"Skipped: {stats.skipped_articles:,}")
            print(f"Failed: {stats.failed_articles:,}")

    except Exception as e:
        print(f"Migration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
