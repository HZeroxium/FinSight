# misc/database_migration.py

"""
Database Migration Service for FinSight News Crawler
Provides functionality to migrate data between local and cloud MongoDB databases
with date range support and boundary validation
"""

import asyncio
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from common.logger.logger_factory import LoggerFactory
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field, field_validator
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.core.config import settings


class MigrationDateRange(BaseModel):
    """Model for migration date range configuration"""

    start_date: Optional[datetime] = Field(None, description="Migration start date")
    end_date: Optional[datetime] = Field(None, description="Migration end date")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_dates(cls, v):
        """Ensure dates are timezone-aware"""
        if v is None:
            return v
        if isinstance(v, str):
            # Parse string dates to datetime
            try:
                parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(f"Invalid date format: {v}")
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("end_date")
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end_date is after start_date"""
        if v is not None and info.data.get("start_date") is not None:
            if v <= info.data["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v


class MigrationStats(BaseModel):
    """Statistics for migration operation using Pydantic v2"""

    total_articles: int = Field(default=0, description="Total articles found")
    migrated_articles: int = Field(
        default=0, description="Successfully migrated articles"
    )
    skipped_articles: int = Field(default=0, description="Skipped articles")
    failed_articles: int = Field(default=0, description="Failed articles")
    start_time: Optional[datetime] = Field(None, description="Migration start time")
    end_time: Optional[datetime] = Field(None, description="Migration end time")
    actual_date_range: Optional[MigrationDateRange] = Field(
        None, description="Actual date range used"
    )

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate migration duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success_rate(self) -> float:
        """Calculate migration success rate"""
        if self.total_articles == 0:
            return 0.0
        return (self.migrated_articles / self.total_articles) * 100.0


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

    async def get_database_date_boundaries(
        self, env: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the earliest and latest published_at dates from the database

        Args:
            env: Environment ('local' or 'cloud')

        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data
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

            # Get date boundaries using aggregation
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "min_date": {"$min": "$published_at"},
                        "max_date": {"$max": "$published_at"},
                    }
                }
            ]

            result = (
                await db[settings.mongodb_collection_news]
                .aggregate(pipeline)
                .to_list(1)
            )

            client.close()

            if result and result[0]["min_date"] is not None:
                min_date = result[0]["min_date"]
                max_date = result[0]["max_date"]

                # Ensure timezone awareness
                if min_date.tzinfo is None:
                    min_date = min_date.replace(tzinfo=timezone.utc)
                if max_date.tzinfo is None:
                    max_date = max_date.replace(tzinfo=timezone.utc)

                return min_date, max_date

            return None, None

        except Exception as e:
            error_msg = f"Failed to get date boundaries for {env}: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)

    async def validate_and_adjust_date_range(
        self, env: str, requested_range: MigrationDateRange
    ) -> MigrationDateRange:
        """
        Validate and adjust date range based on database boundaries

        Args:
            env: Environment to check
            requested_range: Requested date range

        Returns:
            Adjusted date range within database boundaries
        """
        min_db_date, max_db_date = await self.get_database_date_boundaries(env)

        if min_db_date is None or max_db_date is None:
            self.logger.warning(f"No data found in {env} database")
            return MigrationDateRange(start_date=None, end_date=None)

        # Adjust start_date
        adjusted_start = requested_range.start_date
        if adjusted_start is None:
            adjusted_start = min_db_date
        elif adjusted_start < min_db_date:
            self.logger.info(
                f"Requested start_date {adjusted_start} is earlier than database minimum {min_db_date}. "
                f"Using database minimum."
            )
            adjusted_start = min_db_date

        # Adjust end_date
        adjusted_end = requested_range.end_date
        if adjusted_end is None:
            adjusted_end = max_db_date
        elif adjusted_end > max_db_date:
            self.logger.info(
                f"Requested end_date {adjusted_end} is later than database maximum {max_db_date}. "
                f"Using database maximum."
            )
            adjusted_end = max_db_date

        return MigrationDateRange(start_date=adjusted_start, end_date=adjusted_end)

    async def get_database_stats(self, env: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a database environment

        Args:
            env: Environment ('local' or 'cloud')

        Returns:
            Dictionary with database statistics including date boundaries
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

            # Get date boundaries using aggregation in this method
            min_date, max_date = None, None
            if articles_count > 0:
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "min_date": {"$min": "$published_at"},
                            "max_date": {"$max": "$published_at"},
                        }
                    }
                ]

                result = (
                    await db[settings.mongodb_collection_news]
                    .aggregate(pipeline)
                    .to_list(1)
                )

                if result and result[0]["min_date"] is not None:
                    min_date = result[0]["min_date"]
                    max_date = result[0]["max_date"]

                    # Ensure timezone awareness
                    if min_date.tzinfo is None:
                        min_date = min_date.replace(tzinfo=timezone.utc)
                    if max_date.tzinfo is None:
                        max_date = max_date.replace(tzinfo=timezone.utc)

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
                "date_range": {
                    "earliest_article": min_date.isoformat() if min_date else None,
                    "latest_article": max_date.isoformat() if max_date else None,
                    "span_days": (
                        (max_date - min_date).days if min_date and max_date else 0
                    ),
                },
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
        date_range: Optional[MigrationDateRange] = None,
        time_range_days: Optional[
            int
        ] = None,  # Deprecated, kept for backward compatibility
        dropout_ratio: float = 0.0,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> MigrationStats:
        """
        Migrate articles between databases with enhanced date range support

        Args:
            source_env: Source environment ('local' or 'cloud')
            target_env: Target environment ('local' or 'cloud')
            date_range: Specific date range for migration
            time_range_days: Deprecated - Number of recent days to migrate
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

            # Handle backward compatibility for time_range_days
            if date_range is None and time_range_days is not None:
                self.logger.warning(
                    "time_range_days parameter is deprecated. Use date_range parameter instead."
                )
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=time_range_days)
                date_range = MigrationDateRange(
                    start_date=start_date, end_date=end_date
                )
            elif date_range is None:
                # Default to all data
                date_range = MigrationDateRange()

            # Validate and adjust date range based on database boundaries
            adjusted_range = await self.validate_and_adjust_date_range(
                source_env, date_range
            )
            stats.actual_date_range = adjusted_range

            # Build query filter
            query_filter = {}
            if adjusted_range.start_date and adjusted_range.end_date:
                query_filter["published_at"] = {
                    "$gte": adjusted_range.start_date,
                    "$lte": adjusted_range.end_date,
                }
                self.logger.info(
                    f"Migration date range: {adjusted_range.start_date.isoformat()} to "
                    f"{adjusted_range.end_date.isoformat()}"
                )

            # Get total count
            stats.total_articles = await self.source_db[
                settings.mongodb_collection_news
            ].count_documents(query_filter)

            if stats.total_articles == 0:
                self.logger.info(
                    "No articles found to migrate with the specified criteria"
                )
                return stats

            # Calculate effective articles after dropout
            effective_articles = int(stats.total_articles * (1 - dropout_ratio))

            self.logger.info(
                f"Starting migration: {stats.total_articles:,} articles from {source_env} to {target_env}"
            )

            if dropout_ratio > 0:
                self.logger.info(
                    f"Dropout ratio: {dropout_ratio:.2f} - Will randomly sample ~{effective_articles:,} articles"
                )

            if dry_run:
                self.logger.info("DRY RUN MODE - No actual changes will be made")

            # Process articles in batches with random dropout
            cursor = (
                self.source_db[settings.mongodb_collection_news]
                .find(query_filter)
                .sort("published_at", 1)
                .batch_size(batch_size)
            )

            batch_number = 0
            articles_processed = 0
            current_batch: List[Dict] = []

            async for article in cursor:
                articles_processed += 1
                current_batch.append(article)

                # Process when batch is full or at end
                if len(current_batch) >= batch_size:
                    await self._process_article_batch(
                        current_batch, dropout_ratio, stats, dry_run, batch_number + 1
                    )
                    current_batch.clear()
                    batch_number += 1

                    # Log progress
                    progress = (articles_processed / stats.total_articles) * 100
                    self.logger.info(
                        f"Batch {batch_number}: Processed {articles_processed:,}/{stats.total_articles:,} "
                        f"({progress:.1f}%) - Migrated: {stats.migrated_articles:,}, "
                        f"Skipped: {stats.skipped_articles:,}, Failed: {stats.failed_articles:,}"
                    )

            # Process remaining articles in final batch
            if current_batch:
                await self._process_article_batch(
                    current_batch, dropout_ratio, stats, dry_run, batch_number + 1
                )

            stats.end_time = datetime.now(timezone.utc)

            # Log final statistics
            self.logger.info(f"Migration completed in {stats.duration}")
            self.logger.info(f"Total articles: {stats.total_articles:,}")
            self.logger.info(f"Migrated: {stats.migrated_articles:,}")
            self.logger.info(f"Skipped: {stats.skipped_articles:,}")
            self.logger.info(f"Failed: {stats.failed_articles:,}")
            self.logger.info(f"Success rate: {stats.success_rate:.1f}%")

            if dry_run:
                self.logger.info("DRY RUN COMPLETED - No actual changes were made")

            return stats

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            self.logger.error(error_msg)
            raise DatabaseMigrationError(error_msg)

    async def _process_article_batch(
        self,
        batch_articles: List[Dict],
        dropout_ratio: float,
        stats: MigrationStats,
        dry_run: bool,
        batch_number: int,
    ) -> None:
        """
        Process a batch of articles with random dropout sampling

        Args:
            batch_articles: List of articles in the batch
            dropout_ratio: Ratio of articles to randomly skip (0.0-1.0)
            stats: Migration statistics to update
            dry_run: Whether this is a dry run
            batch_number: Current batch number for logging
        """
        try:
            # Apply random dropout to the batch
            if dropout_ratio > 0.0:
                # Random sampling - keep (1 - dropout_ratio) of articles
                keep_count = int(len(batch_articles) * (1 - dropout_ratio))
                if keep_count < len(batch_articles):
                    # Randomly select articles to keep
                    articles_to_process = random.sample(batch_articles, keep_count)
                    skipped_in_batch = len(batch_articles) - keep_count
                    stats.skipped_articles += skipped_in_batch

                    self.logger.debug(
                        f"Batch {batch_number}: Randomly selected {keep_count}/{len(batch_articles)} "
                        f"articles (skipped {skipped_in_batch})"
                    )
                else:
                    articles_to_process = batch_articles
            else:
                articles_to_process = batch_articles

            # Process selected articles
            for article in articles_to_process:
                try:
                    if not dry_run:
                        # Check if article already exists
                        existing = await self.target_db[
                            settings.mongodb_collection_news
                        ].find_one({"_id": article["_id"]})

                        if existing:
                            self.logger.debug(
                                f"Article {article['_id']} already exists, skipping"
                            )
                            stats.skipped_articles += 1
                            continue

                        # Insert article
                        await self.target_db[
                            settings.mongodb_collection_news
                        ].insert_one(article)

                    stats.migrated_articles += 1

                except Exception as e:
                    self.logger.error(
                        f"Failed to migrate article {article.get('_id', 'unknown')}: {e}"
                    )
                    stats.failed_articles += 1

        except Exception as e:
            self.logger.error(f"Failed to process batch {batch_number}: {e}")
            # Count all articles in batch as failed
            stats.failed_articles += len(batch_articles)

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
    """Main function for command-line usage with enhanced date range support"""
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
        "--start-date",
        type=str,
        help="Migration start date (ISO format: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Migration end date (ISO format: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--time-range-days",
        type=int,
        help="Number of recent days to migrate (deprecated - use start-date/end-date)",
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
    parser.add_argument(
        "--show-boundaries",
        action="store_true",
        help="Show database date boundaries",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == args.target:
        print("Error: Source and target environments cannot be the same")
        return 1

    if not (0.0 <= args.dropout_ratio <= 1.0):
        print("Error: Dropout ratio must be between 0.0 and 1.0")
        return 1

    # Parse date range if provided
    date_range = None
    if args.start_date or args.end_date:
        try:
            start_date = None
            end_date = None

            if args.start_date:
                # Handle different date formats
                if "T" not in args.start_date:
                    args.start_date += "T00:00:00Z"
                start_date = datetime.fromisoformat(
                    args.start_date.replace("Z", "+00:00")
                )

            if args.end_date:
                if "T" not in args.end_date:
                    args.end_date += "T23:59:59Z"
                end_date = datetime.fromisoformat(args.end_date.replace("Z", "+00:00"))

            date_range = MigrationDateRange(start_date=start_date, end_date=end_date)
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            print("Use format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")
            return 1

    try:
        async with DatabaseMigrationService() as migration_service:

            if args.show_boundaries:
                # Show database date boundaries
                print("\n=== Database Date Boundaries ===")
                for env in [args.source, args.target]:
                    try:
                        min_date, max_date = (
                            await migration_service.get_database_date_boundaries(env)
                        )
                        print(f"\n{env.upper()} Database:")
                        if min_date and max_date:
                            print(f"  Earliest article: {min_date.isoformat()}")
                            print(f"  Latest article: {max_date.isoformat()}")
                            print(f"  Date span: {(max_date - min_date).days} days")
                        else:
                            print("  No articles found")
                    except Exception as e:
                        print(f"  Error getting {env} boundaries: {e}")
                return 0

            if args.stats_only:
                # Show database statistics
                print("\n=== Database Statistics ===")
                for env in [args.source, args.target]:
                    try:
                        stats = await migration_service.get_database_stats(env)
                        print(f"\n{env.upper()} Database:")
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
                            print(f"    Span: {date_info['span_days']} days")
                        else:
                            print("  No articles found")
                    except Exception as e:
                        print(f"  Error getting {env} stats: {e}")
                return 0

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

                return 0

            # Perform migration
            print(
                f"\n=== Migrating from {args.source.upper()} to {args.target.upper()} ==="
            )

            if date_range:
                if date_range.start_date:
                    print(f"Start date: {date_range.start_date.isoformat()}")
                if date_range.end_date:
                    print(f"End date: {date_range.end_date.isoformat()}")
            elif args.time_range_days:
                print(f"Time range: Last {args.time_range_days} days")

            if args.dropout_ratio > 0:
                print(f"Dropout ratio: {args.dropout_ratio:.2f}")

            if args.dry_run:
                print("DRY RUN MODE - No actual changes will be made")

            stats = await migration_service.migrate_articles(
                source_env=args.source,
                target_env=args.target,
                date_range=date_range,
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
            print(f"Success rate: {stats.success_rate:.1f}%")

            if stats.actual_date_range:
                actual_range = stats.actual_date_range
                print(f"\nActual date range used:")
                if actual_range.start_date:
                    print(f"  Start: {actual_range.start_date.isoformat()}")
                if actual_range.end_date:
                    print(f"  End: {actual_range.end_date.isoformat()}")

    except Exception as e:
        print(f"Migration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
