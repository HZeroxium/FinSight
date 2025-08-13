# Database Migration System - FinSight News Crawler

This document provides comprehensive information about the database migration system that allows seamless switching between local and cloud MongoDB databases.

## 🎯 Overview

The database migration system provides:

- **Centralized Configuration**: Single source of truth for database settings
- **Easy Environment Switching**: Switch between local and cloud databases with environment variables
- **Data Migration**: Migrate articles between local and cloud databases
- **Flexible Options**: Time range filtering, dropout ratios, batch processing
- **Safety Features**: Dry run mode, verification, detailed logging

## 📁 Files Structure

```plaintext
news_crawler/
├── src/
│   ├── core/
│   │   └── config.py                 # Enhanced configuration with local/cloud switching
│   └── misc/
│       ├── database_migration.py     # Main migration service
│       └── demo_migration.py         # Interactive demo script
├── migrate_database.bat              # Windows batch script for easy migration
├── env_template.txt                  # Environment variables template
└── README_DATABASE_MIGRATION.md      # This file
```

## ⚙️ Configuration Setup

### 1. Environment Variables

Copy `env_template.txt` to `.env` and configure:

```bash
# Database Environment: 'local' or 'cloud'
DATABASE_ENVIRONMENT=local

# Local MongoDB Configuration
MONGODB_LOCAL_URL=mongodb://localhost:27017
MONGODB_LOCAL_DATABASE=finsight_coindesk_news

# Cloud MongoDB Configuration (MongoDB Atlas)
MONGODB_CLOUD_URL=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_CLOUD_DATABASE=finsight_coindesk_news_cloud

# Tavily API Key (Required)
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. Configuration Properties

The enhanced `config.py` provides dynamic properties:

```python
from core.config import settings

# Get current active database URL (based on DATABASE_ENVIRONMENT)
current_url = settings.mongodb_url

# Get current active database name
current_db = settings.mongodb_database

# Get complete database info
db_info = settings.database_info

# Get MongoDB connection options
conn_options = settings.get_mongodb_connection_options()
```

## 🚀 Usage Methods

### Method 1: Windows Batch Script (Recommended)

Double-click `migrate_database.bat` for an interactive menu:

```plaintext
1. Show Database Statistics
2. Migrate Local to Cloud (Dry Run)
3. Migrate Local to Cloud (Real)
4. Migrate Cloud to Local (Dry Run)
5. Migrate Cloud to Local (Real)
6. Verify Migration
7. Run Interactive Demo
8. Custom Migration
9. Exit
```

### Method 2: Command Line

```powershell
# Show database statistics
python src/misc/database_migration.py --source local --target cloud --stats-only

# Dry run migration (safe - no changes made)
python src/misc/database_migration.py --source local --target cloud --dry-run

# Real migration with time range
python src/misc/database_migration.py --source local --target cloud --time-range-days 7

# Migration with dropout ratio (skip 10% of articles)
python src/misc/database_migration.py --source local --target cloud --dropout-ratio 0.1

# Verify migration results
python src/misc/database_migration.py --source local --target cloud --verify
```

### Method 3: Interactive Demo

```powershell
python src/misc/demo_migration.py
```

Provides interactive demos for:

- Database statistics
- Configuration switching
- Dry run migration
- Migration verification
- Interactive migration with prompts

### Method 4: Programmatic Usage

```python
from misc.database_migration import DatabaseMigrationService

async def migrate_data():
    async with DatabaseMigrationService() as migration_service:
        # Get database statistics
        stats = await migration_service.get_database_stats("local")

        # Perform migration
        result = await migration_service.migrate_articles(
            source_env="local",
            target_env="cloud",
            time_range_days=7,
            dropout_ratio=0.1,
            dry_run=True
        )

        # Verify migration
        verification = await migration_service.verify_migration("local", "cloud")
```

## 🔧 Advanced Options

### Time Range Filtering

Migrate only recent articles:

```bash
# Last 7 days
--time-range-days 7

# Last 30 days
--time-range-days 30
```

### Dropout Ratio

Skip a percentage of articles (useful for testing):

```bash
# Skip 10% of articles
--dropout-ratio 0.1

# Skip 50% of articles
--dropout-ratio 0.5
```

### Batch Processing

Control batch size for memory efficiency:

```bash
# Process 50 articles at a time
--batch-size 50

# Process 200 articles at a time
--batch-size 200
```

### Dry Run Mode

Test migration without making changes:

```bash
--dry-run
```

## 📊 Migration Statistics

The system provides detailed statistics:

```plaintext
Migration Results:
  Duration: 0:02:45.123456
  Total articles: 1,234
  Migrated: 1,100
  Skipped: 134
  Failed: 0
```

### Statistics Breakdown

- **Total articles**: Articles found matching criteria
- **Migrated**: Successfully transferred articles
- **Skipped**: Articles already exist in target or excluded by dropout
- **Failed**: Articles that failed to migrate (with error logging)

## 🛡️ Safety Features

### 1. Dry Run Mode

- Simulates migration without making changes
- Shows what would be migrated
- Safe for testing and planning

### 2. Duplicate Detection

- Automatically skips articles that already exist in target
- Based on article `_id` field
- Prevents data duplication

### 3. Connection Validation

- Tests database connections before migration
- Validates MongoDB server availability
- Provides clear error messages

### 4. Comprehensive Logging

- Detailed operation logs
- Progress tracking with batch updates
- Error logging with context

### 5. Migration Verification

- Compare article counts between databases
- Identify migration completeness
- Detect data inconsistencies

## 🔍 Troubleshooting

### Common Issues

#### 1. Connection Failures

**Error**: `Failed to connect to databases`

**Solutions**:

- Check if MongoDB server is running
- Verify connection strings in `.env`
- Test network connectivity to cloud database
- Check authentication credentials

#### 2. Import Errors

**Error**: `Import "logger.logger_factory" could not be resolved`

**Solutions**:

- System includes fallback logging
- Check if common/logger directory exists
- Verify Python path configuration

#### 3. Environment Configuration

**Error**: `mongodb_cloud_url is required when database_environment is 'cloud'`

**Solutions**:

- Set `MONGODB_CLOUD_URL` in `.env` file
- Ensure cloud database connection string is correct
- Switch to local environment if cloud not needed

#### 4. Permission Issues

**Error**: Database access denied

**Solutions**:

- Check MongoDB authentication credentials
- Verify database user permissions
- Ensure network access to cloud database

### Debug Mode

Enable debug logging in `.env`:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Connection Testing

Test individual database connections:

```powershell
# Test local connection
python -c "from motor.motor_asyncio import AsyncIOMotorClient; import asyncio; asyncio.run(AsyncIOMotorClient('mongodb://localhost:27017').admin.command('ismaster'))"

# Check environment variables
python -c "from core.config import settings; print(f'Environment: {settings.database_environment}'); print(f'URL: {settings.mongodb_url}')"
```

## 📈 Best Practices

### 1. Migration Planning

- Always start with `--stats-only` to understand data volume
- Use `--dry-run` first to test migration
- Consider `--time-range-days` for large datasets
- Plan migration during low-traffic periods

### 2. Environment Management

- Use separate databases for dev/test/prod
- Keep `.env` files secure and don't commit to version control
- Document your database configuration choices
- Test configuration changes in development first

### 3. Data Safety

- Always backup before major migrations
- Use verification after migration
- Monitor logs during migration
- Keep migration statistics for audit trail

### 4. Performance Optimization

- Adjust `--batch-size` based on available memory
- Use appropriate connection pool settings
- Monitor database performance during migration
- Consider off-peak hours for large migrations

## 🔄 Environment Switching Examples

### Development Setup

```bash
# .env for development
DATABASE_ENVIRONMENT=local
MONGODB_LOCAL_URL=mongodb://localhost:27017
MONGODB_LOCAL_DATABASE=finsight_coindesk_news_dev
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Setup

```bash
# .env for production
DATABASE_ENVIRONMENT=cloud
MONGODB_CLOUD_URL=mongodb+srv://prod_user:password@prod-cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_CLOUD_DATABASE=finsight_coindesk_news_prod
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Testing Setup

```bash
# .env for testing
DATABASE_ENVIRONMENT=local
MONGODB_LOCAL_URL=mongodb://localhost:27017
MONGODB_LOCAL_DATABASE=finsight_coindesk_news_test
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=testing
```

## 🎯 Migration Workflows

### Local Development to Cloud Production

1. **Preparation**:

   ```bash
   # Check local data
   python src/misc/database_migration.py --source local --target cloud --stats-only
   ```

2. **Test Migration**:

   ```bash
   # Dry run with recent data
   python src/misc/database_migration.py --source local --target cloud --time-range-days 7 --dry-run
   ```

3. **Partial Migration**:

   ```bash
   # Migrate last week's data
   python src/misc/database_migration.py --source local --target cloud --time-range-days 7
   ```

4. **Verification**:

   ```bash
   # Verify migration
   python src/misc/database_migration.py --source local --target cloud --verify
   ```

5. **Full Migration**:

   ```bash
   # Migrate all data
   python src/misc/database_migration.py --source local --target cloud
   ```

### Cloud Backup to Local

1. **Create Local Backup**:

   ```bash
   python src/misc/database_migration.py --source cloud --target local --dry-run
   python src/misc/database_migration.py --source cloud --target local
   ```

2. **Verify Backup**:

   ```bash
   python src/misc/database_migration.py --source cloud --target local --verify
   ```

## 📝 Migration Logs

Logs are written to the configured log directory with detailed information:

```plaintext
2024-01-15 10:30:00 - INFO - Starting migration: 1,234 articles from local to cloud
2024-01-15 10:30:15 - INFO - Batch 1: Processed 100/1,234 (8.1%) - Migrated: 95, Skipped: 5, Failed: 0
2024-01-15 10:30:30 - INFO - Batch 2: Processed 200/1,234 (16.2%) - Migrated: 190, Skipped: 10, Failed: 0
...
2024-01-15 10:35:45 - INFO - Migration completed in 0:05:45.123456
2024-01-15 10:35:45 - INFO - Total articles: 1,234
2024-01-15 10:35:45 - INFO - Migrated: 1,200
2024-01-15 10:35:45 - INFO - Skipped: 34
2024-01-15 10:35:45 - INFO - Failed: 0
```

## 🤝 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review log files for error details
3. Test with `--dry-run` and `--stats-only` first
4. Verify environment configuration
5. Check database connectivity

## 🔮 Future Enhancements

Planned improvements:

- [ ] Support for additional databases (PostgreSQL, InfluxDB)
- [ ] Incremental migration (only new/changed articles)
- [ ] Migration scheduling and automation
- [ ] Data transformation during migration
- [ ] Migration progress web interface
- [ ] Rollback functionality
- [ ] Migration templates and presets
- [ ] Performance metrics and optimization recommendations
