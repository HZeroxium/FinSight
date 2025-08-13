# gRPC Implementation Guide

## Overview

This guide explains how to use the gRPC implementation for the FinSight news crawler service. The gRPC API provides high-performance, type-safe access to all news service functionality alongside the existing REST API.

## Architecture

The gRPC implementation follows a clean architecture pattern:

```plaintext
proto/                          # Protocol Buffer definitions
├── news_service.proto          # Service contract

src/grpc_services/              # gRPC implementation
├── news_grpc_service.py        # Service logic
└── grpc_server.py              # Server management

src/utils/                      # Utilities
└── grpc_converters.py          # Data conversion

src/grpc_generated/             # Generated code (after compilation)
├── news_service_pb2.py         # Message classes
└── news_service_pb2_grpc.py    # Service stubs

scripts/                        # Tools
└── generate_grpc_code.py       # Code generation script

examples/                       # Usage examples
└── grpc_client_example.py      # Client examples
```

## Setup and Installation

### 1. Install Dependencies

```bash
pip install grpcio grpcio-tools
```

### 2. Generate gRPC Code

Run the code generation script to create Python bindings from the Protocol Buffer definitions:

```bash
python scripts/generate_grpc_code.py
```

This will create the generated gRPC modules in `src/grpc_generated/`.

### 3. Start the Server

The gRPC server can be started in two ways:

#### Option A: Integrated with FastAPI (Recommended)

```bash
python src/main.py
```

This starts both REST API (port 8000) and gRPC server (port 50051).

#### Option B: Standalone gRPC Server

```bash
python src/grpc_services/run_grpc_server.py
```

This starts only the gRPC server on port 50051.

## API Reference

### Service Definition

The `NewsService` provides the following RPC methods:

```protobuf
service NewsService {
    rpc SearchNews(SearchNewsRequest) returns (NewsResponse);
    rpc GetRecentNews(GetRecentNewsRequest) returns (NewsResponse);
    rpc GetNewsBySource(GetNewsBySourceRequest) returns (NewsResponse);
    rpc GetNewsByTags(GetNewsByTagsRequest) returns (NewsResponse);
    rpc GetAvailableTags(GetAvailableTagsRequest) returns (TagsResponse);
    rpc GetNewsStatistics(google.protobuf.Empty) returns (StatisticsResponse);
}
```

### Request/Response Types

#### SearchNewsRequest

```protobuf
message SearchNewsRequest {
    repeated string keywords = 1;
    NewsSource source = 2;
    repeated string tags = 3;
    google.protobuf.Timestamp start_date = 4;
    google.protobuf.Timestamp end_date = 5;
    int32 skip = 6;
    int32 limit = 7;
}
```

#### NewsResponse

```protobuf
message NewsResponse {
    repeated NewsItem items = 1;
    int32 total_count = 2;
    bool has_more = 3;
}
```

#### NewsItem

```protobuf
message NewsItem {
    string id = 1;
    string title = 2;
    string content = 3;
    string url = 4;
    string source_url = 5;
    NewsSource source = 6;
    repeated string tags = 7;
    google.protobuf.Timestamp published_at = 8;
    google.protobuf.Timestamp created_at = 9;
    google.protobuf.Timestamp updated_at = 10;
}
```

### News Sources

Available news sources are defined as an enum:

```protobuf
enum NewsSource {
    NEWS_SOURCE_UNSPECIFIED = 0;
    NEWS_SOURCE_COINDESK = 1;
    NEWS_SOURCE_COINTELEGRAPH = 2;
    NEWS_SOURCE_CRYPTONEWS = 3;
    NEWS_SOURCE_DECRYPT = 4;
    NEWS_SOURCE_THEBLOCK = 5;
    NEWS_SOURCE_BITCOIN_MAGAZINE = 6;
}
```

## Client Usage Examples

### Basic Connection

```python
import grpc
from src.grpc_generated import news_service_pb2_grpc, news_service_pb2

# Create channel and stub
channel = grpc.aio.insecure_channel("localhost:50051")
stub = news_service_pb2_grpc.NewsServiceStub(channel)
```

### Search News

```python
async def search_news():
    request = news_service_pb2.SearchNewsRequest()
    request.keywords.extend(["bitcoin", "cryptocurrency"])
    request.source = news_service_pb2.NEWS_SOURCE_COINDESK
    request.limit = 10

    response = await stub.SearchNews(request)

    print(f"Found {len(response.items)} articles")
    for item in response.items:
        print(f"- {item.title}")
```

### Get Recent News

```python
async def get_recent_news():
    request = news_service_pb2.GetRecentNewsRequest()
    request.hours = 24
    request.limit = 5

    response = await stub.GetRecentNews(request)

    for item in response.items:
        print(f"- {item.title} ({item.source})")
```

### Get News by Tags

```python
async def get_news_by_tags():
    request = news_service_pb2.GetNewsByTagsRequest()
    request.tags.extend(["crypto", "blockchain"])
    request.limit = 10

    response = await stub.GetNewsByTags(request)

    for item in response.items:
        print(f"- {item.title} (Tags: {', '.join(item.tags)})")
```

### Get Available Tags

```python
async def get_available_tags():
    request = news_service_pb2.GetAvailableTagsRequest()
    request.limit = 20

    response = await stub.GetAvailableTags(request)

    print(f"Available tags: {', '.join(response.tags)}")
```

### Get Statistics

```python
from google.protobuf.empty_pb2 import Empty

async def get_statistics():
    response = await stub.GetNewsStatistics(Empty())

    print(f"Total articles: {response.total_articles}")
    print(f"Recent articles (24h): {response.recent_articles_24h}")
    print("Articles by source:")
    for source, count in response.articles_by_source.items():
        print(f"  {source}: {count}")
```

## Error Handling

gRPC errors should be handled using `grpc.RpcError`:

```python
import grpc

try:
    response = await stub.SearchNews(request)
except grpc.RpcError as e:
    print(f"gRPC error: {e.code()} - {e.details()}")

    if e.code() == grpc.StatusCode.NOT_FOUND:
        print("No articles found")
    elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        print("Invalid request parameters")
```

## Performance Considerations

### Connection Pooling

For production use, consider implementing connection pooling:

```python
# Keep connections alive
options = [
    ('grpc.keepalive_time_ms', 30000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
]

channel = grpc.aio.insecure_channel("localhost:50051", options=options)
```

### Batch Operations

When making multiple requests, use async patterns:

```python
async def batch_requests():
    # Execute requests concurrently
    recent_task = stub.GetRecentNews(recent_request)
    tags_task = stub.GetAvailableTags(tags_request)
    stats_task = stub.GetNewsStatistics(Empty())

    recent_response, tags_response, stats_response = await asyncio.gather(
        recent_task, tags_task, stats_task
    )
```

## Configuration

### Server Configuration

gRPC server settings can be configured in `src/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # gRPC settings
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 10
    grpc_max_message_length: int = 4 * 1024 * 1024  # 4MB
```

### Health Monitoring

The integrated health check endpoint includes gRPC status:

```bash
curl http://localhost:8000/health
```

Response includes gRPC server status:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "grpc": "healthy"
  }
}
```

## Testing

### Unit Tests

Run the test suite to verify gRPC functionality:

```bash
python -m pytest src/tests/test_grpc_service.py -v
```

### Manual Testing

Use the provided client example:

```bash
python examples/grpc_client_example.py
```

### Using grpcurl

Test with `grpcurl` command-line tool:

```bash
# List services
grpcurl -plaintext localhost:50051 list

# Call method
grpcurl -plaintext -d '{"limit": 5}' localhost:50051 NewsService/GetRecentNews
```

## Deployment

### Docker

The gRPC server is included in the Docker setup. Update `docker-compose.yml` to expose the gRPC port:

```yaml
services:
  news-crawler:
    # ... existing configuration ...
    ports:
      - "8000:8000" # REST API
      - "50051:50051" # gRPC
```

### Production Considerations

1. **TLS/SSL**: Use secure channels in production:

   ```python
   credentials = grpc.ssl_channel_credentials()
   channel = grpc.aio.secure_channel("your-domain:443", credentials)
   ```

2. **Load Balancing**: Configure gRPC load balancing for multiple server instances.

3. **Monitoring**: Use gRPC interceptors for logging and metrics collection.

4. **Authentication**: Implement authentication interceptors for secure access.

## Troubleshooting

### Common Issues

1. **Import errors for generated code**: Run `python scripts/generate_grpc_code.py`

2. **Connection refused**: Ensure the gRPC server is running on the correct port

3. **Missing dependencies**: Install `grpcio` and `grpcio-tools`

### Debugging

Enable detailed gRPC logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable gRPC debug logs
import os
os.environ['GRPC_VERBOSITY'] = 'DEBUG'
os.environ['GRPC_TRACE'] = 'all'
```

## Migration from REST

### Endpoint Mapping

| REST Endpoint             | gRPC Method         |
| ------------------------- | ------------------- |
| `GET /search`             | `SearchNews`        |
| `GET /recent`             | `GetRecentNews`     |
| `GET /by-source/{source}` | `GetNewsBySource`   |
| `GET /by-tag`             | `GetNewsByTags`     |
| `GET /tags`               | `GetAvailableTags`  |
| `GET /statistics`         | `GetNewsStatistics` |

### Data Format Changes

- Timestamps use `google.protobuf.Timestamp` instead of ISO strings
- Enums use integer values instead of string constants
- Response structure includes metadata (total_count, has_more)

This completes the comprehensive gRPC implementation for your news crawler service!
