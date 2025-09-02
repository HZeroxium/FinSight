# 4. Design Patterns

## 🎯 Overview

The FinSight platform extensively employs design patterns to achieve clean architecture, maintainability, and flexibility. These patterns follow SOLID principles and enable the system to evolve while maintaining code quality.

## 🏗️ Core Design Patterns

### 1. **Factory Pattern**

**Purpose**: Centralized object creation with runtime implementation selection.

**Implementation**:

```python
# Market Data Repository Factory
class MarketDataRepositoryFactory:
    @staticmethod
    def create_repository(repository_type: str, config: Dict[str, Any]) -> MarketDataRepository:
        """Create repository based on configuration."""
        if repository_type == "mongodb":
            return MongoDBMarketDataRepository(
                connection_string=config["mongodb_url"],
                database_name=config["database_name"]
            )
        elif repository_type == "influxdb":
            return InfluxMarketDataRepository(
                url=config["influxdb_url"],
                token=config["token"],
                org=config["org"],
                bucket=config["bucket"]
            )
        elif repository_type == "csv":
            return CSVMarketDataRepository(
                base_path=config["csv_path"]
            )
        else:
            raise ValueError(f"Unknown repository type: {repository_type}")

# Usage
config = {"mongodb_url": "mongodb://localhost:27017", "database_name": "finsight"}
repository = MarketDataRepositoryFactory.create_repository("mongodb", config)
```

**Benefits**:

- **Flexibility**: Easy switching between implementations
- **Configuration**: Runtime repository selection
- **Testing**: Easy mocking for unit tests
- **Maintenance**: Centralized creation logic

### 2. **Repository Pattern**

**Purpose**: Abstract data access behind consistent interfaces.

**Implementation**:

```python
# Repository Interface
class MarketDataRepository(ABC):
    @abstractmethod
    async def save_ohlcv(self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]) -> bool:
        pass

    @abstractmethod
    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        pass

# MongoDB Implementation
class MongoDBMarketDataRepository(MarketDataRepository):
    def __init__(self, connection_string: str, database_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.ohlcv_collection = self.db["ohlcv"]

    async def save_ohlcv(self, exchange: str, symbol: str, timeframe: str, data: List[OHLCVSchema]) -> bool:
        try:
            documents = [self._convert_schema_to_document(item) for item in data]
            result = await self.ohlcv_collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids) == len(documents)
        except Exception as e:
            logger.error(f"Failed to save OHLCV data: {e}")
            return False

    async def get_ohlcv(self, query: OHLCVQuerySchema) -> List[OHLCVSchema]:
        try:
            filter_dict = {
                "exchange": query.exchange,
                "symbol": query.symbol,
                "timeframe": query.timeframe
            }

            if query.start_date:
                filter_dict["timestamp"] = {"$gte": query.start_date}
            if query.end_date:
                filter_dict["timestamp"] = {"$lte": query.end_date}

            cursor = self.ohlcv_collection.find(filter_dict).sort("timestamp", -1)
            if query.limit:
                cursor = cursor.limit(query.limit)

            documents = await cursor.to_list(length=query.limit or 1000)
            return [self._convert_document_to_schema(doc) for doc in documents]

        except Exception as e:
            logger.error(f"Failed to retrieve OHLCV data: {e}")
            return []
```

**Benefits**:

- **Abstraction**: Business logic independent of data access
- **Testability**: Easy mocking for unit tests
- **Flexibility**: Multiple storage implementations
- **Consistency**: Uniform interface across data sources

### 3. **Strategy Pattern**

**Purpose**: Encapsulate algorithms and make them interchangeable.

**Implementation**:

```python
# Strategy Interface
class NewsCollectionStrategy(ABC):
    @abstractmethod
    async def collect_news(self, keywords: List[str], max_articles: int) -> List[NewsArticle]:
        pass

# RSS Strategy
class RSSNewsCollectionStrategy(NewsCollectionStrategy):
    async def collect_news(self, keywords: List[str], max_articles: int) -> List[NewsArticle]:
        articles = []
        for keyword in keywords:
            # RSS feed collection logic
            rss_articles = await self._fetch_rss_feed(keyword)
            articles.extend(rss_articles[:max_articles])
        return articles

# API Strategy
class APINewsCollectionStrategy(NewsCollectionStrategy):
    async def collect_news(self, keywords: List[str], max_articles: int) -> List[NewsArticle]:
        articles = []
        for keyword in keywords:
            # API-based collection logic
            api_articles = await self._fetch_from_api(keyword)
            articles.extend(api_articles[:max_articles])
        return articles

# Strategy Context
class NewsCollector:
    def __init__(self, strategy: NewsCollectionStrategy):
        self.strategy = strategy

    async def collect_news(self, keywords: List[str], max_articles: int) -> List[NewsArticle]:
        return await self.strategy.collect_news(keywords, max_articles)

    def set_strategy(self, strategy: NewsCollectionStrategy):
        self.strategy = strategy

# Usage
rss_strategy = RSSNewsCollectionStrategy()
api_strategy = APINewsCollectionStrategy()

collector = NewsCollector(rss_strategy)
rss_articles = await collector.collect_news(["bitcoin", "crypto"], 100)

collector.set_strategy(api_strategy)
api_articles = await collector.collect_news(["bitcoin", "crypto"], 100)
```

**Benefits**:

- **Algorithm Selection**: Runtime strategy switching
- **Extensibility**: Easy addition of new strategies
- **Maintainability**: Isolated algorithm logic
- **Testing**: Individual strategy testing

### 4. **Observer Pattern**

**Purpose**: Enable loose coupling between event producers and consumers.

**Implementation**:

```python
# Event Observer Interface
class EventObserver(ABC):
    @abstractmethod
    async def on_event(self, event_type: str, payload: Dict[str, Any]):
        pass

# News Collection Observer
class NewsCollectionObserver(EventObserver):
    def __init__(self, sentiment_service: SentimentAnalysisService):
        self.sentiment_service = sentiment_service

    async def on_event(self, event_type: str, payload: Dict[str, Any]):
        if event_type == "news_collected":
            await self._process_news_for_sentiment(payload)

    async def _process_news_for_sentiment(self, payload: Dict[str, Any]):
        try:
            # Get collected news articles
            articles = await self._get_recent_articles(payload["source"])

            # Analyze sentiment for each article
            for article in articles:
                sentiment_result = await self.sentiment_service.analyze_sentiment(
                    article["content"],
                    context=f"News from {payload['source']}"
                )

                # Update article with sentiment
                await self._update_article_sentiment(article["id"], sentiment_result)

        except Exception as e:
            logger.error(f"Failed to process news for sentiment: {e}")

# Event Bus Implementation
class EventBus:
    def __init__(self):
        self.observers: Dict[str, List[EventObserver]] = {}

    def subscribe(self, event_type: str, observer: EventObserver):
        if event_type not in self.observers:
            self.observers[event_type] = []
        self.observers[event_type].append(observer)

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        if event_type in self.observers:
            for observer in self.observers[event_type]:
                try:
                    await observer.on_event(event_type, payload)
                except Exception as e:
                    logger.error(f"Observer failed to handle event: {e}")

# Usage
event_bus = EventBus()
sentiment_observer = NewsCollectionObserver(sentiment_service)

event_bus.subscribe("news_collected", sentiment_observer)

# Publish event
await event_bus.publish_event("news_collected", {
    "source": "coindesk",
    "articles_count": 25,
    "collection_time": datetime.utcnow().isoformat()
})
```

**Benefits**:

- **Loose Coupling**: Producers don't know about consumers
- **Extensibility**: Easy addition of new observers
- **Maintainability**: Isolated event handling logic
- **Testing**: Independent observer testing

### 5. **Facade Pattern**

**Purpose**: Provide simplified interface to complex subsystems.

**Implementation**:

```python
# News Collection Facade
class NewsCollectionFacade:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rss_collector = RSSNewsCollector(config["rss"])
        self.api_collector = APINewsCollector(config["api"])
        self.repository = MongoNewsRepository(config["mongodb"])
        self.event_bus = EventBus()

    async def collect_news_from_all_sources(self, keywords: List[str], max_articles: int) -> Dict[str, Any]:
        """Simplified interface for collecting news from all sources."""
        try:
            results = {
                "rss_sources": {},
                "api_sources": {},
                "total_articles": 0,
                "errors": []
            }

            # Collect from RSS sources
            for source in self.config["rss"]["sources"]:
                try:
                    articles = await self.rss_collector.collect_from_source(
                        source, keywords, max_articles
                    )
                    results["rss_sources"][source] = len(articles)
                    results["total_articles"] += len(articles)

                    # Save articles
                    await self.repository.save_articles(articles)

                except Exception as e:
                    error_msg = f"RSS collection failed for {source}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

            # Collect from API sources
            for source in self.config["api"]["sources"]:
                try:
                    articles = await self.api_collector.collect_from_source(
                        source, keywords, max_articles
                    )
                    results["api_sources"][source] = len(articles)
                    results["total_articles"] += len(articles)

                    # Save articles
                    await self.repository.save_articles(articles)

                except Exception as e:
                    error_msg = f"API collection failed for {source}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

            # Publish collection completion event
            if results["total_articles"] > 0:
                await self.event_bus.publish_event("news_collection_completed", {
                    "total_articles": results["total_articles"],
                    "sources": {**results["rss_sources"], **results["api_sources"]},
                    "collection_time": datetime.utcnow().isoformat()
                })

            return results

        except Exception as e:
            logger.error(f"News collection facade failed: {e}")
            return {
                "rss_sources": {},
                "api_sources": {},
                "total_articles": 0,
                "errors": [f"Facade error: {e}"]
            }

# Usage
facade = NewsCollectionFacade(config)
results = await facade.collect_news_from_all_sources(["bitcoin", "ethereum"], 100)
print(f"Collected {results['total_articles']} articles from {len(results['rss_sources']) + len(results['api_sources'])} sources")
```

**Benefits**:

- **Simplified Interface**: Complex operations hidden behind simple methods
- **Decoupling**: Clients don't need to know subsystem details
- **Maintainability**: Changes isolated to facade implementation
- **Testing**: Easier testing of high-level operations

### 6. **Dependency Injection Pattern**

**Purpose**: Invert dependencies and enable loose coupling.

**Implementation**:

```python
# Dependency Injection Container
class ServiceContainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all services with proper dependencies."""
        # Initialize core services
        self.services["event_bus"] = EventBus()
        self.services["cache"] = RedisCache(self.config["redis_url"])

        # Initialize repositories
        self.services["news_repository"] = MongoNewsRepository(
            self.config["mongodb_url"],
            self.config["database_name"]
        )

        self.services["market_data_repository"] = MongoDBMarketDataRepository(
            self.config["mongodb_url"],
            self.config["database_name"]
        )

        # Initialize business services
        self.services["news_service"] = NewsService(
            self.services["news_repository"],
            self.services["event_bus"],
            self.services["cache"]
        )

        self.services["market_data_service"] = MarketDataService(
            self.services["market_data_repository"],
            self.services["event_bus"],
            self.services["cache"]
        )

        # Initialize external adapters
        self.services["binance_adapter"] = BinanceMarketDataAdapter(
            self.config["binance"]
        )

        self.services["tavily_adapter"] = TavilySearchAdapter(
            self.config["tavily"]
        )

    def get_service(self, service_name: str):
        """Get service by name."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        return self.services[service_name]

    async def start_all_services(self):
        """Start all services."""
        try:
            # Start services that need initialization
            for service_name, service in self.services.items():
                if hasattr(service, 'start'):
                    await service.start()

            logger.info("All services started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            return False

# Service with Dependency Injection
class NewsService:
    def __init__(self, repository: NewsRepository, event_bus: EventBus, cache: Cache):
        self.repository = repository
        self.event_bus = event_bus
        self.cache = cache

    async def collect_news(self, request: NewsCollectionRequest) -> NewsCollectionResult:
        """Collect news with injected dependencies."""
        try:
            # Check cache first
            cache_key = f"news_collection:{hash(str(request))}"
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                return NewsCollectionResult(**cached_result)

            # Collect news
            articles = await self._collect_from_sources(request)

            # Save to repository
            saved_count = await self.repository.save_articles(articles)

            # Create result
            result = NewsCollectionResult(
                success=True,
                articles_collected=saved_count,
                collection_time=datetime.utcnow()
            )

            # Cache result
            await self.cache.set(cache_key, result.dict(), ttl=300)

            # Publish event
            await self.event_bus.publish_event("news_collected", {
                "articles_count": saved_count,
                "source": request.source
            })

            return result

        except Exception as e:
            logger.error(f"News collection failed: {e}")
            return NewsCollectionResult(
                success=False,
                error=str(e),
                articles_collected=0
            )

# Usage
container = ServiceContainer(config)
await container.start_all_services()

news_service = container.get_service("news_service")
result = await news_service.collect_news(NewsCollectionRequest(
    source="coindesk",
    keywords=["bitcoin", "crypto"],
    max_articles=100
))
```

**Benefits**:

- **Loose Coupling**: Services don't create their own dependencies
- **Testability**: Easy mocking of dependencies for testing
- **Flexibility**: Runtime dependency configuration
- **Maintainability**: Centralized dependency management

## 🔧 Advanced Patterns

### 1. **Decorator Pattern**

**Purpose**: Add functionality without modifying existing classes.

**Implementation**:

```python
# Caching Decorator
class CachingDecorator:
    def __init__(self, cache: Cache, ttl: int = 300):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Check cache
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await self.cache.set(cache_key, result, self.ttl)

            return result

        return wrapper

# Usage
@CachingDecorator(redis_cache, ttl=600)
async def get_market_data(symbol: str, timeframe: str) -> List[OHLCVSchema]:
    # Expensive market data retrieval
    return await market_data_repository.get_ohlcv(symbol, timeframe)
```

### 2. **Chain of Responsibility Pattern**

**Purpose**: Process requests through a chain of handlers.

**Implementation**:

```python
# News Processing Chain
class NewsProcessor(ABC):
    def __init__(self):
        self.next_processor = None

    def set_next(self, processor: 'NewsProcessor'):
        self.next_processor = processor
        return processor

    @abstractmethod
    async def process(self, article: NewsArticle) -> NewsArticle:
        pass

class ContentFilterProcessor(NewsProcessor):
    async def process(self, article: NewsArticle) -> NewsArticle:
        # Filter inappropriate content
        if self._is_inappropriate(article.content):
            article.status = "filtered"
            return article

        if self.next_processor:
            return await self.next_processor.process(article)
        return article

class SentimentProcessor(NewsProcessor):
    async def process(self, article: NewsArticle) -> NewsArticle:
        # Add sentiment analysis
        article.sentiment = await self._analyze_sentiment(article.content)

        if self.next_processor:
            return await self.next_processor.process(article)
        return article

class TaggingProcessor(NewsProcessor):
    async def process(self, article: NewsArticle) -> NewsArticle:
        # Add automatic tags
        article.tags = await self._extract_tags(article.content)

        if self.next_processor:
            return await self.next_processor.process(article)
        return article

# Usage
filter_processor = ContentFilterProcessor()
sentiment_processor = SentimentProcessor()
tagging_processor = TaggingProcessor()

# Build chain
filter_processor.set_next(sentiment_processor).set_next(tagging_processor)

# Process article
processed_article = await filter_processor.process(raw_article)
```

## 📊 Pattern Usage Summary

| Pattern                     | Usage Count | Key Benefits            | Implementation Quality |
| --------------------------- | ----------- | ----------------------- | ---------------------- |
| **Factory**                 | High        | Runtime object creation | Excellent              |
| **Repository**              | High        | Data access abstraction | Excellent              |
| **Strategy**                | Medium      | Algorithm flexibility   | Good                   |
| **Observer**                | Medium      | Event handling          | Good                   |
| **Facade**                  | Medium      | Simplified interfaces   | Good                   |
| **Dependency Injection**    | High        | Loose coupling          | Excellent              |
| **Decorator**               | Low         | Function enhancement    | Good                   |
| **Chain of Responsibility** | Low         | Request processing      | Good                   |

## 🎯 Pattern Benefits

### **Architectural Benefits**

- **Maintainability**: Clear separation of concerns
- **Testability**: Easy mocking and unit testing
- **Flexibility**: Runtime behavior changes
- **Extensibility**: Easy addition of new features

### **Development Benefits**

- **Code Reuse**: Common patterns across services
- **Consistency**: Uniform implementation approach
- **Documentation**: Self-documenting code structure
- **Onboarding**: Easier for new developers

### **Operational Benefits**

- **Monitoring**: Clear component boundaries
- **Debugging**: Isolated failure points
- **Scaling**: Independent component scaling
- **Deployment**: Granular deployment control

---

_The extensive use of design patterns in FinSight demonstrates mature software engineering practices, resulting in a maintainable, testable, and extensible codebase that can evolve with changing requirements._
