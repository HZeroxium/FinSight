# 6. Advanced Features

## 🚀 Overview

The FinSight platform implements several advanced features that enhance system reliability, performance, and user experience. These features demonstrate production-ready engineering practices and address real-world operational challenges.

## 🔄 Event-Driven Architecture

### **Message Queue Integration**

**RabbitMQ Implementation**:

- **Reliable Message Delivery**: Persistent messages with acknowledgment
- **Dead Letter Queues**: Failed message handling and analysis
- **Message Routing**: Topic-based routing for different event types
- **Consumer Management**: Automatic consumer scaling and health monitoring

**Event Types**:

- **Market Data Events**: OHLCV data collection completion
- **News Events**: Article collection and processing status
- **Sentiment Events**: Analysis completion and results
- **Training Events**: Model training progress and completion

### **Event Processing Pipeline**

```python
# Event Consumer Service
class EventConsumerService:
    def __init__(self, broker: RabbitMQBroker, handlers: Dict[str, EventHandler]):
        self.broker = broker
        self.handlers = handlers
        self.consumers = {}

    async def start_consuming(self, queue_name: str):
        """Start consuming events from specified queue."""
        try:
            # Declare queue with dead letter configuration
            queue = await self.broker.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "finsight.dlx",
                    "x-dead-letter-routing-key": "dead.letter",
                    "x-message-ttl": 300000  # 5 minutes TTL
                }
            )

            # Bind queue to exchange
            await queue.bind("finsight.events", routing_key=f"*.{queue_name}")

            # Start consuming
            await queue.consume(self._process_message)

            self.consumers[queue_name] = queue
            logger.info(f"Started consuming events from {queue_name}")

        except Exception as e:
            logger.error(f"Failed to start consuming from {queue_name}: {e}")

    async def _process_message(self, message):
        """Process incoming message."""
        async with message.process():
            try:
                # Parse message
                payload = json.loads(message.body.decode())
                event_type = payload.get("event_type")

                # Route to appropriate handler
                if event_type in self.handlers:
                    await self.handlers[event_type].handle(payload)
                    logger.info(f"Successfully processed {event_type} event")
                else:
                    logger.warning(f"No handler found for event type: {event_type}")

            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                # Message will be rejected and sent to dead letter queue
                await message.reject(requeue=False)
```

## 🔒 Advanced Security Features

### **API Key Management**

**Secure Authentication**:

- **Environment Variable Storage**: API keys stored in environment variables
- **Key Rotation**: Support for automatic key rotation
- **Access Control**: Role-based access control for different endpoints
- **Audit Logging**: Comprehensive logging of all authentication attempts

**Rate Limiting**:

- **Per-Client Limits**: Different limits for different client types
- **Sliding Window**: Accurate rate limiting with sliding time windows
- **Queue Management**: Request queuing when limits are exceeded
- **Dynamic Adjustment**: Automatic limit adjustment based on client behavior

### **Data Encryption**

**Encryption Implementation**:

- **At-Rest Encryption**: Database-level encryption for sensitive data
- **In-Transit Encryption**: TLS/SSL for all communications
- **Key Management**: Secure key storage and rotation
- **Audit Trail**: Comprehensive encryption audit logging

## 📊 Advanced Monitoring & Observability

### **Distributed Tracing**

**Request Correlation**:

- **Correlation IDs**: Unique identifiers for request tracking
- **Service Chain Tracking**: Complete request flow across services
- **Performance Metrics**: Response time tracking for each service
- **Error Correlation**: Link errors to specific requests

**Implementation**:

```python
# Tracing Middleware
class TracingMiddleware:
    def __init__(self):
        self.trace_id_generator = uuid.uuid4

    async def __call__(self, request: Request, call_next):
        # Generate trace ID
        trace_id = str(self.trace_id_generator())
        request.state.trace_id = trace_id

        # Add trace headers
        request.headers["X-Trace-ID"] = trace_id

        # Start timing
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Add trace headers to response
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Response-Time"] = str(time.time() - start_time)

            # Log trace information
            logger.info(f"Request completed", extra={
                "trace_id": trace_id,
                "response_time": time.time() - start_time,
                "status_code": response.status_code
            })

            return response

        except Exception as e:
            # Log error with trace information
            logger.error(f"Request failed", extra={
                "trace_id": trace_id,
                "error": str(e),
                "response_time": time.time() - start_time
            })
            raise
```

### **Health Check System**

**Comprehensive Health Monitoring**:

- **Service Health**: Individual service status monitoring
- **Dependency Health**: Database, message queue, and external service health
- **Performance Metrics**: Response time and throughput monitoring
- **Resource Monitoring**: CPU, memory, and disk usage tracking

**Health Check Implementation**:

```python
# Health Check Service
class HealthCheckService:
    def __init__(self, services: Dict[str, Any], dependencies: Dict[str, Any]):
        self.services = services
        self.dependencies = dependencies
        self.health_checks = {}
        self._setup_health_checks()

    def _setup_health_checks(self):
        """Setup health checks for all services and dependencies."""
        # Service health checks
        for service_name, service in self.services.items():
            if hasattr(service, 'health_check'):
                self.health_checks[f"service_{service_name}"] = service.health_check

        # Dependency health checks
        for dep_name, dep in self.dependencies.items():
            if hasattr(dep, 'health_check'):
                self.health_checks[f"dependency_{dep_name}"] = dep.health_check

    async def check_system_health(self) -> SystemHealthStatus:
        """Perform comprehensive system health check."""
        health_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "services": {},
            "dependencies": {},
            "performance_metrics": {}
        }

        # Check services
        for check_name, health_check in self.health_checks.items():
            try:
                if check_name.startswith("service_"):
                    service_name = check_name.replace("service_", "")
                    service = self.services[service_name]
                    result = await health_check(service)
                    health_results["services"][service_name] = result
                elif check_name.startswith("dependency_"):
                    dep_name = check_name.replace("dependency_", "")
                    dep = self.dependencies[dep_name]
                    result = await health_check(dep)
                    health_results["dependencies"][dep_name] = result
            except Exception as e:
                logger.error(f"Health check failed for {check_name}: {e}")
                if check_name.startswith("service_"):
                    service_name = check_name.replace("service_", "")
                    health_results["services"][service_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                else:
                    dep_name = check_name.replace("dependency_", "")
                    health_results["dependencies"][dep_name] = {
                        "status": "error",
                        "error": str(e)
                    }

        # Calculate overall status
        all_healthy = (
            all(status.get("status") == "healthy"
                for status in health_results["services"].values()) and
            all(status.get("status") == "healthy"
                for status in health_results["dependencies"].values())
        )

        health_results["overall_status"] = "healthy" if all_healthy else "degraded"

        return SystemHealthStatus(**health_results)
```

## 🚀 Performance Optimization Features

### **Advanced Caching Strategies**

**Multi-Level Caching**:

- **L1 Cache**: In-memory caching for fastest access
- **L2 Cache**: Redis caching for distributed access
- **Cache Warming**: Proactive cache population
- **Intelligent Eviction**: ML-based cache eviction strategies

**Cache Implementation**:

```python
# Multi-Level Cache
class MultiLevelCache:
    def __init__(self, l1_cache: Dict, l2_cache: Cache, config: Dict[str, Any]):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l1_ttl = config.get("l1_ttl", 60)
        self.l2_ttl = config.get("l2_ttl", 300)
        self.cache_stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Check L1 cache first
        if key in self.l1_cache:
            value, timestamp = self.l1_cache[key]
            if time.time() - timestamp < self.l1_ttl:
                self.cache_stats["l1_hits"] += 1
                return value
            else:
                del self.l1_cache[key]

        # Check L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Update L1 cache
            self.l1_cache[key] = (value, time.time())
            self.cache_stats["l2_hits"] += 1
            return value

        self.cache_stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in multi-level cache."""
        # Set in L1 cache
        self.l1_cache[key] = (value, time.time())

        # Set in L2 cache
        await self.l2_cache.set(key, value, ttl or self.l2_ttl)

    def get_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = sum(self.cache_stats.values())
        if total_requests > 0:
            l1_hit_rate = self.cache_stats["l1_hits"] / total_requests
            l2_hit_rate = self.cache_stats["l2_hits"] / total_requests
            overall_hit_rate = (self.cache_stats["l1_hits"] + self.cache_stats["l2_hits"]) / total_requests
        else:
            l1_hit_rate = l2_hit_rate = overall_hit_rate = 0

        return {
            **self.cache_stats,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "overall_hit_rate": overall_hit_rate
        }
```

### **Connection Pooling**

**Database Connection Management**:

- **Connection Pooling**: Efficient database connection management
- **Health Monitoring**: Connection health and performance monitoring
- **Automatic Recovery**: Automatic connection recovery from failures
- **Load Balancing**: Connection distribution across database instances

**Connection Pool Implementation**:

```python
# Database Connection Pool
class DatabaseConnectionPool:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = None
        self.health_check_interval = config.get("health_check_interval", 30)
        self.max_pool_size = config.get("max_pool_size", 50)
        self.min_pool_size = config.get("min_pool_size", 10)
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.pool = AsyncIOMotorClient(
                self.config["connection_string"],
                maxPoolSize=self.max_pool_size,
                minPoolSize=self.min_pool_size,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )

            logger.info("Database connection pool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    async def get_connection(self):
        """Get connection from pool."""
        try:
            if not self.pool:
                self._initialize_pool()

            return self.pool

        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise

    async def check_pool_health(self) -> Dict[str, Any]:
        """Check connection pool health."""
        try:
            if not self.pool:
                return {"status": "unhealthy", "error": "Pool not initialized"}

            # Test connection
            await self.pool.admin.command('ping')

            # Get pool statistics
            pool_stats = self.pool.options.pool_options

            return {
                "status": "healthy",
                "max_pool_size": pool_stats.max_pool_size,
                "min_pool_size": pool_stats.min_pool_size,
                "active_connections": len(self.pool._topology._servers)
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close_pool(self):
        """Close connection pool."""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Database connection pool closed successfully")

        except Exception as e:
            logger.error(f"Failed to close connection pool: {e}")
```

## 🔄 Fault Tolerance & Resilience

### **Circuit Breaker Pattern**

**Implementation**:

- **Failure Threshold**: Configurable failure thresholds
- **Recovery Timeout**: Automatic recovery after timeout
- **Half-Open State**: Gradual recovery testing
- **Monitoring**: Comprehensive circuit breaker metrics

**Circuit Breaker Usage**:

```python
# Circuit Breaker for External APIs
class ExternalAPICircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.metrics = {"total_requests": 0, "successful_requests": 0, "failed_requests": 0}

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.metrics["total_requests"] += 1

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.metrics["successful_requests"] += 1

        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker recovered, transitioning to CLOSED state")

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.metrics["failed_requests"] += 1

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            **self.metrics,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_rate": (
                self.metrics["successful_requests"] /
                max(self.metrics["total_requests"], 1)
            )
        }
```

### **Retry Pattern with Exponential Backoff**

**Implementation**:

- **Exponential Backoff**: Increasing delay between retries
- **Maximum Retries**: Configurable retry limits
- **Jitter**: Random delay variation to prevent thundering herd
- **Retry Conditions**: Configurable conditions for retry attempts

**Retry Implementation**:

```python
# Retry Handler with Exponential Backoff
class RetryHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_conditions: List[Callable] = None,
        **kwargs
    ):
        """Execute function with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Check if retry is needed based on result
                if retry_conditions:
                    should_retry = any(condition(result) for condition in retry_conditions)
                    if should_retry and attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(f"Retry condition met, retrying in {delay}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
                        continue

                return result

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)

        # Add jitter (random variation)
        jitter = random.uniform(0, 0.1 * delay)
        delay += jitter

        return delay
```

## 📈 Advanced Analytics Features

### **Real-Time Data Processing**

**Stream Processing**:

- **Real-Time Aggregation**: Live calculation of market metrics
- **Event Correlation**: Real-time correlation of market events
- **Alert Generation**: Automatic alert generation for market conditions
- **Performance Monitoring**: Real-time performance tracking

**Stream Processing Implementation**:

```python
# Real-Time Market Data Processor
class RealTimeMarketDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregators = {}
        self.alert_generators = []
        self._setup_aggregators()
        self._setup_alert_generators()

    def _setup_aggregators(self):
        """Setup real-time aggregators."""
        # Moving average aggregator
        self.aggregators["sma"] = MovingAverageAggregator(
            window_size=self.config.get("sma_window", 20)
        )

        # Volatility aggregator
        self.aggregators["volatility"] = VolatilityAggregator(
            window_size=self.config.get("volatility_window", 20)
        )

        # Volume aggregator
        self.aggregators["volume"] = VolumeAggregator(
            window_size=self.config.get("volume_window", 20)
        )

    def _setup_alert_generators(self):
        """Setup alert generators."""
        # Price change alerts
        self.alert_generators.append(PriceChangeAlertGenerator(
            threshold=self.config.get("price_change_threshold", 0.05)
        ))

        # Volume spike alerts
        self.alert_generators.append(VolumeSpikeAlertGenerator(
            threshold=self.config.get("volume_spike_threshold", 2.0)
        ))

        # Volatility alerts
        self.alert_generators.append(VolatilityAlertGenerator(
            threshold=self.config.get("volatility_threshold", 0.02)
        ))

    async def process_market_data(self, market_data: OHLCVSchema):
        """Process incoming market data in real-time."""
        try:
            # Update aggregators
            for name, aggregator in self.aggregators.items():
                aggregator.update(market_data)

            # Check for alerts
            alerts = []
            for alert_generator in self.alert_generators:
                alert = alert_generator.check_alert(market_data, self.aggregators)
                if alert:
                    alerts.append(alert)

            # Process alerts
            if alerts:
                await self._process_alerts(alerts)

            # Update real-time metrics
            await self._update_real_time_metrics(market_data, self.aggregators)

        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")

    async def _process_alerts(self, alerts: List[Alert]):
        """Process generated alerts."""
        for alert in alerts:
            try:
                # Log alert
                logger.warning(f"Market alert: {alert.message}", extra={
                    "alert_type": alert.alert_type,
                    "symbol": alert.symbol,
                    "timestamp": alert.timestamp,
                    "severity": alert.severity
                })

                # Send notification
                await self._send_notification(alert)

                # Store alert
                await self._store_alert(alert)

            except Exception as e:
                logger.error(f"Failed to process alert: {e}")

    async def _update_real_time_metrics(self, market_data: OHLCVSchema, aggregators: Dict):
        """Update real-time market metrics."""
        metrics = {
            "timestamp": market_data.timestamp,
            "symbol": market_data.symbol,
            "price": market_data.close,
            "sma_20": aggregators["sma"].get_current_value(),
            "volatility": aggregators["volatility"].get_current_value(),
            "volume_sma": aggregators["volume"].get_current_value()
        }

        # Store metrics
        await self._store_real_time_metrics(metrics)
```

## 🔧 Configuration Management

### **Dynamic Configuration**

**Runtime Configuration Updates**:

- **Hot Reloading**: Configuration changes without service restart
- **Validation**: Configuration validation before application
- **Rollback**: Automatic rollback on configuration errors
- **Audit Trail**: Configuration change tracking

**Configuration Management Implementation**:

```python
# Dynamic Configuration Manager
class DynamicConfigurationManager:
    def __init__(self, config_path: str, validation_schema: Dict[str, Any]):
        self.config_path = config_path
        self.validation_schema = validation_schema
        self.current_config = {}
        self.config_watchers = []
        self._load_configuration()
        self._start_config_watcher()

    def _load_configuration(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            # Validate configuration
            validated_config = self._validate_configuration(config_data)

            # Update current configuration
            self.current_config = validated_config

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema."""
        try:
            # Use Pydantic for validation
            validated_config = ConfigModel(**config_data)
            return validated_config.dict()

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.current_config.get(key, default)

    def add_watcher(self, watcher: Callable):
        """Add configuration change watcher."""
        self.config_watchers.append(watcher)

    def _notify_watchers(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notify all watchers of configuration changes."""
        for watcher in self.config_watchers:
            try:
                watcher(old_config, new_config)
            except Exception as e:
                logger.error(f"Configuration watcher failed: {e}")

    def _start_config_watcher(self):
        """Start watching for configuration file changes."""
        async def watch_config():
            last_modified = os.path.getmtime(self.config_path)

            while True:
                try:
                    await asyncio.sleep(5)  # Check every 5 seconds

                    current_modified = os.path.getmtime(self.config_path)
                    if current_modified > last_modified:
                        logger.info("Configuration file changed, reloading...")

                        old_config = self.current_config.copy()
                        self._load_configuration()

                        # Notify watchers
                        self._notify_watchers(old_config, self.current_config)

                        last_modified = current_modified

                except Exception as e:
                    logger.error(f"Configuration watcher error: {e}")
                    await asyncio.sleep(30)  # Wait longer on error

        # Start watcher in background
        asyncio.create_task(watch_config())
```

## 📊 Performance Benchmarking

### **Load Testing Framework**

**Comprehensive Testing**:

- **API Load Testing**: Test API endpoints under various loads
- **Database Performance**: Database query performance testing
- **Memory Usage**: Memory consumption monitoring
- **Response Time**: Response time distribution analysis

**Load Testing Implementation**:

```python
# Load Testing Service
class LoadTestingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_results = []

    async def run_api_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Dict[str, Any] = None,
        concurrent_users: int = 10,
        duration: int = 60
    ) -> LoadTestResult:
        """Run load test on API endpoint."""
        try:
            start_time = time.time()
            end_time = start_time + duration

            # Create test tasks
            tasks = []
            for user_id in range(concurrent_users):
                task = self._simulate_user_requests(
                    user_id, endpoint, method, payload, end_time
                )
                tasks.append(task)

            # Run concurrent requests
            user_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            all_requests = []
            for user_result in user_results:
                if isinstance(user_result, list):
                    all_requests.extend(user_result)

            # Calculate metrics
            metrics = self._calculate_performance_metrics(all_requests)

            result = LoadTestResult(
                endpoint=endpoint,
                method=method,
                concurrent_users=concurrent_users,
                duration=duration,
                total_requests=len(all_requests),
                metrics=metrics,
                timestamp=datetime.utcnow()
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            logger.error(f"Load test failed: {e}")
            raise

    async def _simulate_user_requests(
        self,
        user_id: int,
        endpoint: str,
        method: str,
        payload: Dict[str, Any],
        end_time: float
    ) -> List[RequestResult]:
        """Simulate user making requests."""
        requests = []

        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                try:
                    # Make request
                    request_start = time.time()

                    if method == "GET":
                        async with session.get(endpoint) as response:
                            response_data = await response.text()
                    elif method == "POST":
                        async with session.post(endpoint, json=payload) as response:
                            response_data = await response.text()
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    request_end = time.time()

                    # Record result
                    request_result = RequestResult(
                        user_id=user_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=request_end - request_start,
                        timestamp=datetime.utcnow(),
                        success=response.status < 400
                    )

                    requests.append(request_result)

                    # Random delay between requests
                    await asyncio.sleep(random.uniform(0.1, 1.0))

                except Exception as e:
                    logger.error(f"Request failed for user {user_id}: {e}")

                    # Record failed request
                    request_result = RequestResult(
                        user_id=user_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=0,
                        response_time=0,
                        timestamp=datetime.utcnow(),
                        success=False,
                        error=str(e)
                    )

                    requests.append(request_result)

        return requests

    def _calculate_performance_metrics(self, requests: List[RequestResult]) -> PerformanceMetrics:
        """Calculate performance metrics from request results."""
        if not requests:
            return PerformanceMetrics()

        # Response times
        response_times = [r.response_time for r in requests if r.success]

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0

        # Success rate
        successful_requests = sum(1 for r in requests if r.success)
        total_requests = len(requests)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        # Requests per second
        if requests:
            first_request = min(requests, key=lambda x: x.timestamp)
            last_request = max(requests, key=lambda x: x.timestamp)
            duration = (last_request.timestamp - first_request.timestamp).total_seconds()
            requests_per_second = total_requests / duration if duration > 0 else 0
        else:
            requests_per_second = 0

        return PerformanceMetrics(
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            success_rate=success_rate,
            requests_per_second=requests_per_second,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_requests - successful_requests
        )
```

---

_These advanced features demonstrate the FinSight platform's production readiness and sophisticated engineering practices. The system provides comprehensive monitoring, fault tolerance, and performance optimization capabilities that enable reliable operation in production environments._
