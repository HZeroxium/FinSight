# FinSight Prediction Service - Architecture & Design

> **Complete System Architecture Documentation and Design Patterns**

## 🌐 Overview

The FinSight Prediction Service is built using **Hexagonal Architecture** (Ports & Adapters) principles, providing a robust, scalable, and maintainable foundation for AI-powered financial time series forecasting.

### Key Architectural Features

- **Clean Architecture**: Clear separation of concerns with dependency inversion
- **Hexagonal Design**: Adapter pattern for external integrations
- **Event-Driven**: Asynchronous processing with message queues
- **Microservices-Ready**: Service discovery and health monitoring
- **Cloud-Native**: Stateless design with externalized configuration

## 🏗️ Architecture Principles

### 1. Separation of Concerns

- **Domain Layer**: Core business logic and entities
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External integrations and persistence
- **Interface Layer**: API endpoints and controllers

### 2. Dependency Inversion

- High-level modules don't depend on low-level modules
- Both depend on abstractions
- Abstractions don't depend on details

### 3. Single Responsibility

- Each module has one reason to change
- Clear boundaries and interfaces
- Focused functionality

### 4. Open/Closed Principle

- Open for extension, closed for modification
- Plugin architecture for serving adapters
- Configurable fallback strategies

## 🏛️ System Architecture

### High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "External Systems"
        E[Eureka Server]
        S3[S3/Cloud Storage]
        ML[MLflow Server]
        TR[Triton Server]
        TS[TorchServe]
    end

    subgraph "FinSight Prediction Service"
        subgraph "API Layer"
            F[FastAPI App]
            R[API Routers]
        end

        subgraph "Service Layer"
            TS[Training Service]
            PS[Prediction Service]
            DS[Dataset Service]
            CS[Cloud Storage Service]
        end

        subgraph "Facade Layer"
            UF[Unified Model Facade]
            TF[Training Facade]
            SF[Serving Facade]
        end

        subgraph "Adapter Layer"
            SA[Serving Adapters]
            ET[Experiment Trackers]
            DL[Data Loaders]
        end

        subgraph "Core Layer"
            M[Models]
            C[Configuration]
            U[Utilities]
        end
    end

    F --> R
    R --> TS
    R --> PS
    R --> DS
    R --> CS

    TS --> UF
    PS --> UF
    DS --> UF

    UF --> TF
    UF --> SF

    SF --> SA
    TF --> ET
    DS --> DL

    SA --> TR
    SA --> TS
    ET --> ML
    DL --> S3

    UF --> E
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Service
    participant Facade
    participant Adapter
    participant External

    Client->>API: Request
    API->>Service: Process
    Service->>Facade: Orchestrate
    Facade->>Adapter: Execute
    Adapter->>External: Call
    External-->>Adapter: Response
    Adapter-->>Facade: Result
    Facade-->>Service: Data
    Service-->>API: Response
    API-->>Client: Result
```

## 🧩 Module Architecture

### 1. API Layer (`src/routers/`)

**Purpose**: HTTP endpoint definitions and request/response handling

**Components**:

- **Training Router**: Model training endpoints
- **Prediction Router**: Prediction endpoints
- **Models Router**: Model management endpoints
- **Serving Router**: Model serving endpoints
- **Datasets Router**: Dataset management endpoints
- **Cloud Storage Router**: Cloud storage operations
- **Eureka Router**: Service discovery management
- **Cleanup Router**: System maintenance endpoints

**Design Pattern**: Controller pattern with dependency injection

```python
@router.post("/train")
async def train_model(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service)
) -> TrainingResponse:
    return await training_service.start_training(request)
```

### 2. Service Layer (`src/services/`)

**Purpose**: Business logic orchestration and workflow management

**Components**:

- **TrainingService**: Model training orchestration
- **PredictionService**: Prediction workflow management
- **DatasetManagementService**: Data lifecycle management
- **CloudStorageService**: Cloud storage operations
- **EurekaClientService**: Service discovery management
- **BackgroundTaskManager**: Asynchronous task management

**Design Pattern**: Service layer pattern with async support

```python
class TrainingService:
    async def start_async_training(self, request: AsyncTrainingRequest) -> AsyncTrainingResponse:
        # Validate request
        # Check data availability
        # Create training job
        # Start background training
        # Return job ID
```

### 3. Facade Layer (`src/facades/`)

**Purpose**: Unified interface for complex subsystems

**Components**:

- **UnifiedModelFacade**: Combined training and serving interface
- **ModelTrainingFacade**: Training-specific operations
- **ModelServingFacade**: Serving-specific operations

**Design Pattern**: Facade pattern with dependency injection

```python
class UnifiedModelFacade:
    def __init__(self, serving_adapter: Optional[IModelServingAdapter] = None):
        self.training = ModelTrainingFacade()
        self.serving = ModelServingFacade(serving_adapter)

    async def train_model(self, ...) -> Dict[str, Any]:
        return await self.training.train_model(...)

    async def predict(self, ...) -> Dict[str, Any]:
        return await self.serving.predict(...)
```

### 4. Adapter Layer (`src/adapters/`)

**Purpose**: External system integration and abstraction

**Components**:

- **Serving Adapters**: Model serving backends
- **Experiment Trackers**: ML experiment management
- **Data Loaders**: Data source integration

**Design Pattern**: Adapter pattern with strategy selection

```python
class ServingAdapterFactory:
    _adapter_classes = {
        ServingAdapterType.SIMPLE: SimpleServingAdapter,
        ServingAdapterType.TRITON: TritonServingAdapter,
        ServingAdapterType.TORCHSERVE: TorchServeAdapter,
        ServingAdapterType.TORCHSCRIPT: TorchScriptServingAdapter,
    }

    @classmethod
    def create_adapter(cls, adapter_type: str, config: Dict[str, Any]) -> IModelServingAdapter:
        adapter_class = cls._adapter_classes.get(ServingAdapterType(adapter_type))
        return adapter_class(config)
```

### 5. Core Layer (`src/core/`, `src/models/`, `src/utils/`)

**Purpose**: Core business logic, models, and utilities

**Components**:

- **Configuration**: Environment-based settings
- **Models**: Time series model implementations
- **Utilities**: Common functionality and helpers

**Design Pattern**: Core domain pattern

```python
class BaseTimeSeriesAdapter(ITimeSeriesModel):
    def __init__(self, context_length: int = 64, prediction_length: int = 1, ...):
        self.context_length = context_length
        self.prediction_length = prediction_length
        # Initialize common components

    async def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, ...) -> Dict[str, Any]:
        # Common training logic
        return await self._train_model(train_data, val_data, ...)

    @abstractmethod
    def _train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, ...) -> Dict[str, Any]:
        pass
```

## 🔄 Data Flow

### 1. Training Workflow

```mermaid
flowchart TD
    A[Training Request] --> B[Validate Input]
    B --> C[Check Data Availability]
    C --> D[Create Training Job]
    D --> E[Queue Training Job]
    E --> F[Background Training]
    F --> G[Load Training Data]
    G --> H[Feature Engineering]
    H --> I[Model Training]
    I --> J[Validation]
    J --> K[Save Model]
    K --> L[Update Job Status]
    L --> M[Training Complete]

    F --> N[Progress Updates]
    N --> O[Update Job Progress]

    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style F fill:#fff3e0
```

### 2. Prediction Workflow

```mermaid
flowchart TD
    A[Prediction Request] --> B[Validate Input]
    B --> C[Model Selection]
    C --> D{Model Found?}
    D -->|Yes| E[Load Model]
    D -->|No| F[Apply Fallback Strategy]
    F --> G[Select Alternative Model]
    G --> E
    E --> H[Data Preprocessing]
    H --> I[Feature Engineering]
    I --> J[Model Inference]
    J --> K[Post-processing]
    K --> L[Format Response]
    L --> M[Return Prediction]

    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style F fill:#fff3e0
```

### 3. Data Management Flow

```mermaid
flowchart TD
    A[Data Request] --> B{Local Available?}
    B -->|Yes| C[Load Local Data]
    B -->|No| D{Cloud Available?}
    D -->|Yes| E[Download from Cloud]
    D -->|No| F[Return Not Found]
    E --> G[Cache Data]
    G --> C
    C --> H[Validate Data]
    H --> I[Return Data]

    style A fill:#e1f5fe
    style I fill:#c8e6c9
    style E fill:#fff3e0
```

## 🎨 Design Patterns

### 1. Factory Pattern

**Usage**: Creating serving adapters and model instances

```python
class ServingAdapterFactory:
    @classmethod
    def create_adapter(cls, adapter_type: str, config: Dict[str, Any]) -> IModelServingAdapter:
        adapter_class = cls._adapter_classes.get(ServingAdapterType(adapter_type))
        if not adapter_class:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        return adapter_class(config)
```

### 2. Strategy Pattern

**Usage**: Fallback strategies and serving backends

```python
class FallbackStrategy(Enum):
    NONE = "none"
    TIMEFRAME_ONLY = "timeframe_only"
    SYMBOL_ONLY = "symbol_only"
    TIMEFRAME_AND_SYMBOL = "timeframe_and_symbol"

class ModelFallbackUtils:
    async def select_model_with_fallback(self, strategy: FallbackStrategy, ...) -> ModelSelectionResult:
        if strategy == FallbackStrategy.TIMEFRAME_AND_SYMBOL:
            return await self._apply_timeframe_and_symbol_fallback(...)
        elif strategy == FallbackStrategy.TIMEFRAME_ONLY:
            return await self._apply_timeframe_fallback(...)
        # ... other strategies
```

### 3. Facade Pattern

**Usage**: Simplified interface for complex subsystems

```python
class UnifiedModelFacade:
    def __init__(self, serving_adapter: Optional[IModelServingAdapter] = None):
        self.training = ModelTrainingFacade()
        self.serving = ModelServingFacade(serving_adapter)

    async def train_and_predict(self, ...) -> Dict[str, Any]:
        # Complex orchestration hidden behind simple interface
        training_result = await self.training.train_model(...)
        prediction_result = await self.serving.predict(...)
        return self._combine_results(training_result, prediction_result)
```

### 4. Repository Pattern

**Usage**: Data access abstraction

```python
class TrainingJobRepositoryInterface(ABC):
    @abstractmethod
    async def create_job(self, job_info: TrainingJobInfo) -> bool:
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[TrainingJobInfo]:
        pass

class FileTrainingJobRepository(TrainingJobRepositoryInterface):
    async def create_job(self, job_info: TrainingJobInfo) -> bool:
        # File-based implementation

class RedisTrainingJobRepository(TrainingJobRepositoryInterface):
    async def create_job(self, job_info: TrainingJobInfo) -> bool:
        # Redis-based implementation
```

### 5. Observer Pattern

**Usage**: Progress tracking and event notification

```python
class TrainingProgressObserver:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_progress_update(self, progress: float, stage: str):
        # Notify progress updates
        asyncio.create_task(self._notify_progress(progress, stage))

    def on_completion(self, result: Dict[str, Any]):
        # Notify completion
        asyncio.create_task(self._notify_completion(result))
```

## 🛠️ Technology Stack

### Core Framework

- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for production deployment

### Machine Learning

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **PyTorch Lightning**: Training framework
- **MLflow**: Experiment tracking and model registry

### Data Processing

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

### Serving & Inference

- **Triton Inference Server**: NVIDIA's inference server
- **TorchServe**: PyTorch model serving
- **TorchScript**: Optimized model format

### Storage & Infrastructure

- **Redis**: In-memory data store and job queue
- **MinIO**: S3-compatible object storage
- **AWS S3**: Cloud object storage
- **DigitalOcean Spaces**: S3-compatible storage

### Monitoring & Observability

- **Eureka**: Service discovery and registration
- **Custom Logging**: Structured logging with correlation IDs
- **Health Checks**: Comprehensive health monitoring

## 🔗 Dependencies

### External Dependencies

```mermaid
graph TB
    subgraph "FinSight Prediction Service"
        PS[Prediction Service]
    end

    subgraph "External Services"
        E[Eureka Server]
        S3[S3/Cloud Storage]
        ML[MLflow Server]
        TR[Triton Server]
        TS[TorchServe]
        R[Redis]
    end

    subgraph "Data Sources"
        B[Binance API]
        M[Market Data Service]
    end

    PS --> E
    PS --> S3
    PS --> ML
    PS --> TR
    PS --> TS
    PS --> R
    PS --> B
    PS --> M

    style PS fill:#e3f2fd
    style E fill:#f3e5f5
    style S3 fill:#e8f5e8
    style ML fill:#fff3e0
    style TR fill:#fce4ec
    style TS fill:#f1f8e9
    style R fill:#fafafa
    style B fill:#e0f2f1
    style M fill:#f9fbe7
```

### Internal Dependencies

```mermaid
graph TB
    subgraph "API Layer"
        R[API Routers]
    end

    subgraph "Service Layer"
        TS[Training Service]
        PS[Prediction Service]
        DS[Dataset Service]
    end

    subgraph "Facade Layer"
        UF[Unified Facade]
    end

    subgraph "Adapter Layer"
        SA[Serving Adapters]
        ET[Experiment Trackers]
        DL[Data Loaders]
    end

    subgraph "Core Layer"
        C[Configuration]
        M[Models]
        U[Utilities]
    end

    R --> TS
    R --> PS
    R --> DS

    TS --> UF
    PS --> UF
    DS --> UF

    UF --> SA
    UF --> ET
    UF --> DL

    SA --> C
    ET --> C
    DL --> C

    UF --> M
    UF --> U

    style R fill:#e1f5fe
    style TS fill:#f3e5f5
    style PS fill:#e8f5e8
    style DS fill:#fff3e0
    style UF fill:#fce4ec
    style SA fill:#f1f8e9
    style ET fill:#fafafa
    style DL fill:#e0f2f1
    style C fill:#f9fbe7
    style M fill:#e8eaf6
    style U fill:#f3e5f5
```

## 📈 Scalability & Performance

### Horizontal Scaling

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end

    subgraph "Service Instances"
        PS1[Prediction Service 1]
        PS2[Prediction Service 2]
        PS3[Prediction Service 3]
    end

    subgraph "Shared Infrastructure"
        R[Redis Cluster]
        S3[S3 Storage]
        E[Eureka Server]
    end

    LB --> PS1
    LB --> PS2
    LB --> PS3

    PS1 --> R
    PS2 --> R
    PS3 --> R

    PS1 --> S3
    PS2 --> S3
    PS3 --> S3

    PS1 --> E
    PS2 --> E
    PS3 --> E

    style LB fill:#e3f2fd
    style PS1 fill:#f3e5f5
    style PS2 fill:#f3e5f5
    style PS3 fill:#f3e5f5
    style R fill:#e8f5e8
    style S3 fill:#fff3e0
    style E fill:#fce4ec
```

### Performance Optimizations

1. **Async Processing**: Full async/await support for I/O operations
2. **Connection Pooling**: Database and storage connection reuse
3. **Model Caching**: Intelligent model loading and eviction
4. **Batch Processing**: Configurable batch sizes for inference
5. **Memory Management**: Automatic memory cleanup and optimization

### Caching Strategy

```mermaid
graph TB
    subgraph "Request Flow"
        R[Request] --> C1[Memory Cache]
        C1 --> C2[Redis Cache]
        C2 --> S[Storage]
    end

    subgraph "Cache Layers"
        C1[In-Memory Cache<br/>Fastest Access]
        C2[Redis Cache<br/>Shared Across Instances]
        S[Persistent Storage<br/>Slowest Access]
    end

    style R fill:#e1f5fe
    style C1 fill:#c8e6c9
    style C2 fill:#fff3e0
    style S fill:#fce4ec
```

## 🔒 Security Architecture

### Security Layers

```mermaid
graph TB
    subgraph "External Layer"
        LB[Load Balancer]
        FW[Firewall]
    end

    subgraph "Application Layer"
        API[FastAPI App]
        VAL[Input Validation]
        AUTH[Authentication]
    end

    subgraph "Data Layer"
        ENC[Data Encryption]
        AUD[Audit Logging]
        ACC[Access Control]
    end

    LB --> FW
    FW --> API
    API --> VAL
    VAL --> AUTH
    AUTH --> ENC
    ENC --> AUD
    AUD --> ACC

    style LB fill:#e3f2fd
    style FW fill:#f3e5f5
    style API fill:#e8f5e8
    style VAL fill:#fff3e0
    style AUTH fill:#fce4ec
    style ENC fill:#f1f8e9
    style AUD fill:#fafafa
    style ACC fill:#e0f2f1
```

### Security Features

1. **Input Validation**: Pydantic schema validation
2. **Rate Limiting**: Configurable API rate limiting
3. **Error Handling**: Secure error responses
4. **Logging**: Security event logging
5. **Data Encryption**: At-rest and in-transit encryption

## 📊 Monitoring & Observability

### Monitoring Architecture

```mermaid
graph TB
    subgraph "Application"
        PS[Prediction Service]
        HC[Health Checks]
        ML[Metrics Logger]
    end

    subgraph "Monitoring Stack"
        PM[Prometheus]
        G[Grafana]
        A[Alert Manager]
    end

    subgraph "Logging Stack"
        FL[Fluentd]
        ES[Elasticsearch]
        K[Kibana]
    end

    PS --> HC
    PS --> ML

    HC --> PM
    ML --> PM

    PS --> FL
    FL --> ES
    ES --> K

    PM --> G
    PM --> A

    style PS fill:#e3f2fd
    style HC fill:#f3e5f5
    style ML fill:#e8f5e8
    style PM fill:#fff3e0
    style G fill:#fce4ec
    style A fill:#f1f8e9
    style FL fill:#fafafa
    style ES fill:#e0f2f1
    style K fill:#f9fbe7
```

### Health Check Endpoints

- **`/health`**: Overall service health
- **`/serving/health`**: Model serving health
- **`/eureka/status`**: Service discovery status
- **`/cleanup/status`**: Background maintenance status

### Metrics Collection

1. **Application Metrics**: Request rates, response times, error rates
2. **Business Metrics**: Training success rates, prediction accuracy
3. **Infrastructure Metrics**: Memory usage, CPU utilization, disk I/O
4. **Custom Metrics**: Model loading times, inference latency

### Logging Strategy

1. **Structured Logging**: JSON format with correlation IDs
2. **Log Levels**: Configurable per component
3. **Log Rotation**: Automatic log file management
4. **Centralized Logging**: Aggregated log collection

## 🚀 Deployment Architecture

### Container Architecture

```mermaid
graph TB
    subgraph "Docker Compose"
        PS[Prediction Service]
        E[Eureka Server]
        R[Redis]
        M[MinIO]
    end

    subgraph "Kubernetes"
        PS_K8S[Prediction Service Pods]
        E_K8S[Eureka Service]
        R_K8S[Redis StatefulSet]
        M_K8S[MinIO StatefulSet]
    end

    PS --> E
    PS --> R
    PS --> M

    PS_K8S --> E_K8S
    PS_K8S --> R_K8S
    PS_K8S --> M_K8S

    style PS fill:#e3f2fd
    style E fill:#f3e5f5
    style R fill:#e8f5e8
    style M fill:#fff3e0
    style PS_K8S fill:#fce4ec
    style E_K8S fill:#f1f8e9
    style R_K8S fill:#fafafa
    style M_K8S fill:#e0f2f1
```

### Environment Configurations

1. **Development**: Local Docker Compose with simple adapters
2. **Staging**: Kubernetes with production-like configuration
3. **Production**: High-availability Kubernetes with external services

## 🔮 Future Architecture

### Planned Enhancements

1. **Event Sourcing**: CQRS pattern for training events
2. **GraphQL API**: Flexible data querying
3. **WebSocket Support**: Real-time prediction streaming
4. **Multi-Tenancy**: Isolated environments per client
5. **Federated Learning**: Distributed model training

### Architecture Evolution

```mermaid
graph LR
    A[Current: Monolithic Service] --> B[Next: Event-Driven]
    B --> C[Future: Microservices]
    C --> D[Target: Serverless Functions]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
```

---

**For more information, see the [Configuration Guide](configuration.md) and [API Documentation](api.md).**
