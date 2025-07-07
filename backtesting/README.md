# FinSight Backtesting API

Advanced backtesting system for cryptocurrency trading strategies with comprehensive data management, strategy execution, and administrative operations.

## 🚀 Features

### Core Capabilities

- **Multi-Strategy Backtesting**: Support for multiple trading strategies (MA Crossover, RSI, Bollinger Bands, MACD, Buy & Hold)
- **Multi-Exchange Support**: Binance integration with extensible framework for additional exchanges
- **Cross-Timeframe Analysis**: Convert and analyze data across different timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- **Pluggable Storage**: Support for CSV, MongoDB, and InfluxDB storage backends
- **Administrative Operations**: Comprehensive data management and system monitoring

### Architecture

Built with **Ports & Adapters (Hexagonal Architecture)** for maximum flexibility:

- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Pluggable storage adapters
- **Strategy Layer**: Extensible trading strategies
- **Adapter Layer**: Exchange connectors and backtesting engines

## 📋 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd FinSight/backtesting

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env
# Edit .env with your settings
```

### 2. Start the Server

```bash
# Start the FastAPI server
python start_server.py

# Or use uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### 3. Access Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **API Root**: <http://localhost:8000/>

### 4. Run Admin Demo

```bash
# Test admin functionality
python admin_demo.py
```

## 🔧 API Endpoints

### Admin Endpoints

- `GET /admin/health` - System health check (public)
- `GET /admin/stats` - System statistics (auth required)
- `POST /admin/data/ensure` - Ensure data availability
- `POST /admin/data/convert-timeframe` - Convert timeframe data
- `POST /admin/data/cleanup` - Clean up old data
- `GET /admin/info` - Admin endpoint documentation

### Market Data Endpoints

- `GET /api/v1/market-data/ohlcv` - Retrieve OHLCV data
- `GET /api/v1/market-data/ohlcv/stats` - Get data statistics
- `GET /api/v1/market-data/ohlcv/gaps` - Identify data gaps
- `GET /api/v1/market-data/exchanges` - List available exchanges
- `GET /api/v1/market-data/symbols` - List available symbols
- `GET /api/v1/market-data/timeframes` - List available timeframes

### Backtesting Endpoints

- `GET /api/v1/backtesting/strategies` - List available strategies
- `GET /api/v1/backtesting/strategies/{name}/config` - Get strategy configuration
- `POST /api/v1/backtesting/run` - Execute backtest
- `GET /api/v1/backtesting/engines` - List backtesting engines

## 🔐 Authentication

Admin endpoints require API key authentication:

```bash
# Set in environment
export ADMIN_API_KEY="your-secure-api-key"

# Or in .env file
AI_PREDICTION_ADMIN_API_KEY=your-secure-api-key

# Use in requests
curl -H "Authorization: Bearer your-secure-api-key" \
     http://localhost:8000/admin/stats
```

## 📊 Data Management

### Supported Exchanges

- **Binance**: Full OHLCV data collection and real-time updates

### Supported Timeframes

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

### Storage Backends

- **CSV**: File-based storage for development and small datasets
- **MongoDB**: Document-based storage for production
- **InfluxDB**: Time-series database for high-performance scenarios

## 🎯 Trading Strategies

### Built-in Strategies

1. **Moving Average Crossover**: Buy/sell signals based on MA crossovers
2. **RSI Strategy**: Overbought/oversold signals using RSI indicator
3. **Bollinger Bands**: Mean reversion strategy using Bollinger Bands
4. **MACD Strategy**: Trend following using MACD indicator
5. **Simple Buy & Hold**: Baseline strategy for comparison

### Custom Strategies

Implement the `Strategy` interface to create custom strategies:

```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, data: List[OHLCVSchema]) -> List[Dict[str, Any]]:
        # Implement your strategy logic
        return signals
```

## 🔧 Configuration

### Environment Variables

Key configuration options (see `.env.example` for complete list):

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Application
AI_PREDICTION_DEBUG=true
AI_PREDICTION_ENVIRONMENT=development
AI_PREDICTION_ADMIN_API_KEY=your-api-key

# Storage
STORAGE_BASE_DIRECTORY=data
MONGODB_URL=mongodb://localhost:27017/

# Data Collection
DEFAULT_SYMBOLS=BTC/USDT,ETH/USDT,BNB/USDT
DEFAULT_TIMEFRAMES=1h,4h,1d
```

### Repository Configuration

Switch between storage backends in code:

```python
# CSV Repository
repository = create_repository("csv", {
    "base_directory": "data"
})

# MongoDB Repository
repository = create_repository("mongodb", {
    "connection_string": "mongodb://localhost:27017/",
    "database_name": "finsight_backtesting"
})

# InfluxDB Repository
repository = create_repository("influxdb", {
    "url": "http://localhost:8086",
    "token": "your-token",
    "org": "your-org",
    "bucket": "market_data"
})
```

## 📈 Usage Examples

### Ensure Data Availability

```bash
curl -X POST "http://localhost:8000/admin/data/ensure" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "force_refresh": false
  }'
```

### Convert Timeframe Data

```bash
curl -X POST "http://localhost:8000/admin/data/convert-timeframe" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "source_timeframe": "1h",
    "target_timeframe": "4h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "save_converted": true
  }'
```

### Run Backtest

```bash
curl -X POST "http://localhost:8000/api/v1/backtesting/run" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "moving_average_crossover",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "initial_capital": 10000,
    "strategy_params": {
      "fast_period": 10,
      "slow_period": 30
    }
  }'
```

### Retrieve Market Data

```bash
curl "http://localhost:8000/api/v1/market-data/ohlcv?exchange=binance&symbol=BTCUSDT&timeframe=1h&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z"
```

## 🏗️ Development

### Project Structure

```plaintext
backtesting/
├── src/
│   ├── adapters/          # External system adapters
│   ├── common/           # Shared utilities (logging, etc.)
│   ├── converters/       # Data format converters
│   ├── core/            # Configuration and settings
│   ├── factories/       # Dependency injection factories
│   ├── interfaces/      # Abstract interfaces (ports)
│   ├── models/          # Data models
│   ├── routers/         # FastAPI route definitions
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Business logic layer
│   ├── strategies/      # Trading strategy implementations
│   ├── utils/           # Utility functions
│   └── app.py          # FastAPI application
├── data/               # Data storage directory
├── logs/               # Log files
├── start_server.py     # Server startup script
├── admin_demo.py       # Admin functionality demo
└── requirements.txt    # Python dependencies
```

### Adding New Strategies

1. Create strategy class in `src/strategies/`
2. Implement the `Strategy` interface
3. Register in `StrategyFactory`
4. Add tests

### Adding New Storage Backends

1. Implement `MarketDataRepository` interface
2. Add to `MarketDataRepositoryFactory`
3. Update configuration schemas

### Running Tests

```bash
# Run unit tests
pytest src/tests/

# Run integration tests
pytest src/tests/integration/

# Run with coverage
pytest --cov=src src/tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**

   - Check Python version (3.8+ required)
   - Install dependencies: `pip install -r requirements.txt`
   - Check port availability: `netstat -an | grep 8000`

2. **Authentication errors**

   - Verify API key in `.env` file
   - Use Bearer token format: `Bearer your-api-key`

3. **Data collection fails**

   - Check Binance API credentials
   - Verify internet connection
   - Check rate limits

4. **Storage errors**
   - Ensure data directory exists and is writable
   - Check MongoDB/InfluxDB connection
   - Verify disk space

### Debug Mode

Enable debug mode for detailed logging:

```bash
export AI_PREDICTION_DEBUG=true
export AI_PREDICTION_LOG_LEVEL=DEBUG
python start_server.py
```

## 📝 License

This project is part of the FinSight platform. See the main repository for license information.

## 🤝 Contributing

1. Follow the existing code patterns
2. Add tests for new functionality
3. Update documentation
4. Use proper type hints and docstrings
5. Follow SOLID principles and design patterns

## 📞 Support

For issues and questions:

- Check the API documentation at `/docs`
- Review the troubleshooting section
- Check logs in the `logs/` directory
- Use the admin demo script to test functionality
