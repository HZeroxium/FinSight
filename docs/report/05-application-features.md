# 5. Application Features

## 🎯 Overview

The FinSight platform implements a comprehensive set of features for financial data analysis, news processing, and AI-powered predictions. Each service provides specific functionality while maintaining integration with the overall system architecture.

## 📊 Market Dataset Service

### **Core Functionality**

The Market Dataset Service is responsible for collecting, storing, and managing financial market data from various sources, primarily cryptocurrency exchanges.

#### **1. Market Data Collection**

**Binance API Integration**:

- **Real-time OHLCV Data**: Collects Open, High, Low, Close, Volume data for cryptocurrency pairs
- **Multiple Timeframes**: Supports 1m, 5m, 15m, 1h, 4h, 1d, 1w timeframes
- **Batch Processing**: Efficient collection of historical data with configurable limits
- **Rate Limiting**: Respects Binance API rate limits with intelligent backoff

**Implementation Details**:

```python
class BinanceMarketDataCollector:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.rate_limiter = RateLimiter(max_requests=1200, time_window=60)

    async def collect_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[OHLCVSchema]:
        """Collect OHLCV data from Binance with rate limiting."""
        try:
            # Convert timeframe to Binance format
            binance_timeframe = self._convert_timeframe(timeframe)

            # Calculate data points needed
            total_points = self._calculate_data_points(start_date, end_date, timeframe)

            # Collect data in batches
            all_data = []
            current_start = start_date

            while current_start < end_date:
                # Respect rate limits
                await self.rate_limiter.wait_if_needed()

                # Get batch of data
                batch_data = self.client.get_klines(
                    symbol=symbol,
                    interval=binance_timeframe,
                    startTime=int(current_start.timestamp() * 1000),
                    limit=1000
                )

                # Convert to internal format
                ohlcv_data = [
                    OHLCVSchema(
                        timestamp=datetime.fromtimestamp(item[0] / 1000),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        symbol=symbol,
                        exchange="binance",
                        timeframe=timeframe
                    )
                    for item in batch_data
                ]

                all_data.extend(ohlcv_data)

                # Update start time for next batch
                if batch_data:
                    last_timestamp = datetime.fromtimestamp(batch_data[-1][0] / 1000)
                    current_start = last_timestamp + timedelta(minutes=1)
                else:
                    break

            return all_data

        except Exception as e:
            logger.error(f"Failed to collect OHLCV data for {symbol}: {e}")
            raise
```

#### **2. Data Storage & Management**

**Multi-Storage Strategy**:

- **MongoDB**: Primary storage for structured market data with indexing
- **InfluxDB**: Time-series optimized storage for high-frequency data
- **CSV/Parquet**: File-based storage for data export and analysis
- **MinIO/S3**: Object storage for large datasets and backups

**Data Validation & Quality**:

```python
class OHLCVValidator:
    @staticmethod
    def validate_ohlcv_data(data: List[OHLCVSchema]) -> ValidationResult:
        """Validate OHLCV data for consistency and quality."""
        errors = []
        warnings = []

        for i, item in enumerate(data):
            # Check OHLC consistency
            if item.high < max(item.open, item.close):
                errors.append(f"Row {i}: High price {item.high} is less than open/close")

            if item.low > min(item.open, item.close):
                errors.append(f"Row {i}: Low price {item.low} is greater than open/close")

            # Check for zero or negative values
            if item.volume <= 0:
                errors.append(f"Row {i}: Invalid volume {item.volume}")

            if item.open <= 0 or item.high <= 0 or item.low <= 0 or item.close <= 0:
                errors.append(f"Row {i}: Invalid price values")

            # Check timestamp ordering
            if i > 0 and item.timestamp <= data[i-1].timestamp:
                errors.append(f"Row {i}: Timestamp {item.timestamp} is not after previous row")

        # Check for data gaps
        gaps = OHLCVValidator._detect_data_gaps(data)
        if gaps:
            warnings.append(f"Detected {len(gaps)} data gaps")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            total_records=len(data)
        )

    @staticmethod
    def _detect_data_gaps(data: List[OHLCVSchema]) -> List[DataGap]:
        """Detect gaps in time series data."""
        gaps = []

        for i in range(1, len(data)):
            expected_timestamp = data[i-1].timestamp + timedelta(minutes=1)
            if data[i].timestamp > expected_timestamp:
                gaps.append(DataGap(
                    start_time=data[i-1].timestamp,
                    end_time=data[i].timestamp,
                    gap_duration=data[i].timestamp - data[i-1].timestamp
                ))

        return gaps
```

#### **3. Data Conversion & Export**

**Timeframe Conversion**:

- **Upsampling**: Convert lower frequency data to higher frequency (e.g., 1h to 1m)
- **Downsampling**: Aggregate higher frequency data to lower frequency (e.g., 1m to 1h)
- **Format Conversion**: Support for CSV, Parquet, JSON export formats

**Implementation**:

```python
class TimeframeConverter:
    @staticmethod
    async def convert_timeframe(
        data: List[OHLCVSchema],
        target_timeframe: str
    ) -> List[OHLCVSchema]:
        """Convert data to target timeframe."""
        if target_timeframe == "1m":
            return await TimeframeConverter._upsample_to_1m(data)
        elif target_timeframe in ["5m", "15m", "1h", "4h", "1d"]:
            return await TimeframeConverter._downsample_to_timeframe(data, target_timeframe)
        else:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

    @staticmethod
    async def _downsample_to_timeframe(
        data: List[OHLCVSchema],
        target_timeframe: str
    ) -> List[OHLCVSchema]:
        """Downsample data to target timeframe."""
        # Group data by target timeframe
        grouped_data = {}

        for item in data:
            # Calculate target timestamp
            target_timestamp = TimeframeConverter._round_to_timeframe(
                item.timestamp, target_timeframe
            )

            if target_timestamp not in grouped_data:
                grouped_data[target_timestamp] = []
            grouped_data[target_timestamp].append(item)

        # Aggregate each group
        result = []
        for timestamp, group in grouped_data.items():
            if group:
                aggregated = TimeframeConverter._aggregate_ohlcv(group)
                aggregated.timestamp = timestamp
                aggregated.timeframe = target_timeframe
                result.append(aggregated)

        return sorted(result, key=lambda x: x.timestamp)

    @staticmethod
    def _aggregate_ohlcv(group: List[OHLCVSchema]) -> OHLCVSchema:
        """Aggregate OHLCV data for a time period."""
        if not group:
            raise ValueError("Cannot aggregate empty group")

        return OHLCVSchema(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(item.high for item in group),
            low=min(item.low for item in group),
            close=group[-1].close,
            volume=sum(item.volume for item in group),
            symbol=group[0].symbol,
            exchange=group[0].exchange,
            timeframe=group[0].timeframe
        )
```

### **Backtesting Capabilities**

**Strategy Implementation**:

- **Moving Average Crossover**: Simple trend-following strategy
- **Bollinger Bands**: Mean reversion strategy
- **RSI Strategy**: Momentum-based strategy
- **MACD Strategy**: Trend and momentum combination

**Backtesting Engine**:

```python
class BacktestingEngine:
    def __init__(self, data: List[OHLCVSchema], initial_capital: float = 10000):
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

    async def run_backtest(self, strategy: Strategy) -> BacktestResult:
        """Run backtest with specified strategy."""
        try:
            # Initialize strategy
            strategy.initialize(self.data)

            # Run simulation
            for i, bar in enumerate(self.data):
                # Update strategy with current bar
                signals = strategy.generate_signals(bar, i)

                # Execute signals
                await self._execute_signals(signals, bar, i)

                # Update equity curve
                current_equity = self._calculate_current_equity(bar)
                self.equity_curve.append({
                    "timestamp": bar.timestamp,
                    "equity": current_equity,
                    "capital": self.current_capital,
                    "positions_value": current_equity - self.current_capital
                })

            # Calculate performance metrics
            performance = self._calculate_performance_metrics()

            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_return=performance["total_return"],
                sharpe_ratio=performance["sharpe_ratio"],
                max_drawdown=performance["max_drawdown"],
                trades=self.trades,
                equity_curve=self.equity_curve
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    async def _execute_signals(self, signals: List[Signal], bar: OHLCVSchema, index: int):
        """Execute trading signals."""
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                await self._execute_buy(signal, bar, index)
            elif signal.signal_type == SignalType.SELL:
                await self._execute_sell(signal, bar, index)

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]["equity"]
            curr_equity = self.equity_curve[i]["equity"]
            returns.append((curr_equity - prev_equity) / prev_equity)

        if not returns:
            return {}

        # Calculate metrics
        total_return = (self.equity_curve[-1]["equity"] - self.initial_capital) / self.initial_capital
        avg_return = sum(returns) / len(returns)
        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak = self.equity_curve[0]["equity"]
        max_drawdown = 0

        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            drawdown = (peak - point["equity"]) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_return": total_return,
            "avg_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
```

## 📰 News Service

### **Core Functionality**

The News Service collects, processes, and stores financial news from multiple sources, providing comprehensive coverage of market-relevant information.

#### **1. Multi-Source News Collection**

**RSS Feed Collection**:

- **CoinDesk**: Cryptocurrency and blockchain news
- **CoinTelegraph**: Digital asset market coverage
- **Custom RSS Sources**: Configurable RSS feed sources

**API-Based Collection**:

- **Tavily Search**: AI-powered news search and aggregation
- **NewsAPI**: Comprehensive news coverage
- **Custom APIs**: Extensible API integration framework

**Implementation Details**:

```python
class RSSNewsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.parser = RSSParser()

    async def collect_from_source(self, source: str, keywords: List[str], max_articles: int) -> List[NewsArticle]:
        """Collect news from RSS source with keyword filtering."""
        try:
            # Get RSS feed URL
            feed_url = self.config["sources"][source]["url"]

            # Fetch RSS feed
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch RSS feed: {response.status}")

                content = await response.text()

            # Parse RSS content
            feed = self.parser.parse(content)

            # Filter articles by keywords
            filtered_articles = []
            for entry in feed.entries:
                if self._matches_keywords(entry, keywords):
                    article = NewsArticle(
                        title=entry.title,
                        content=entry.description,
                        summary=self._extract_summary(entry.description),
                        source=source,
                        published_at=entry.published,
                        url=entry.link,
                        tags=self._extract_tags(entry)
                    )
                    filtered_articles.append(article)

                    if len(filtered_articles) >= max_articles:
                        break

            return filtered_articles

        except Exception as e:
            logger.error(f"RSS collection failed for {source}: {e}")
            return []

    def _matches_keywords(self, entry, keywords: List[str]) -> bool:
        """Check if article matches any keywords."""
        text = f"{entry.title} {entry.description}".lower()
        return any(keyword.lower() in text for keyword in keywords)

    def _extract_summary(self, description: str) -> str:
        """Extract summary from description."""
        # Remove HTML tags and limit length
        clean_text = re.sub(r'<[^>]+>', '', description)
        return clean_text[:200] + "..." if len(clean_text) > 200 else clean_text

    def _extract_tags(self, entry) -> List[str]:
        """Extract tags from RSS entry."""
        tags = []

        # Extract from categories
        if hasattr(entry, 'tags'):
            tags.extend([tag.term for tag in entry.tags])

        # Extract from keywords
        if hasattr(entry, 'keywords'):
            tags.extend(entry.keywords.split(','))

        return [tag.strip() for tag in tags if tag.strip()]
```

#### **2. News Processing Pipeline**

**Content Processing**:

- **HTML Cleaning**: Remove HTML tags and formatting
- **Text Extraction**: Extract clean text content
- **Summary Generation**: Create article summaries
- **Tag Extraction**: Identify relevant topics and categories

**Duplicate Detection**:

```python
class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

    def detect_duplicates(self, articles: List[NewsArticle]) -> List[DuplicateGroup]:
        """Detect duplicate articles using TF-IDF similarity."""
        if len(articles) < 2:
            return []

        # Prepare text for vectorization
        texts = [f"{article.title} {article.summary}" for article in articles]

        # Create TF-IDF vectors
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        except ValueError:
            # Handle empty or invalid texts
            return []

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Group duplicates
        duplicate_groups = []
        processed = set()

        for i in range(len(articles)):
            if i in processed:
                continue

            group = [articles[i]]
            processed.add(i)

            for j in range(i + 1, len(articles)):
                if j in processed:
                    continue

                if similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(articles[j])
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(DuplicateGroup(
                    primary_article=group[0],
                    duplicate_articles=group[1:],
                    similarity_score=similarity_matrix[i][i+1] if len(group) > 1 else 1.0
                ))

        return duplicate_groups
```

#### **3. News Search & Retrieval**

**Full-Text Search**:

- **MongoDB Text Search**: Native MongoDB text search capabilities
- **Keyword Matching**: Exact and fuzzy keyword matching
- **Date Range Filtering**: Time-based article filtering
- **Source Filtering**: Filter by specific news sources

**Search Implementation**:

```python
class NewsSearchService:
    def __init__(self, repository: NewsRepository):
        self.repository = repository

    async def search_news(
        self,
        query: str,
        filters: NewsSearchFilters = None
    ) -> NewsSearchResult:
        """Search news articles with advanced filtering."""
        try:
            # Build search query
            search_query = self._build_search_query(query, filters)

            # Execute search
            articles = await self.repository.search_articles(search_query)

            # Apply post-processing filters
            filtered_articles = self._apply_post_filters(articles, filters)

            # Calculate relevance scores
            scored_articles = self._calculate_relevance_scores(filtered_articles, query)

            # Sort by relevance
            sorted_articles = sorted(scored_articles, key=lambda x: x.relevance_score, reverse=True)

            return NewsSearchResult(
                articles=sorted_articles,
                total_count=len(sorted_articles),
                search_time=time.time(),
                query=query,
                filters=filters
            )

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return NewsSearchResult(
                articles=[],
                total_count=0,
                search_time=time.time(),
                query=query,
                filters=filters,
                error=str(e)
            )

    def _build_search_query(self, query: str, filters: NewsSearchFilters) -> Dict[str, Any]:
        """Build MongoDB search query."""
        search_query = {}

        # Text search
        if query.strip():
            search_query["$text"] = {"$search": query}

        # Date filters
        if filters and filters.start_date:
            search_query["published_at"] = {"$gte": filters.start_date}

        if filters and filters.end_date:
            if "published_at" in search_query:
                search_query["published_at"]["$lte"] = filters.end_date
            else:
                search_query["published_at"] = {"$lte": filters.end_date}

        # Source filters
        if filters and filters.sources:
            search_query["source"] = {"$in": filters.sources}

        # Tag filters
        if filters and filters.tags:
            search_query["tags"] = {"$in": filters.tags}

        return search_query

    def _calculate_relevance_scores(self, articles: List[NewsArticle], query: str) -> List[ScoredArticle]:
        """Calculate relevance scores for search results."""
        scored_articles = []

        for article in articles:
            score = 0.0

            # Title relevance (highest weight)
            title_score = self._calculate_text_similarity(article.title, query)
            score += title_score * 0.5

            # Summary relevance
            summary_score = self._calculate_text_similarity(article.summary, query)
            score += summary_score * 0.3

            # Content relevance
            content_score = self._calculate_text_similarity(article.content, query)
            score += content_score * 0.2

            # Recency bonus
            days_old = (datetime.utcnow() - article.published_at).days
            if days_old <= 1:
                score += 0.1  # Recent articles get bonus
            elif days_old <= 7:
                score += 0.05  # Week-old articles get small bonus

            scored_articles.append(ScoredArticle(
                article=article,
                relevance_score=min(score, 1.0)  # Cap at 1.0
            ))

        return scored_articles
```

## 🧠 Sentiment Analysis Service

### **Core Functionality**

The Sentiment Analysis Service provides AI-powered sentiment analysis for financial news and market data, enabling sentiment-driven trading strategies.

#### **1. LLM-Based Sentiment Analysis**

**OpenAI Integration**:

- **GPT-3.5-turbo**: Primary sentiment analysis model
- **Structured Outputs**: Consistent sentiment classification
- **Context Awareness**: Financial domain-specific analysis
- **Batch Processing**: Efficient processing of multiple articles

**Implementation Details**:

```python
class OpenAISentimentAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)

    async def analyze_sentiment(
        self,
        text: str,
        context: str = None
    ) -> SentimentResult:
        """Analyze sentiment using OpenAI GPT model."""
        try:
            # Respect rate limits
            await self.rate_limiter.wait_if_needed()

            # Prepare prompt
            prompt = self._build_sentiment_prompt(text, context)

            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            sentiment_data = json.loads(content)

            return SentimentResult(
                text=text,
                sentiment=sentiment_data["sentiment"],
                confidence=sentiment_data["confidence"],
                score=sentiment_data["score"],
                reasoning=sentiment_data["reasoning"],
                model=self.model,
                analyzed_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            raise

    def _build_sentiment_prompt(self, text: str, context: str = None) -> str:
        """Build prompt for sentiment analysis."""
        prompt = f"""
        Analyze the sentiment of the following financial text:

        Text: {text}
        """

        if context:
            prompt += f"\nContext: {context}"

        prompt += """

        Provide your analysis in the following JSON format:
        {
            "sentiment": "positive|negative|neutral",
            "confidence": 0.0-1.0,
            "score": -1.0 to 1.0,
            "reasoning": "Brief explanation of your analysis"
        }

        Consider the financial implications and market sentiment when analyzing.
        """

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for consistent analysis."""
        return """
        You are a financial sentiment analysis expert. Your task is to analyze the sentiment
        of financial news, market commentary, and trading-related text. Consider:

        1. Market impact and trading implications
        2. Investor sentiment and market psychology
        3. Economic and financial context
        4. Tone and language used

        Provide consistent, reliable sentiment analysis that can be used for trading decisions.
        """
```

#### **2. Batch Processing & Optimization**

**Efficient Processing**:

- **Parallel Processing**: Concurrent analysis of multiple articles
- **Caching**: Cache analysis results to avoid redundant API calls
- **Rate Limiting**: Respect API rate limits with intelligent queuing
- **Error Handling**: Graceful degradation for failed analyses

**Batch Implementation**:

```python
class BatchSentimentAnalyzer:
    def __init__(self, analyzer: SentimentAnalyzer, max_concurrent: int = 5):
        self.analyzer = analyzer
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_batch(
        self,
        texts: List[str],
        contexts: List[str] = None
    ) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts concurrently."""
        if contexts is None:
            contexts = [None] * len(texts)

        # Create analysis tasks
        tasks = []
        for text, context in zip(texts, contexts):
            task = self._analyze_with_semaphore(text, context)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        sentiment_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Sentiment analysis failed for text {i}: {result}")
                # Create fallback result
                sentiment_results.append(SentimentResult(
                    text=texts[i],
                    sentiment="neutral",
                    confidence=0.0,
                    score=0.0,
                    reasoning=f"Analysis failed: {str(result)}",
                    model="fallback",
                    analyzed_at=datetime.utcnow()
                ))
            else:
                sentiment_results.append(result)

        return sentiment_results

    async def _analyze_with_semaphore(self, text: str, context: str = None) -> SentimentResult:
        """Analyze single text with semaphore control."""
        async with self.semaphore:
            return await self.analyzer.analyze_sentiment(text, context)
```

#### **3. Sentiment Aggregation & Analysis**

**Market Sentiment Metrics**:

- **Overall Sentiment**: Aggregate sentiment across multiple sources
- **Sentiment Trends**: Track sentiment changes over time
- **Source Reliability**: Weight sources based on historical accuracy
- **Market Impact**: Correlate sentiment with price movements

**Aggregation Implementation**:

```python
class SentimentAggregator:
    def __init__(self, repository: SentimentRepository):
        self.repository = repository

    async def calculate_market_sentiment(
        self,
        symbols: List[str],
        time_range: str = "24h"
    ) -> MarketSentimentResult:
        """Calculate overall market sentiment for specified symbols."""
        try:
            # Get sentiment data for time range
            end_time = datetime.utcnow()
            start_time = self._calculate_start_time(end_time, time_range)

            all_sentiments = []
            for symbol in symbols:
                sentiments = await self.repository.get_sentiments_for_symbol(
                    symbol, start_time, end_time
                )
                all_sentiments.extend(sentiments)

            if not all_sentiments:
                return MarketSentimentResult(
                    symbols=symbols,
                    time_range=time_range,
                    overall_sentiment="neutral",
                    sentiment_score=0.0,
                    confidence=0.0,
                    total_articles=0
                )

            # Calculate aggregate metrics
            sentiment_scores = [s.score for s in all_sentiments if s.score is not None]
            confidences = [s.confidence for s in all_sentiments if s.confidence is not None]

            if not sentiment_scores:
                return MarketSentimentResult(
                    symbols=symbols,
                    time_range=time_range,
                    overall_sentiment="neutral",
                    sentiment_score=0.0,
                    confidence=0.0,
                    total_articles=len(all_sentiments)
                )

            # Weighted average sentiment score
            weighted_score = sum(s * c for s, c in zip(sentiment_scores, confidences))
            total_confidence = sum(confidences)
            avg_sentiment_score = weighted_score / total_confidence if total_confidence > 0 else 0

            # Determine overall sentiment
            if avg_sentiment_score >= 0.1:
                overall_sentiment = "positive"
            elif avg_sentiment_score <= -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

            # Calculate confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return MarketSentimentResult(
                symbols=symbols,
                time_range=time_range,
                overall_sentiment=overall_sentiment,
                sentiment_score=avg_sentiment_score,
                confidence=avg_confidence,
                total_articles=len(all_sentiments),
                sentiment_distribution=self._calculate_sentiment_distribution(all_sentiments)
            )

        except Exception as e:
            logger.error(f"Market sentiment calculation failed: {e}")
            raise

    def _calculate_sentiment_distribution(self, sentiments: List[SentimentResult]) -> Dict[str, int]:
        """Calculate distribution of sentiment categories."""
        distribution = {"positive": 0, "negative": 0, "neutral": 0}

        for sentiment in sentiments:
            if sentiment.sentiment in distribution:
                distribution[sentiment.sentiment] += 1

        return distribution
```

## 🤖 AI Prediction Service

### **Core Functionality**

The AI Prediction Service provides machine learning-based time series forecasting for financial markets, enabling data-driven trading decisions.

#### **1. Model Training & Management**

**MLflow Integration**:

- **Experiment Tracking**: Comprehensive ML experiment management
- **Model Versioning**: Version control for trained models
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Model Registry**: Centralized model storage and deployment

**Training Pipeline**:

```python
class ModelTrainingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_client = MlflowClient()
        self.experiment_tracker = MLflowExperimentTracker(config["mlflow"])

    async def train_model(
        self,
        training_request: ModelTrainingRequest
    ) -> ModelTrainingResult:
        """Train a new prediction model."""
        try:
            # Start MLflow run
            with self.experiment_tracker.start_run(
                experiment_name=training_request.experiment_name
            ) as run:
                # Log parameters
                self.experiment_tracker.log_params({
                    "symbol": training_request.symbol,
                    "timeframe": training_request.timeframe,
                    "model_type": training_request.model_type,
                    "epochs": training_request.epochs,
                    "learning_rate": training_request.learning_rate,
                    "batch_size": training_request.batch_size
                })

                # Load and prepare data
                data = await self._load_training_data(training_request)
                train_data, val_data = self._split_data(data, training_request.validation_split)

                # Create and train model
                model = self._create_model(training_request.model_type, training_request)

                # Training loop
                training_history = await self._train_model(
                    model, train_data, val_data, training_request
                )

                # Evaluate model
                evaluation_metrics = await self._evaluate_model(model, val_data)

                # Log metrics
                self.experiment_tracker.log_metrics(evaluation_metrics)

                # Save model
                model_path = await self._save_model(model, training_request)

                # Log model artifacts
                self.experiment_tracker.log_artifact(model_path)

                return ModelTrainingResult(
                    success=True,
                    model_path=model_path,
                    run_id=run.info.run_id,
                    metrics=evaluation_metrics,
                    training_time=time.time()
                )

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return ModelTrainingResult(
                success=False,
                error=str(e),
                training_time=time.time()
            )

    async def _load_training_data(self, request: ModelTrainingRequest) -> pd.DataFrame:
        """Load training data for specified symbol and timeframe."""
        # Load OHLCV data
        data_loader = self._get_data_loader(request.data_source)
        raw_data = await data_loader.load_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Feature engineering
        features = self._engineer_features(raw_data)

        return features

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time series prediction."""
        features = data.copy()

        # Technical indicators
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()
        features['rsi'] = self._calculate_rsi(features['close'])
        features['macd'] = self._calculate_macd(features['close'])

        # Price changes
        features['price_change'] = features['close'].pct_change()
        features['price_change_5'] = features['close'].pct_change(periods=5)
        features['price_change_20'] = features['close'].pct_change(periods=20)

        # Volume features
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']

        # Time features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month

        # Remove NaN values
        features = features.dropna()

        return features
```

#### **2. Model Serving & Prediction**

**Multiple Serving Backends**:

- **Simple Serving**: Direct model loading for development
- **TorchScript**: Optimized PyTorch model serving
- **TorchServe**: Production-ready model serving
- **Triton**: NVIDIA's high-performance inference server

**Prediction Service**:

```python
class PredictionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_facade = ModelServingFacade(config["serving"])
        self.data_service = DataService(config["data"])

    async def make_prediction(
        self,
        prediction_request: PredictionRequest
    ) -> PredictionResult:
        """Make prediction using trained model."""
        try:
            # Load model
            model = await self.model_facade.load_model(
                model_id=prediction_request.model_id
            )

            # Prepare input data
            input_data = await self._prepare_prediction_input(prediction_request)

            # Make prediction
            prediction = await model.predict(input_data)

            # Post-process prediction
            processed_prediction = self._post_process_prediction(
                prediction, prediction_request
            )

            return PredictionResult(
                success=True,
                prediction=processed_prediction,
                model_id=prediction_request.model_id,
                prediction_time=datetime.utcnow(),
                confidence=processed_prediction.get("confidence", 0.0)
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                success=False,
                error=str(e),
                prediction_time=datetime.utcnow()
            )

    async def _prepare_prediction_input(self, request: PredictionRequest) -> np.ndarray:
        """Prepare input data for prediction."""
        # Load recent market data
        data = await self.data_service.get_recent_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            lookback_periods=request.lookback_periods
        )

        # Feature engineering
        features = self._engineer_features(data)

        # Normalize features
        normalized_features = self._normalize_features(features)

        # Reshape for model input
        input_shape = (1, normalized_features.shape[0], normalized_features.shape[1])
        return normalized_features.values.reshape(input_shape)

    def _post_process_prediction(self, prediction: np.ndarray, request: PredictionRequest) -> Dict[str, Any]:
        """Post-process raw prediction output."""
        # Convert prediction to interpretable format
        if request.output_type == "price":
            # Convert to price prediction
            current_price = request.current_price
            price_change = prediction[0][0]  # Assuming first output is price change

            predicted_price = current_price * (1 + price_change)

            return {
                "predicted_price": predicted_price,
                "price_change": price_change,
                "confidence": self._calculate_prediction_confidence(prediction),
                "direction": "up" if price_change > 0 else "down"
            }

        elif request.output_type == "classification":
            # Convert to classification output
            probabilities = softmax(prediction[0])
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)

            class_labels = ["sell", "hold", "buy"]

            return {
                "predicted_action": class_labels[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    label: float(prob) for label, prob in zip(class_labels, probabilities)
                }
            }

        else:
            return {
                "raw_prediction": prediction.tolist(),
                "confidence": 0.5
            }
```

## 🔧 Common Module Features

### **Core Utilities**

**Logging System**:

- **Structured Logging**: Consistent log format across services
- **Log Levels**: Configurable logging levels
- **Log Rotation**: Automatic log file management
- **Correlation IDs**: Request tracing across services

**Caching System**:

- **Multi-Level Caching**: In-memory and Redis caching
- **TTL Management**: Automatic cache expiration
- **Cache Invalidation**: Intelligent cache management
- **Fallback Strategies**: Graceful degradation when cache fails

**LLM Integration**:

- **Provider Abstraction**: Support for multiple LLM providers
- **Rate Limiting**: API rate limit management
- **Fallback Strategies**: Automatic provider switching
- **Response Caching**: Cache LLM responses for efficiency

## 📱 User Interface Features

### **API Endpoints**

**RESTful APIs**:

- **OpenAPI Documentation**: Comprehensive API documentation
- **Request Validation**: Pydantic-based input validation
- **Error Handling**: Consistent error response format
- **Rate Limiting**: Per-client rate limiting

**gRPC Services**:

- **High-Performance Communication**: Binary protocol for inter-service communication
- **Type Safety**: Protocol buffer definitions
- **Bidirectional Streaming**: Real-time data streaming
- **Service Discovery**: Dynamic service location

### **Monitoring & Management**

**Health Checks**:

- **Service Health**: Individual service health monitoring
- **Dependency Health**: Database and external service monitoring
- **Performance Metrics**: Response time and throughput monitoring
- **Alerting**: Automatic alerting for service issues

**Admin Interface**:

- **Service Management**: Start, stop, and restart services
- **Configuration Management**: Runtime configuration updates
- **Performance Monitoring**: Real-time performance metrics
- **Log Management**: Centralized log viewing and analysis

---

_The FinSight platform provides a comprehensive set of features that enable sophisticated financial analysis, from basic data collection to advanced AI-powered predictions. Each service is designed to be independently scalable while maintaining integration with the overall system architecture._
