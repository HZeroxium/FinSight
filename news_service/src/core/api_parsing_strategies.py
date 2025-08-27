# core/api_parsing_strategies.py

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List

import dateutil.parser
from common.logger import LoggerFactory, LoggerType, LogLevel

from ..schemas.news_schemas import NewsItem, NewsSource

logger = LoggerFactory.get_logger(
    name="api-parsing-strategies",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/api_parsing_strategies.log",
)


class APIParsingStrategy(ABC):
    """Abstract base class for API response parsing strategies"""

    @abstractmethod
    def parse_response(
        self, response_data: Dict[str, Any], source: NewsSource
    ) -> List[NewsItem]:
        """Parse API response data into NewsItem objects"""
        pass


class CoinTelegraphAPIStrategy(APIParsingStrategy):
    """Parsing strategy for CoinTelegraph GraphQL API responses"""

    def parse_response(
        self, response_data: Dict[str, Any], source: NewsSource
    ) -> List[NewsItem]:
        """
        Parse CoinTelegraph GraphQL response into NewsItem objects

        Args:
            response_data: GraphQL response data
            source: News source

        Returns:
            List[NewsItem]: Parsed news items
        """
        items = []

        try:
            # Navigate to posts data
            locale = response_data.get("data", {}).get("locale", {})
            category = locale.get("category", {})
            posts = category.get("posts", {})
            post_data = posts.get("data", [])

            for post in post_data:
                try:
                    post_translate = post.get("postTranslate", {})

                    # Extract basic information
                    title = post_translate.get("title", "").strip()
                    if not title:
                        continue

                    slug = post.get("slug", "")
                    url = f"https://cointelegraph.com/news/{slug}"

                    description = post_translate.get("leadText", "").strip()

                    # Parse publication date
                    published_str = post_translate.get("published", "")
                    published_at = self._parse_date(published_str)

                    # Extract author
                    author_info = post_translate.get("author", {})
                    author_translates = author_info.get("authorTranslates", [])
                    author = (
                        author_translates[0].get("name") if author_translates else None
                    )

                    # Extract tags/categories
                    tags = []
                    category_info = post.get("category", {})
                    category_translates = category_info.get("categoryTranslates", [])
                    if category_translates:
                        tags.append(category_translates[0].get("title", ""))

                    post_badge = post.get("postBadge", {})
                    badge_translates = post_badge.get("postBadgeTranslates", [])
                    if badge_translates:
                        tags.append(badge_translates[0].get("title", ""))

                    # Remove empty tags
                    tags = [tag for tag in tags if tag]

                    # Create metadata
                    metadata = {
                        "post_id": post.get("id"),
                        "views": post.get("views", 0),
                        "avatar": post_translate.get("avatar"),
                        "published_human": post_translate.get("publishedHumanFormat"),
                        "show_shares": post.get("showShares", False),
                        "show_stats": post.get("showStats", False),
                    }

                    news_item = NewsItem(
                        source=source,
                        title=title,
                        url=url,
                        description=description,
                        published_at=published_at,
                        author=author,
                        guid=post.get("id"),  # Use post ID as GUID
                        tags=tags,
                        metadata=metadata,
                    )

                    items.append(news_item)

                except Exception as e:
                    # Skip malformed posts but continue processing
                    continue

        except Exception as e:
            raise ValueError(f"Failed to parse CoinTelegraph API response: {e}")

        return items

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            return dateutil.parser.parse(date_str).replace(tzinfo=timezone.utc)
        except:
            return datetime.now(timezone.utc)


class CoinDeskAPIStrategy(APIParsingStrategy):
    """Parsing strategy for CoinDesk REST API responses"""

    def parse_response(
        self, response_data: Dict[str, Any], source: NewsSource
    ) -> List[NewsItem]:
        """
        Parse CoinDesk API response into NewsItem objects

        Args:
            response_data: CoinDesk API response data
            source: News source

        Returns:
            List[NewsItem]: Parsed news items
        """
        items: List[NewsItem] = []

        try:
            # Get articles from Data array
            articles = response_data.get("Data", [])

            for article in articles:
                try:
                    # Extract basic information
                    title = (article.get("TITLE") or "").strip()
                    if not title:
                        continue

                    url = (article.get("URL") or "").strip()
                    if not url:
                        continue

                    # Parse publication date from Unix timestamp
                    published_ts = article.get("PUBLISHED_ON")
                    if published_ts:
                        published_at = datetime.fromtimestamp(
                            published_ts, tz=timezone.utc
                        )
                    else:
                        published_at = datetime.now(timezone.utc)

                    # Extract description/body safely
                    subtitle = article.get("SUBTITLE") or ""
                    description = subtitle.strip()
                    if not description:
                        description = (article.get("BODY") or "").strip()

                    # Extract author
                    author = (article.get("AUTHORS") or "").strip() or None

                    # Extract and process tags from keywords and categories
                    tags: List[str] = []
                    # Keywords
                    for k in (
                        (article.get("KEYWORDS") or "").replace("|", ",").split(",")
                    ):
                        k = k.strip()
                        if k:
                            tags.append(k)
                    # Categories
                    for cat in article.get("CATEGORY_DATA", []):
                        name = (cat.get("NAME") or "").strip()
                        if name:
                            tags.append(name)
                    # Remove duplicates, preserve order
                    tags = list(dict.fromkeys(tags))

                    # Build metadata
                    metadata: Dict[str, Any] = {
                        "article_id": article.get("ID"),
                        "guid_original": article.get("GUID"),
                        "published_on_ns": article.get("PUBLISHED_ON_NS"),
                        "image_url": article.get("IMAGE_URL"),
                        "subtitle": article.get("SUBTITLE"),
                        "source_id": article.get("SOURCE_ID"),
                        "body": article.get("BODY"),
                        "keywords": article.get("KEYWORDS"),
                        "language": article.get("LANG", "EN"),
                        "upvotes": article.get("UPVOTES", 0),
                        "downvotes": article.get("DOWNVOTES", 0),
                        "score": article.get("SCORE", 0),
                        "sentiment": article.get("SENTIMENT"),
                        "status": article.get("STATUS", "ACTIVE"),
                        "created_on": article.get("CREATED_ON"),
                        "updated_on": article.get("UPDATED_ON"),
                        "created_by": article.get("CREATED_BY"),
                        "updated_by": article.get("UPDATED_BY"),
                        "created_by_username": article.get("CREATED_BY_USERNAME"),
                        "updated_by_username": article.get("UPDATED_BY_USERNAME"),
                    }

                    # Source info
                    source_data = article.get("SOURCE_DATA", {})
                    if source_data:
                        metadata["source_info"] = {
                            "source_key": source_data.get("SOURCE_KEY"),
                            "name": source_data.get("NAME"),
                            "image_url": source_data.get("IMAGE_URL"),
                            "source_url": source_data.get("URL"),
                            "source_type": source_data.get("SOURCE_TYPE"),
                            "launch_date": source_data.get("LAUNCH_DATE"),
                            "benchmark_score": source_data.get("BENCHMARK_SCORE"),
                            "last_updated_ts": source_data.get("LAST_UPDATED_TS"),
                        }
                    # Categories detail
                    if article.get("CATEGORY_DATA"):
                        metadata["categories"] = [
                            {
                                "id": cat.get("ID"),
                                "name": cat.get("NAME"),
                                "category": cat.get("CATEGORY"),
                            }
                            for cat in article["CATEGORY_DATA"]
                            if cat.get("ID") is not None
                        ]

                    # Clean out None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    news_item = NewsItem(
                        source=source,
                        title=title,
                        url=url,
                        description=description or None,
                        published_at=published_at,
                        author=author,
                        guid=article.get("GUID"),
                        tags=tags or None,
                        metadata=metadata,
                    )
                    items.append(news_item)

                except Exception:
                    # Skip this article but continue
                    continue

        except Exception as e:
            raise ValueError(f"Failed to parse CoinDesk API response: {e}")

        return items


def get_api_parsing_strategy(source: NewsSource) -> APIParsingStrategy:
    """
    Get appropriate parsing strategy for the given source

    Args:
        source: News source

    Returns:
        APIParsingStrategy: Parsing strategy instance
    """
    strategies = {
        NewsSource.COINTELEGRAPH: CoinTelegraphAPIStrategy(),
        NewsSource.COINDESK: CoinDeskAPIStrategy(),
    }

    strategy = strategies.get(source)
    if not strategy:
        raise ValueError(f"No API parsing strategy found for source: {source}")

    return strategy
