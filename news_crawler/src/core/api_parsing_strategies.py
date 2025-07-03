# core/api_parsing_strategies.py

from typing import Dict, Any, List
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import dateutil.parser

from ..schemas.news_schemas import NewsItem, NewsSource


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
    }

    strategy = strategies.get(source)
    if not strategy:
        raise ValueError(f"No API parsing strategy found for source: {source}")

    return strategy
