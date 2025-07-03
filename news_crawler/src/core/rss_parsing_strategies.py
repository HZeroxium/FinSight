from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from dateutil import parser as date_parser

from ..schemas.news_schemas import NewsItem, NewsSource


class RSSParsingStrategy(ABC):
    """Abstract strategy for parsing RSS feeds"""

    @abstractmethod
    def parse_item(self, entry: Dict[str, Any], source: NewsSource) -> NewsItem:
        """
        Parse a single RSS entry into a NewsItem

        Args:
            entry: RSS feed entry from feedparser
            source: News source identifier

        Returns:
            Parsed NewsItem
        """
        pass

    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """Parse datetime string with fallback to current time"""
        if not date_str:
            return datetime.now()

        try:
            return date_parser.parse(date_str)
        except Exception:
            return datetime.now()

    def _extract_tags(self, entry: Dict[str, Any]) -> List[str]:
        """Extract tags from RSS entry"""
        tags = []

        # Try different tag fields
        if "tags" in entry:
            for tag in entry["tags"]:
                if isinstance(tag, dict) and "term" in tag:
                    tags.append(tag["term"])
                elif isinstance(tag, str):
                    tags.append(tag)

        # Try categories
        if "category" in entry:
            if isinstance(entry["category"], list):
                tags.extend(entry["category"])
            else:
                tags.append(entry["category"])

        return [tag.strip() for tag in tags if tag and tag.strip()]


class CoinDeskParsingStrategy(RSSParsingStrategy):
    """Parsing strategy for CoinDesk RSS feeds"""

    def parse_item(self, entry: Dict[str, Any], source: NewsSource) -> NewsItem:
        """Parse CoinDesk RSS entry"""

        # Extract author from dc:creator
        author = None
        if "author" in entry:
            author = entry["author"]
        elif "authors" in entry and entry["authors"]:
            author = entry["authors"][0].get("name")

        # Clean description
        description = entry.get("summary", entry.get("description"))

        # Extract tags
        tags = self._extract_tags(entry)

        # Build metadata with CoinDesk-specific fields
        metadata = {
            "content_encoded": entry.get("content"),
            "media_content": entry.get("media_content"),
            "guidislink": entry.get("guidislink"),
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return NewsItem(
            source=source,
            title=entry.get("title", ""),
            url=entry.get("link", ""),
            description=description,
            published_at=self._parse_datetime(entry.get("published")),
            author=author,
            guid=entry.get("id", entry.get("guid")),
            tags=tags,
            metadata=metadata,
        )


class CoinTelegraphParsingStrategy(RSSParsingStrategy):
    """Parsing strategy for CoinTelegraph RSS feeds"""

    def parse_item(self, entry: Dict[str, Any], source: NewsSource) -> NewsItem:
        """Parse CoinTelegraph RSS entry"""

        # Extract author from dc:creator with CoinTelegraph format
        author = None
        if "author" in entry:
            author = entry["author"]
            # CoinTelegraph format: "Cointelegraph by Author Name"
            if author and "by " in author:
                author = author.split("by ")[-1].strip()

        # Clean description
        description = entry.get("summary", entry.get("description"))

        # Extract tags
        tags = self._extract_tags(entry)

        # Build metadata with CoinTelegraph-specific fields
        metadata = {
            "media_content": entry.get("media_content"),
            "enclosure": entry.get("enclosures"),
            "guidislink": entry.get("guidislink"),
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return NewsItem(
            source=source,
            title=entry.get("title", ""),
            url=entry.get("link", ""),
            description=description,
            published_at=self._parse_datetime(entry.get("published")),
            author=author,
            guid=entry.get("id", entry.get("guid")),
            tags=tags,
            metadata=metadata,
        )


class DefaultRSSParsingStrategy(RSSParsingStrategy):
    """Default parsing strategy for generic RSS feeds"""

    def parse_item(self, entry: Dict[str, Any], source: NewsSource) -> NewsItem:
        """Parse generic RSS entry"""

        # Extract author
        author = None
        if "author" in entry:
            author = entry["author"]
        elif "authors" in entry and entry["authors"]:
            author = entry["authors"][0].get("name")

        # Clean description
        description = entry.get("summary", entry.get("description"))

        # Extract tags
        tags = self._extract_tags(entry)

        # Build metadata with all available fields
        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in [
                "title",
                "link",
                "summary",
                "description",
                "published",
                "author",
                "id",
                "guid",
            ]
            and v is not None
        }

        return NewsItem(
            source=source,
            title=entry.get("title", ""),
            url=entry.get("link", ""),
            description=description,
            published_at=self._parse_datetime(entry.get("published")),
            author=author,
            guid=entry.get("id", entry.get("guid")),
            tags=tags,
            metadata=metadata,
        )


def get_parsing_strategy(source: NewsSource) -> RSSParsingStrategy:
    """
    Get parsing strategy for a specific news source

    Args:
        source: News source identifier

    Returns:
        Appropriate parsing strategy
    """
    strategies = {
        NewsSource.COINDESK: CoinDeskParsingStrategy(),
        NewsSource.COINTELEGRAPH: CoinTelegraphParsingStrategy(),
    }

    return strategies.get(source, DefaultRSSParsingStrategy())
